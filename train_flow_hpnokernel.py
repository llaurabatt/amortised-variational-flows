"""Flow model for the LALME model."""
import os
import math
import pathlib

from absl import logging
import syne_tune

import numpy as np
import arviz as az
from arviz import InferenceData

from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

from tensorflow_probability.substrates import jax as tfp

from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, initial_state_ckpt, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict,
                                      IntLike, List, Optional, PRNGKey,
                                      Sequence, SmiEta, Tuple, Union)

import log_prob_fun_test
from log_prob_fun_test import ModelParamsGlobal, ModelParamsLocations, PriorHparams
import data
import flows
import plot
from misc import issymmetric

kernels = tfp.math.psd_kernels

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def load_data(prng_key: PRNGKey, config: ConfigDict) -> Dict[str, Array]:
  """Load LALME data."""

  prng_seq = hk.PRNGSequence(prng_key)

  lalme_dataset = data.load_lalme(dataset_id=config.dataset_id)

  # Get locations bounds
  # loc_bounds = jnp.array([[0., 1.], [0., 1.]])
  loc_bounds = jnp.stack(
      [lalme_dataset['loc'].min(axis=0), lalme_dataset['loc'].max(axis=0)],
      axis=1).astype(float)
  # Shift x's and y's to start on zero
  loc_bounds = loc_bounds - loc_bounds[:, [0]]
  # scale to the unit square, preserving relative size of axis
  loc_scale = 1 / loc_bounds.max()
  loc_bounds = loc_bounds * loc_scale

  # Select Linguistic Profiles to use
  assert config.num_lp_anchor_train > 0
  assert (config.num_lp_anchor_train + config.num_lp_anchor_val +
          config.num_lp_anchor_test <= lalme_dataset['num_profiles_anchor'])
  assert config.num_lp_floating_train > 0

  ### Choose profiles to use ###
  ## Anchor profiles
  # Start with all the anchor profiles
  lp_anchor_ = lalme_dataset['LP'][:lalme_dataset['num_profiles_anchor']]
  # Choose training anchor profiles
  lp_anchor_train = jax.random.choice(
      next(prng_seq),
      a=lp_anchor_,
      shape=(config.num_lp_anchor_train,),
      replace=False)
  lp_anchor_ = jnp.setdiff1d(lp_anchor_, lp_anchor_train)
  lp_anchor_train = jnp.sort(lp_anchor_train).tolist()
  if config.num_lp_anchor_val > 0:
    # Choose validation anchor profiles
    lp_anchor_val = jax.random.choice(
        next(prng_seq),
        a=lp_anchor_,
        shape=(config.num_lp_anchor_val,),
        replace=False)
    lp_anchor_ = jnp.setdiff1d(lp_anchor_, lp_anchor_val)
    lp_anchor_val = jnp.sort(lp_anchor_val).tolist()
  else:
    lp_anchor_val = None
  if config.num_lp_anchor_test > 0:
    # Choose test anchor profiles
    lp_anchor_test = jax.random.choice(
        next(prng_seq),
        a=lp_anchor_,
        shape=(config.num_lp_anchor_test,),
        replace=False)
    lp_anchor_test = jnp.sort(lp_anchor_test).tolist()
  else:
    lp_anchor_test = None

  ## Floating profiles
  lp_floating_ = lalme_dataset['LP'][lalme_dataset['num_profiles_anchor']:]
  # Choose training floating profiles
  if 'lp_floating_train' not in config or len(config.lp_floating_train) == 0:
    lp_floating_train = jax.random.choice(
        next(prng_seq),
        a=lp_floating_,
        shape=(config.num_lp_floating_train,),
        replace=False)
  else:
    assert len(config.lp_floating_train) == config.num_lp_floating_train
    lp_floating_train = config.lp_floating_train

  lp_floating_train = np.sort(lp_floating_train).tolist()

  ### Choose items to use ###
  # Start with all the items
  items_keep = lalme_dataset['items'][jax.random.choice(
      next(prng_seq),
      len(lalme_dataset['items']),
      shape=(config.num_items_keep,),
      replace=False)]
  items_keep = np.sort(items_keep).tolist()

  # Process data
  lalme_dataset_out = data.process_lalme(
      lalme_dataset=lalme_dataset,
      lp_anchor_train=lp_anchor_train,
      lp_floating_train=lp_floating_train,
      items_keep=items_keep,
      loc_bounds=loc_bounds,
      lp_anchor_val=lp_anchor_val,
      lp_anchor_test=lp_anchor_test,
      remove_empty_forms=config.remove_empty_forms,
  )

  return lalme_dataset_out


def make_optimizer(
    lr_schedule_name,
    lr_schedule_kwargs,
    grad_clip_value,
) -> optax.GradientTransformation:
  """Define optimizer to train the VHP map."""
  schedule = getattr(optax, lr_schedule_name)(**lr_schedule_kwargs)

  optimizer = optax.chain(*[
      optax.clip_by_global_norm(max_norm=grad_clip_value),
      optax.adabelief(learning_rate=schedule),
  ])
  return optimizer


def get_inducing_points(
    dataset: Batch,
    inducing_grid_shape: Tuple[int, int],
    kernel_name: str,
    kernel_kwargs: Dict[str, Any],
    gp_jitter: float,
) -> Dict[str, Array]:
  """Define grid of inducing point for GPs."""
  dataset = dataset.copy()

  num_inducing_points = math.prod(inducing_grid_shape)

  # Inducing points are defined as a grid on the unit square
  loc_inducing = jnp.meshgrid(
      jnp.linspace(0, 1, inducing_grid_shape[0]),
      jnp.linspace(0, 1, inducing_grid_shape[1]))
  loc_inducing = jnp.stack(loc_inducing, axis=-1).reshape(-1, 2)
  dataset['loc_inducing'] = loc_inducing
  # Compute GP covariance between inducing values
  dataset['cov_inducing'] = getattr(kernels,
                                    kernel_name)(**kernel_kwargs).matrix(
                                        x1=dataset['loc_inducing'],
                                        x2=dataset['loc_inducing'],
                                    )

  # Add jitter
  dataset['cov_inducing'] = dataset['cov_inducing'] + gp_jitter * jnp.eye(
      num_inducing_points)
  # Check that the covarince is symmetric
  assert issymmetric(
      dataset['cov_inducing']), 'Covariance Matrix is not symmetric'

  # Cholesky factor of covariance
  dataset['cov_inducing_chol'] = jnp.linalg.cholesky(dataset['cov_inducing'])

  # Inverse of covariance of inducing values
  # dataset['cov_inducing_inv'] = jnp.linalg.inv(dataset['cov_inducing'])
  cov_inducing_chol_inv = jax.scipy.linalg.solve_triangular(
      a=dataset['cov_inducing_chol'],
      b=jnp.eye(num_inducing_points),
      lower=True,
  )
  dataset['cov_inducing_inv'] = jnp.matmul(
      cov_inducing_chol_inv.T, cov_inducing_chol_inv, precision='highest')

  # Check that the inverse is symmetric
  assert issymmetric(
      dataset['cov_inducing_inv']), 'Covariance Matrix is not symmetric'
  # Check that there are no NaNs
  assert ~jnp.any(jnp.isnan(dataset['cov_inducing_inv']))
  # Cross covariance between anchor and inducing values
  dataset['cov_anchor_inducing'] = getattr(
      kernels, kernel_name)(**kernel_kwargs).matrix(
          x1=dataset['loc'][:dataset['num_profiles_anchor'], :],
          x2=dataset['loc_inducing'],
      )

  return dataset


def q_distr_global(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    sample_shape: Union[IntLike, Sequence[IntLike]],
) -> Dict[str, Any]:
  """Sample from the posterior of the LALME model.

  The posterior distribution is factorized as:
    q(mu, zeta, Gamma_U, a, W) q(loc_floating | Gamma_U, a, W)

  The first term is the distribution of "global" parameters, and the second term
  is the posterior for locations of floating profiles.

  We make this separation so we are able to use trained posteriors for the
  global parameters even after changing the number of floating profiles.

  Args:
    flow_name: String specifying the type of flow to be used.
    flow_kwargs: Dictionary with keyword arguments for the flow function.
    kernel_name: String specifiying the Kernel function to be used.
    kernel_kwargs: Dictionary with keyword arguments for the Kernel function.
    sample_shape: Sample shape.
    batch: Data batch.
  """

  # params = state.params
  # prng_key=next(prng_seq)
  # batch = train_ds
  # flow_sample, flow_log_prob = q_distr_global_sample.apply(
  #      params, prng_key, sample_shape=(config.num_samples_elbo,), batch=batch)

  q_distr_out = {}

  ## Posterior for global parameters ##
  # Define normalizing flow
  q_distr = getattr(flows, flow_name + '_global_params')(**flow_kwargs)
  # Sample from flow
  (sample_flow_concat, sample_logprob,
   sample_base) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=sample_shape,
   )

  # Split flow into model parameters
  q_distr_out['sample'] = flows.split_flow_global_params(
      samples=sample_flow_concat,
      **flow_kwargs,
  )

  # Log_probabilities of the sample
  q_distr_out['sample_logprob'] = sample_logprob

  # Sample from base distribution are preserved
  # These are used for the posterior of profiles locations
  q_distr_out['sample_base'] = sample_base

  return q_distr_out


def q_distr_loc_floating(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    global_params_base_sample: Array,
    name: str = 'loc_floating',
) -> Dict[str, Any]:
  """Sample from the posterior of floating locations

  Conditional on global parameters.

  The posterior distribution is factorized as:
    q(mu, zeta, Gamma_U, a, W) q(loc_floating | Gamma_U, a, W)

  The first term is the distribution of "global" parameters, and the second term
  is the posterior for locations of floating profiles.

  We make this separation so we are able to use trained posteriors for the
  global parameters even after changing the number of floating profiles.

  Args:
    flow_name: String specifying the type of flow to be used.
    flow_kwargs: Dictionary with keyword arguments for the flow function.
    kernel_name: String specifiying the Kernel function to be used.
    kernel_kwargs: Dictionary with keyword arguments for the Kernel function.
    sample_shape: Sample shape.
    batch: Data batch.
  """

  # params = state.params
  # prng_key=next(prng_seq)
  # batch = train_ds
  # flow_sample, flow_log_prob = q_distr_global_sample.apply(
  #      params, prng_key, sample_shape=(config.num_samples_elbo,), batch=batch)

  q_distr_out = {}

  ## Posterior for locations of floating profiles ##
  # Define normalizing flow
  q_distr = getattr(flows, flow_name + '_locations')(
      num_profiles=flow_kwargs['num_profiles_floating'], **flow_kwargs)

  # Sample from flow
  num_samples = global_params_base_sample.shape[0]
  sample_flow_concat, sample_logprob = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
      context=global_params_base_sample,
  )

  # Split flow into model parameters
  # (and add to existing posterior_sample_dict)
  q_distr_out['sample'] = flows.split_flow_locations(
      samples=sample_flow_concat,
      num_profiles=flow_kwargs['num_profiles_floating'],
      name=name,
  )

  q_distr_out['sample_logprob'] = sample_logprob

  return q_distr_out


def q_distr_loc_random_anchor(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    global_params_base_sample: Array,
) -> Dict[str, Any]:
  """Sample from the posterior of floating locations

  Conditional on global parameters.

  The posterior distribution is factorized as:
    q(mu, zeta, Gamma_U, a, W) q(loc_floating | Gamma_U, a, W)

  The first term is the distribution of "global" parameters, and the second term
  is the posterior for locations of floating profiles.

  We make this separation so we are able to use trained posteriors for the
  global parameters even after changing the number of floating profiles.

  Args:
    flow_name: String specifying the type of flow to be used.
    flow_kwargs: Dictionary with keyword arguments for the flow function.
    kernel_name: String specifiying the Kernel function to be used.
    kernel_kwargs: Dictionary with keyword arguments for the Kernel function.
    sample_shape: Sample shape.
    batch: Data batch.
  """

  q_distr_out = {}

  ## Posterior for locations of anchor profiles treated as unkbowb locations##

  # Define normalizing flow
  q_distr = getattr(flows, flow_name + '_locations')(
      num_profiles=flow_kwargs['num_profiles_anchor'], **flow_kwargs)

  # Sample from flow
  num_samples = global_params_base_sample.shape[0]
  (sample_flow_concat, sample_logprob) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
      context=global_params_base_sample,
  )

  # Split flow into model parameters
  # (and add to existing posterior_sample_dict)
  q_distr_out['sample'] = flows.split_flow_locations(
      samples=sample_flow_concat,
      num_profiles=flow_kwargs['num_profiles_anchor'],
      name='loc_random_anchor',
  )

  # Log_probabilities of the sample
  q_distr_out['sample_logprob'] = sample_logprob

  return q_distr_out


def sample_all_flows(
    params_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    sample_shape: Union[IntLike, Sequence[IntLike]],
    include_random_anchor: bool,
) -> Dict[str, Any]:
  """Generate a sample from the entire flow posterior."""

  prng_seq = hk.PRNGSequence(prng_key)

  q_distr_out = {}

  # Global parameters
  q_distr_out_global = hk.transform(q_distr_global).apply(
      params_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=sample_shape,
  )
  q_distr_out.update({f"global_{k}": v for k, v in q_distr_out_global.items()})

  # Floating profiles locations
  q_distr_out_loc_floating = hk.transform(q_distr_loc_floating).apply(
      params_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      global_params_base_sample=q_distr_out_global['sample_base'],
      name='loc_floating',
  )
  q_distr_out['locations_sample'] = q_distr_out_loc_floating['sample']
  q_distr_out['loc_floating_logprob'] = q_distr_out_loc_floating[
      'sample_logprob']

  if flow_kwargs.is_smi:
    # Auxiliary Floating profiles locations
    q_distr_out_loc_floating_aux = hk.transform(q_distr_loc_floating).apply(
        params_tuple[2],
        next(prng_seq),
        flow_name=flow_name,
        flow_kwargs=flow_kwargs,
        global_params_base_sample=q_distr_out_global['sample_base'],
        name='loc_floating_aux',
    )
    q_distr_out['locations_sample'] = ModelParamsLocations(
        loc_floating=q_distr_out_loc_floating['sample'].loc_floating,
        loc_floating_aux=q_distr_out_loc_floating_aux['sample']
        .loc_floating_aux,
    )
    q_distr_out['loc_floating_aux_logprob'] = q_distr_out_loc_floating_aux[
        'sample_logprob']

  # Anchor profiles locations
  if include_random_anchor:
    q_distr_out_loc_random_anchor = hk.transform(
        q_distr_loc_random_anchor).apply(
            params_tuple[3],
            next(prng_seq),
            flow_name=flow_name,
            flow_kwargs=flow_kwargs,
            global_params_base_sample=q_distr_out_global['sample_base'],
        )
    q_distr_out['locations_sample'] = ModelParamsLocations(
        loc_floating=q_distr_out_loc_floating['sample'].loc_floating,
        loc_floating_aux=q_distr_out_loc_floating_aux['sample']
        .loc_floating_aux,
        loc_random_anchor=q_distr_out_loc_random_anchor['sample']
        .loc_random_anchor,
    )
    q_distr_out['loc_random_anchor_logprob'] = q_distr_out_loc_random_anchor[
        'sample_logprob']

  return q_distr_out


def sample_lalme_az(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    config: ConfigDict,
    lalme_dataset: Dict[str, Any],
    include_gamma: bool = False,
    num_samples_chunk: int = 1_000,
) -> InferenceData:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  global_sample = []
  locations_sample = []
  gamma_sample = []

  for _ in range(config.num_samples_plot // num_samples_chunk):
    # Sample from variational posterior
    q_distr_out = sample_all_flows(
        params_tuple=[state.params for state in state_list],
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        sample_shape=(num_samples_chunk,),
        include_random_anchor=config.include_random_anchor,
    )

    global_sample.append(q_distr_out['global_sample'])
    locations_sample.append(q_distr_out['locations_sample'])

    if include_gamma:
      # Get a sample of the basis GPs on profiles locations
      # conditional on values at the inducing locations.
      gamma_sample_, _ = jax.vmap(
          lambda key_, global_, locations_: log_prob_fun_test.
          sample_gamma_profiles_given_gamma_inducing(
              batch=batch,
              model_params_global=global_,
              model_params_locations=locations_,
              prng_key=key_,
              kernel_name=config.kernel_name,
              kernel_kwargs=config.kernel_kwargs,
              gp_jitter=config.gp_jitter,
              include_random_anchor=config.include_random_anchor,
          ))(
              jax.random.split(next(prng_seq), num_samples_chunk),
              q_distr_out['global_sample'],
              q_distr_out['locations_sample'],
          )
      gamma_sample.append(gamma_sample_)

  global_sample = jax.tree_map(  # pylint: disable=no-value-for-parameter
      lambda *x: jnp.concatenate([xi[None, ...] for xi in x], axis=1),
      *global_sample)
  locations_sample = jax.tree_map(  # pylint: disable=no-value-for-parameter
      lambda *x: jnp.concatenate([xi[None, ...] for xi in x], axis=1),
      *locations_sample)
  if include_gamma:
    gamma_sample = jax.tree_map(  # pylint: disable=no-value-for-parameter
        lambda *x: jnp.concatenate([xi[None, ...] for xi in x], axis=1),
        *gamma_sample)
  else:
    gamma_sample = None

  ### Posterior visualisation with Arviz

  # Create InferenceData object
  lalme_az = plot.lalme_az_from_samples(
      lalme_dataset=lalme_dataset,
      model_params_global=global_sample,
      model_params_locations=locations_sample,
      model_params_gamma=gamma_sample,
  )

  return lalme_az


def logprob_lalme(
    batch: Batch,
    prng_key: PRNGKey,
    model_params_global: ModelParamsGlobal,
    model_params_locations: ModelParamsLocations,
    prior_hparams: Dict[str, Any],
    kernel_name: str,
    kernel_kwargs: Dict[str, Any],
    num_samples_gamma_profiles: int,
    smi_eta_profiles: Optional[Array],
    gp_jitter: float = 1e-5,
    random_anchor: bool = False,
):
  """Joint log probability of the LALME model.
  Using unbounded input parameters.
  """

  # Put Smi eta values into a dictionary.
  if smi_eta_profiles is not None:
    smi_eta = {
        'profiles': smi_eta_profiles,
        'items': jnp.ones(len(batch['num_forms_tuple']))
    }
  else:
    smi_eta = None

  # Sample the basis GPs on profiles locations conditional on GP values on the
  # inducing points.
  model_params_gamma_profiles_sample, gamma_profiles_logprob_sample = jax.vmap(
      lambda key_: log_prob_fun_test.sample_gamma_profiles_given_gamma_inducing(
          batch=batch,
          model_params_global=model_params_global,
          model_params_locations=model_params_locations,
          prng_key=key_,
          kernel_name=kernel_name,
          kernel_kwargs=kernel_kwargs,
          gp_jitter=gp_jitter,
          include_random_anchor=random_anchor,
      ))(
          jax.random.split(prng_key, num_samples_gamma_profiles))

  # Average joint logprob across samples of gamma_profiles
  log_prob = jax.vmap(lambda gamma_profiles_, gamma_profiles_logprob_:
                      log_prob_fun_test.logprob_joint(
                          batch=batch,
                          model_params_global=model_params_global,
                          model_params_locations=model_params_locations,
                          model_params_gamma_profiles=gamma_profiles_,
                          gamma_profiles_logprob=gamma_profiles_logprob_,
                          smi_eta=smi_eta,
                          random_anchor=random_anchor,
                          prior_hparams=prior_hparams,
                        #   **prior_hparams,
                      ))(model_params_gamma_profiles_sample,
                         gamma_profiles_logprob_sample)
  log_prob = jnp.mean(log_prob)

  return log_prob


def elbo_estimate(
    params_tuple: Tuple[hk.Params],
    batch: Optional[Batch],
    prng_key: PRNGKey,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    smi_eta: Optional[SmiEta],
    include_random_anchor: bool,
    prior_hparams: Dict[str, Any],
    kernel_name: Optional[str] = None,
    kernel_kwargs: Optional[Dict[str, Any]] = None,
    num_samples_gamma_profiles: int = 100,
    gp_jitter: Optional[float] = None,
) -> Dict[str, Array]:
  """Estimate the ELBO.

  Monte Carlo Estimate of the evidence lower-bound.
  """
  # params_tuple = [state.params for state in state_list]
  prng_seq = hk.PRNGSequence(prng_key)

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      prng_key=next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples,),
      include_random_anchor=include_random_anchor,
  )

  is_smi = False if smi_eta is None else True

  # ELBO stage 1: Power posterior
  if is_smi:
    locations_stg1_ = ModelParamsLocations(
        loc_floating=q_distr_out['locations_sample'].loc_floating_aux,
        loc_floating_aux=None,
        loc_random_anchor=None,
    )
    log_prob_joint_stg1 = jax.vmap(
        lambda key_, global_, locations_: logprob_lalme(
            batch=batch,
            prng_key=key_,
            model_params_global=global_,
            model_params_locations=locations_,
            prior_hparams=prior_hparams,
            kernel_name=kernel_name,
            kernel_kwargs=kernel_kwargs,
            num_samples_gamma_profiles=num_samples_gamma_profiles,
            smi_eta_profiles=smi_eta['profiles'],
            gp_jitter=gp_jitter,
            random_anchor=False,
        ))(
            jax.random.split(next(prng_seq), num_samples),
            q_distr_out['global_sample'],
            locations_stg1_,
        )

    log_q_stg1 = (
        q_distr_out['global_sample_logprob'] +
        q_distr_out['loc_floating_aux_logprob'])

    elbo_stg1 = log_prob_joint_stg1 - log_q_stg1
  else:
    elbo_stg1 = 0.

  # ELBO stage 2: Refit locations floating profiles
  locations_stg2_ = ModelParamsLocations(
      loc_floating=q_distr_out['locations_sample'].loc_floating,
      loc_floating_aux=None,
      loc_random_anchor=None,
  )
  log_prob_joint_stg2 = jax.vmap(
      lambda key_, global_, locations_: logprob_lalme(
          batch=batch,
          prng_key=key_,
          model_params_global=global_,
          model_params_locations=locations_,
          prior_hparams=prior_hparams,
          kernel_name=kernel_name,
          kernel_kwargs=kernel_kwargs,
          num_samples_gamma_profiles=num_samples_gamma_profiles,
          smi_eta_profiles=None,
          gp_jitter=gp_jitter,
          random_anchor=False,
      ))(
          jax.random.split(next(prng_seq), num_samples),
          jax.lax.stop_gradient(q_distr_out['global_sample']),
          locations_stg2_,
      )
  if is_smi:
    log_q_stg2 = (
        jax.lax.stop_gradient(q_distr_out['global_sample_logprob']) +
        q_distr_out['loc_floating_logprob'])
  else:
    log_q_stg2 = (
        q_distr_out['global_sample_logprob'] +
        q_distr_out['loc_floating_logprob'])

  elbo_stg2 = log_prob_joint_stg2 - log_q_stg2

  # ELBO stage 3: fit posteriors for locations of anchor profiles
  # mainly for model evaluation (choosing eta)
  if include_random_anchor:
    locations_stg3_ = ModelParamsLocations(
        loc_floating=jax.lax.stop_gradient(
            q_distr_out['locations_sample'].loc_floating),
        loc_floating_aux=None,
        loc_random_anchor=q_distr_out['locations_sample'].loc_random_anchor,
    )
    log_prob_joint_stg3 = jax.vmap(
        lambda key_, global_, locations_: logprob_lalme(
            batch=batch,
            prng_key=key_,
            model_params_global=global_,
            model_params_locations=locations_,
            prior_hparams=prior_hparams,
            kernel_name=kernel_name,
            kernel_kwargs=kernel_kwargs,
            num_samples_gamma_profiles=num_samples_gamma_profiles,
            smi_eta_profiles=None,
            gp_jitter=gp_jitter,
            random_anchor=True,
        ))(
            jax.random.split(next(prng_seq), num_samples),
            jax.lax.stop_gradient(q_distr_out['global_sample']),
            locations_stg3_,
        )

    log_q_stg3 = (
        jax.lax.stop_gradient(q_distr_out['global_sample_logprob']) +
        jax.lax.stop_gradient(q_distr_out['loc_floating_logprob']) +
        q_distr_out['loc_random_anchor_logprob'])

    elbo_stg3 = log_prob_joint_stg3 - log_q_stg3
  else:
    elbo_stg3 = 0.

  elbo_dict = {
      'stage_1': elbo_stg1,
      'stage_2': elbo_stg2,
      'stage_3': elbo_stg3,
  }

  return elbo_dict


def loss(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_estimate(params_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(
      jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2'] +
                  elbo_dict['stage_3']))

  return loss_avg


def error_locations_estimate(
    locations_sample: Dict[str, Any],
    batch: Optional[Batch],
) -> Dict[str, Array]:
  """Compute average distance error."""

  error_loc_out = {}

  # Locations of LPs
  # (anchor, anchor_val, anchor_test, floating)
  targets_all = np.split(
      batch['loc'],
      np.cumsum(batch['num_profiles_split']),
      axis=0,
  )[:-1]  # Last element is empty
  assert len(targets_all) == 4

  # Predicted locations of floating profiles
  # (anchor_val, anchor_test, floating)
  pred_floating = np.split(
      locations_sample.loc_floating,
      np.cumsum(batch['num_profiles_split'][1:]),
      axis=1,
  )[:-1]  # Last element is empty
  assert len(pred_floating) == 3

  ## Anchor validation profiles
  if batch['num_profiles_split'][1] > 0:
    # Average of Mean posterior distance to true locations
    distances = jnp.linalg.norm(
        pred_floating[0] - targets_all[1][None, ...], ord=2, axis=-1)
    error_loc_out['mean_dist_anchor_val'] = distances.mean()
    error_loc_out['mean_sq_dist_anchor_val'] = (distances**2).mean()
    # Average of distance between true locations and posterior mean
    distances = jnp.linalg.norm(
        pred_floating[0].mean(axis=0) - targets_all[1], ord=2, axis=-1)
    error_loc_out['dist_mean_anchor_val'] = distances.mean()

  # Anchor test profiles
  if batch['num_profiles_split'][2] > 0:
    # Average of Mean posterior distance to true locations
    distances = jnp.linalg.norm(
        pred_floating[1] - targets_all[2][None, ...], ord=2, axis=-1)
    error_loc_out['mean_dist_anchor_test'] = distances.mean()
    error_loc_out['mean_sq_dist_anchor_test'] = (distances**2).mean()
    # Average of distance between true locations and posterior mean
    distances = jnp.linalg.norm(
        pred_floating[1].mean(axis=0) - targets_all[2], ord=2, axis=-1)
    error_loc_out['dist_mean_anchor_test'] = distances.mean()

  # Floating profiles
  # Average of Mean posterior distance to Fit-technique locations
  distances = jnp.linalg.norm(
      pred_floating[2] - targets_all[3][None, ...], ord=2, axis=-1)
  error_loc_out['mean_dist_floating'] = distances.mean()
  # Average of distance between Fit-technique locations and posterior mean
  distances = jnp.linalg.norm(
      pred_floating[2].mean(axis=0) - targets_all[3], ord=2, axis=-1)
  error_loc_out['dist_mean_floating'] = distances.mean()

  # Random Anchor profiles
  if locations_sample.loc_random_anchor is not None:
    distances = jnp.linalg.norm(
        locations_sample.loc_random_anchor - targets_all[0][None, ...],
        ord=2,
        axis=-1)
    error_loc_out['mean_dist_random_anchor'] = distances.mean()
    distances = jnp.linalg.norm(
        locations_sample.loc_random_anchor.mean(axis=0) - targets_all[0],
        ord=2,
        axis=-1)
    error_loc_out['dist_mean_random_anchor'] = distances.mean()

  return error_loc_out


def train_and_evaluate(config: ConfigDict, workdir: str) -> None:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  # Remove trailing slash
  workdir = workdir.rstrip("/")

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)

  # Load and process LALME dataset
  lalme_dataset = load_data(
      prng_key=jax.random.PRNGKey(0),  # use fixed seed for data loading
      config=config,
  )

  # Add some parameters to config
  config.num_profiles = lalme_dataset['num_profiles']
  config.num_profiles_anchor = lalme_dataset['num_profiles_anchor']
  config.num_profiles_floating = lalme_dataset['num_profiles_floating']
  config.num_forms_tuple = lalme_dataset['num_forms_tuple']
  config.num_inducing_points = math.prod(config.flow_kwargs.inducing_grid_shape)

  samples_path = workdir + '/lalme_az.nc'

  # For training, we need a Dictionary compatible with jit
  # we remove string vector
  train_ds = {
      k: v for k, v in lalme_dataset.items() if k not in ['items', 'forms']
  }

  # Compute GP covariance between anchor profiles
  train_ds['cov_anchor'] = getattr(
      kernels, config.kernel_name)(**config.kernel_kwargs).matrix(
          x1=train_ds['loc'][:train_ds['num_profiles_anchor'], :],
          x2=train_ds['loc'][:train_ds['num_profiles_anchor'], :],
      )

  train_ds = get_inducing_points(
      dataset=train_ds,
      inducing_grid_shape=config.flow_kwargs.inducing_grid_shape,
      kernel_name=config.kernel_name,
      kernel_kwargs=config.kernel_kwargs,
      gp_jitter=config.gp_jitter,
  )

  # In general, it would be possible to modulate the influence given per
  # individual profile and item.
  # For now, we only tune the influence of the floating profiles
  smi_eta = {
      'profiles':
          jnp.where(
              jnp.arange(train_ds['num_profiles']) <
              train_ds['num_profiles_anchor'],
              1.,
              config.eta_profiles_floating,
          ),
      'items':
          jnp.ones(len(train_ds['num_forms_tuple']))
  }
  # These parameters affect the dimension of the flow
  # so they are also part of the flow parameters
  config.flow_kwargs.num_profiles_anchor = lalme_dataset['num_profiles_anchor']
  config.flow_kwargs.num_profiles_floating = lalme_dataset[
      'num_profiles_floating']
  config.flow_kwargs.num_forms_tuple = lalme_dataset['num_forms_tuple']
  config.flow_kwargs.num_inducing_points = int(
      math.prod(config.flow_kwargs.inducing_grid_shape))
  config.flow_kwargs.is_smi = True
  # Get locations bounds
  # These define the range of values produced by the posterior of locations
  loc_bounds = np.stack(
      [lalme_dataset['loc'].min(axis=0), lalme_dataset['loc'].max(axis=0)],
      axis=1).astype(np.float32)
  config.flow_kwargs.loc_x_range = tuple(loc_bounds[0])
  config.flow_kwargs.loc_y_range = tuple(loc_bounds[1])

  ### Initialize States ###
  # Here we use three different states defining three separate flow models:
  #   -Global parameters
  #   -Posterior locations for floating profiles
  #   -Posterior locations for anchor profiles (treated as floating)

  # Global parameters
  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state_name_list = [
      'global', 'loc_floating', 'loc_floating_aux', 'loc_random_anchor'
  ]
  state_list = []

  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[0]}',
          forward_fn=hk.transform(q_distr_global),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'sample_shape': (config.num_samples_elbo,),
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # Get an initial sample of global parameters
  # (used below to initialize floating locations)
  global_sample_base_ = hk.transform(q_distr_global).apply(
      state_list[0].params,
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_elbo,),
  )['sample_base']

  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[1]}',
          forward_fn=hk.transform(q_distr_loc_floating),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'global_params_base_sample': global_sample_base_,
              'name': 'loc_floating',
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[2]}',
          forward_fn=hk.transform(q_distr_loc_floating),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'global_params_base_sample': global_sample_base_,
              'name': 'loc_floating_aux',
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0 and state_list[0].step < config.training_steps:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))

    # Syne-tune for HPO
    synetune_report = syne_tune.Reporter()
  else:
    summary_writer = None
    synetune_report = None

  # Print a useful summary of the execution of the flows.
  logging.info('FLOW GLOBAL PARAMETERS:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda state, prng_key: hk.transform(q_distr_global).apply(
          state.params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          sample_shape=(config.num_samples_elbo,),
      ),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_list[0], next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)

  logging.info('FLOW LOCATION FLOATING PROFILES:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda state, prng_key: hk.transform(q_distr_loc_floating).apply(
          state.params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          global_params_base_sample=global_sample_base_,
          name='loc_floating',
      ),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_list[1], next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)

  if config.include_random_anchor:
    state_list.append(
        initial_state_ckpt(
            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[3]}',
            forward_fn=hk.transform(q_distr_loc_random_anchor),
            forward_fn_kwargs={
                'flow_name': config.flow_name,
                'flow_kwargs': config.flow_kwargs,
                'global_params_base_sample': global_sample_base_,
            },
            prng_key=next(prng_seq),
            optimizer=make_optimizer(**config.optim_kwargs),
        ))

    logging.info('FLOW RANDOM LOCATION OF ANCHOR PROFILES:')
    tabulate_fn_ = hk.experimental.tabulate(
        f=lambda state, prng_key: hk.transform(q_distr_loc_random_anchor).apply(
            state.params,
            prng_key,
            flow_name=config.flow_name,
            flow_kwargs=config.flow_kwargs,
            global_params_base_sample=global_sample_base_,
        ),
        columns=(
            "module",
            "owned_params",
            "params_size",
            "params_bytes",
        ),
        filters=("has_params",),
    )
    summary = tabulate_fn_(state_list[3], next(prng_seq))
    for line in summary.split("\n"):
      logging.info(line)

  @jax.jit
  def update_states_jit(state_list, batch, prng_key, smi_eta):
    return update_states(
        state_list=state_list,
        batch=batch,
        prng_key=prng_key,
        optimizer=make_optimizer(**config.optim_kwargs),
        loss_fn=loss,
        loss_fn_kwargs={
            'num_samples': config.num_samples_elbo,
            'flow_name': config.flow_name,
            'flow_kwargs': config.flow_kwargs,
            'smi_eta': smi_eta,
            'include_random_anchor': config.include_random_anchor,
            'prior_hparams': config.prior_hparams,
            'kernel_name': config.kernel_name,
            'kernel_kwargs': config.kernel_kwargs,
            'num_samples_gamma_profiles': config.num_samples_gamma_profiles,
            'gp_jitter': config.gp_jitter,
        },
    )

  @jax.jit
  def elbo_validation_jit(state_list, batch, prng_key, smi_eta):
    return elbo_estimate(
        params_tuple=[state.params for state in state_list],
        batch=batch,
        prng_key=prng_key,
        num_samples=config.num_samples_eval,
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        smi_eta=smi_eta,
        include_random_anchor=config.include_random_anchor,
        prior_hparams=config.prior_hparams,
        kernel_name=config.kernel_name,
        kernel_kwargs=config.kernel_kwargs,
        num_samples_gamma_profiles=config.num_samples_gamma_profiles,
        gp_jitter=config.gp_jitter,
    )

  save_last_checkpoint = False
  if state_list[0].step < config.training_steps:
    logging.info('Training variational posterior...')
    # Reset random keys
    prng_seq = hk.PRNGSequence(config.seed)
    save_last_checkpoint = True

  while state_list[0].step < config.training_steps:

    # Plots to monitor training
    if config.log_img_steps > 0:
      if (state_list[0].step % config.log_img_steps == 0):
        logging.info("Logging plots...")
        # Sample from posterior with final state
        lalme_az = sample_lalme_az(
            state_list=state_list,
            batch=train_ds,
            prng_key=next(prng_seq),
            config=config,
            lalme_dataset=lalme_dataset,
            include_gamma=False,
            num_samples_chunk=config.num_samples_chunk_plot,
        )
        plot.lalme_plots_arviz(
            lalme_az=lalme_az,
            lalme_dataset=lalme_dataset,
            step=state_list[0].step,
            show_mu=True,
            show_zeta=True,
            lp_floating_grid10=config.lp_floating_grid10,
            lp_random_anchor_grid10=config.lp_random_anchor_10,
            workdir_png=workdir,
            summary_writer=summary_writer,
            suffix=f"_eta_floating_{config.eta_profiles_floating:.3f}",
            scatter_kwargs={"alpha": 0.05},
        )
        logging.info("...done.")

    # Log learning rate
    summary_writer.scalar(
        tag='learning_rate',
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(state_list[0].step),
        step=state_list[0].step,
    )

    # Training step
    state_list, metrics = update_states_jit(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
        smi_eta=smi_eta,
    )

    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state_list[0].step - 1,
    )

    if state_list[0].step == 1:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    # Metrics for evaluation
    if state_list[0].step % config.eval_steps == 0:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

      # Multi-stage ELBO
      elbo_dict_eval = elbo_validation_jit(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
          smi_eta=smi_eta,
      )
      for k, v in elbo_dict_eval.items():
        summary_writer.scalar(
            tag=f'elbo_{k}',
            value=v.mean(),
            step=state_list[0].step,
        )

      # Estimate posterior distance to true locations
      # Sample from flow
      locations_sample_eval = sample_all_flows(
          params_tuple=[state.params for state in state_list],
          prng_key=next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          sample_shape=(config.num_samples_eval,),
          include_random_anchor=config.include_random_anchor,
      )['locations_sample']

      error_loc_dict = error_locations_estimate(
          locations_sample=locations_sample_eval,
          batch=train_ds,
      )
      for k, v in error_loc_dict.items():
        summary_writer.scalar(
            tag=k,
            value=float(v),
            step=state_list[0].step,
        )
        # Report the metric used by syne-tune
        if k == config.synetune_metric:
          synetune_report(**{k: float(v)})

    if config.checkpoint_steps > 0:
      if state_list[0].step % config.checkpoint_steps == 0:
        for state, state_name in zip(state_list, state_name_list):
          save_checkpoint(
              state=state,
              checkpoint_dir=f'{checkpoint_dir}/{state_name}',
              keep=config.checkpoints_keep,
          )

    # Wait until computations are done before the next step
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info('Final training step: %i', state_list[0].step)

  # Saving checkpoint at the end of the training process
  if save_last_checkpoint:
    logging.info("\t Saving final checkpoint")
    for state, state_name in zip(state_list, state_name_list):
      save_checkpoint(
          state=state,
          checkpoint_dir=f'{checkpoint_dir}/{state_name}',
          keep=config.checkpoints_keep,
      )
    del state

  if os.path.exists(samples_path):
    logging.info("\t Loading final samples")
    lalme_az = az.from_netcdf(samples_path)
  else:
    logging.info("\t Saving final samples")
    # Sample from posterior with final state
    lalme_az = sample_lalme_az(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
        config=config,
        lalme_dataset=lalme_dataset,
        include_gamma=True,
        num_samples_chunk=config.num_samples_chunk_plot,
    )
    # Save InferenceData object with final state
    lalme_az.to_netcdf(samples_path)

  # Plot of final posterior
  if config.log_img_at_end:
    logging.info("Plotting results...")
    plot.lalme_plots_arviz(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        step=state_list[0].step,
        show_mu=True,
        show_zeta=True,
        show_basis_fields=True,
        show_W_items=lalme_dataset['items'],
        show_a_items=lalme_dataset['items'],
        lp_floating=lalme_dataset['LP'][lalme_dataset['num_profiles_anchor']:],
        lp_floating_traces=config.lp_floating_grid10,
        lp_floating_grid10=config.lp_floating_grid10,
        lp_random_anchor=(
            lalme_dataset['LP'][:lalme_dataset['num_profiles_anchor']]
            if config.include_random_anchor else None),
        # lp_random_anchor_grid10=config.lp_random_anchor_10,
        loc_inducing=train_ds['loc_inducing'],
        workdir_png=workdir,
        summary_writer=summary_writer,
        suffix=f"_eta_floating_{config.eta_profiles_floating:.3f}",
        scatter_kwargs={"alpha": 0.10},
    )
    logging.info("...done!")


# # For debugging
# config = get_config()
# config.eta_profiles_floating = 1.000
# workdir = str(pathlib.Path.home() / 'spatial-smi-output-exp/8_items/nsf/eta_floating_1.000')
# # train_and_evaluate(config, workdir)
