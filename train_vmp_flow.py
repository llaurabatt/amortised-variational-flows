"""Flow model for the LALME model."""

import math
import pathlib

from absl import logging

import numpy as np
from matplotlib import pyplot as plt
from arviz import InferenceData

from flax.metrics import tensorboard
import syne_tune

import jax
from jax import numpy as jnp

import haiku as hk
import optax

from tensorflow_probability.substrates import jax as tfp

import log_prob_fun_2
from log_prob_fun_2 import ModelParamsGlobal, ModelParamsLocations
import flows
import plot
from train_flow import (load_data, make_optimizer, get_inducing_points,
                        error_locations_estimate, logprob_lalme)

from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, initial_state_ckpt, update_states,
                          save_checkpoint, plot_to_image, normalize_images)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict, List,
                                      Optional, PRNGKey, SmiEta, SummaryWriter,
                                      Tuple)

kernels = tfp.math.psd_kernels

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def q_distr_global(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta: Array,
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

  num_samples = eta.shape[0]

  # Sample from flow
  (sample_flow_concat, sample_logprob,
   sample_base) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=[eta, None],
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
    eta: Array,
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
      context=[eta, global_params_base_sample],
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
    eta: Array,
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
      context=[eta, global_params_base_sample],
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
    smi_eta: SmiEta,
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
      eta=smi_eta['profiles'],
  )
  q_distr_out.update({f"global_{k}": v for k, v in q_distr_out_global.items()})

  # Floating profiles locations
  q_distr_out_loc_floating = hk.transform(q_distr_loc_floating).apply(
      params_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      global_params_base_sample=q_distr_out_global['sample_base'],
      eta=smi_eta['profiles'],
      name='loc_floating',
  )
  q_distr_out['loc_floating_logprob'] = q_distr_out_loc_floating[
      'sample_logprob']

  # Auxiliary Floating profiles locations
  q_distr_out_loc_floating_aux = hk.transform(q_distr_loc_floating).apply(
      params_tuple[2],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      global_params_base_sample=q_distr_out_global['sample_base'],
      eta=smi_eta['profiles'],
      name='loc_floating_aux',
  )
  q_distr_out['locations_sample'] = ModelParamsLocations(
      loc_floating=q_distr_out_loc_floating['sample'].loc_floating,
      loc_floating_aux=q_distr_out_loc_floating_aux['sample'].loc_floating_aux,
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
            eta=smi_eta['profiles'],
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
    smi_eta: SmiEta,
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

  assert all(x.ndim == 2 for x in smi_eta.values())

  # Sampling divided into chunks, to avoid OOM on GPU
  # Split etas into chunks
  split_idx_ = np.arange(num_samples_chunk, smi_eta['profiles'].shape[0],
                         num_samples_chunk).tolist()
  smi_eta_chunked_ = jax.tree_map(lambda x: jnp.split(x, split_idx_, axis=0),
                                  smi_eta)
  # dict of lists -> list of dicts
  smi_eta_chunked_ = [
      dict(zip(smi_eta_chunked_, t)) for t in zip(*smi_eta_chunked_.values())
  ]

  for smi_eta_ in smi_eta_chunked_:
    # Sample from variational posterior
    q_distr_out = sample_all_flows(
        params_tuple=[state.params for state in state_list],
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        smi_eta=smi_eta_,
        include_random_anchor=config.include_random_anchor,
    )

    global_sample.append(q_distr_out['global_sample'])
    locations_sample.append(q_distr_out['locations_sample'])

    if include_gamma:
      # Get a sample of the basis GPs on profiles locations
      # conditional on values at the inducing locations.
      gamma_sample_, _ = jax.vmap(
          lambda key_, global_, locations_: log_prob_fun_2.
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
              jax.random.split(next(prng_seq), smi_eta_['profiles'].shape[0]),
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


def elbo_estimate_along_eta(
    params_tuple: Tuple[hk.Params],
    batch: Optional[Batch],
    prng_key: PRNGKey,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta_sampling_a: float,
    eta_sampling_b: float,
    include_random_anchor: bool,
    profile_is_anchor: Array,
    prior_hparams: Dict[str, Any],
    kernel_name: Optional[str] = None,
    kernel_kwargs: Optional[Dict[str, Any]] = None,
    num_samples_gamma_profiles: int = 0,
    gp_jitter: Optional[float] = None,
) -> Dict[str, Array]:
  # params_tuple = [state.params for state in state_list]

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample eta values
  etas_profiles_floating = jax.random.beta(
      key=next(prng_seq),
      a=eta_sampling_a,
      b=eta_sampling_b,
      shape=(num_samples,),
  )

  smi_eta_elbo = {
      'profiles':
          jax.vmap(lambda eta_: jnp.where(
              profile_is_anchor,
              1.,
              eta_,
          ))(etas_profiles_floating),
      'items':
          jnp.ones((num_samples, len(batch['num_forms_tuple']))),
  }

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      prng_key=next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      smi_eta=smi_eta_elbo,
      include_random_anchor=include_random_anchor,
  )

  # ELBO stage 1: Power posterior
  locations_stg1_ = ModelParamsLocations(
      loc_floating=q_distr_out['locations_sample'].loc_floating_aux,
      loc_floating_aux=None,
      loc_random_anchor=None,
  )
  log_prob_joint_stg1 = jax.vmap(
      lambda key_, global_, locations_, smi_eta_: logprob_lalme(
          batch=batch,
          prng_key=key_,
          model_params_global=global_,
          model_params_locations=locations_,
          prior_hparams=prior_hparams,
          kernel_name=kernel_name,
          kernel_kwargs=kernel_kwargs,
          num_samples_gamma_profiles=num_samples_gamma_profiles,
          smi_eta_profiles=smi_eta_,
          gp_jitter=gp_jitter,
          random_anchor=False,
      ))(
          jax.random.split(next(prng_seq), num_samples),
          q_distr_out['global_sample'],
          locations_stg1_,
          smi_eta_elbo['profiles'],
      )

  log_q_stg1 = (
      q_distr_out['global_sample_logprob'] +
      q_distr_out['loc_floating_aux_logprob'])

  elbo_stg1 = log_prob_joint_stg1 - log_q_stg1

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
  log_q_stg2 = (
      jax.lax.stop_gradient(q_distr_out['global_sample_logprob']) +
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
  elbo_dict = elbo_estimate_along_eta(
      params_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(
      jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2'] +
                  elbo_dict['stage_3']))

  return loss_avg


def error_locations_vector_estimate(
    state_list: List[TrainState],
    batch: Optional[Batch],
    prng_key: PRNGKey,
    config: ConfigDict,
    eta_eval_grid: Array,
    num_samples: int,
) -> Dict[str, Array]:
  """Compute average distance error along eta_eval_grid.
  
  Note:
    This could be computed using vmap, but ran into OOM issues.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  error_grid = []

  for eta_i in eta_eval_grid:
    # eta_i = eta_eval_grid[0]
    eta_i_profiles = eta_i * jnp.ones((num_samples, config.num_profiles))
    smi_eta_ = {
        'profiles':
            jax.vmap(lambda eta_: jnp.where(
                jnp.arange(config.num_profiles) < config.num_profiles_anchor,
                1.,
                eta_,
            ))(eta_i_profiles),
        'items':
            jnp.ones((num_samples, len(config.num_forms_tuple))),
    }
    q_distr_out = sample_all_flows(
        params_tuple=[state.params for state in state_list],
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        smi_eta=smi_eta_,
        include_random_anchor=config.include_random_anchor,
    )

    error_grid.append(
        error_locations_estimate(
            locations_sample=q_distr_out['locations_sample'],
            batch=batch,
        ))

  error_loc_dict = jax.tree_map(lambda *x: jnp.stack(x, axis=0), *error_grid)  # pylint: disable=no-value-for-parameter

  return error_loc_dict


def log_images(
    state_list: List[TrainState],
    prng_key: PRNGKey,
    config: ConfigDict,
    lalme_dataset: Dict[str, Any],
    batch: Batch,
    show_mu: bool = False,
    show_zeta: bool = False,
    show_basis_fields: bool = False,
    show_W_items: Optional[List[str]] = None,
    show_a_items: Optional[List[str]] = None,
    lp_floating: Optional[List[int]] = None,
    lp_floating_traces: Optional[List[int]] = None,
    lp_floating_grid10: Optional[List[int]] = None,
    lp_random_anchor: Optional[List[int]] = None,
    lp_random_anchor_grid10: Optional[List[int]] = None,
    show_lp_anchor_val: Optional[bool] = False,
    show_lp_anchor_test: Optional[bool] = False,
    loc_inducing: Optional[Array] = None,
    show_eval_metric: bool = False,
    eta_eval_grid: Optional[Array] = None,
    num_samples_chunk: int = 1_000,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  profile_is_anchor = jnp.arange(
      config.num_profiles) < config.num_profiles_anchor

  # List of LP for validation
  if show_lp_anchor_val:
    lp_anchor_val = np.split(lalme_dataset['LP'],
                             np.cumsum(lalme_dataset['num_profiles_split']))[1]
  else:
    lp_anchor_val = None
  # List of LP for test
  if show_lp_anchor_test:
    lp_anchor_test = np.split(lalme_dataset['LP'],
                              np.cumsum(lalme_dataset['num_profiles_split']))[2]
  else:
    lp_anchor_test = None

  # Plot posterior samples
  for eta_i in config.eta_plot:

    eta_i_profiles = jax.vmap(lambda eta_: jnp.where(
        profile_is_anchor,
        1.,
        eta_,
    ))(
        eta_i * jnp.ones((config.num_samples_plot, config.num_profiles)))

    smi_eta_ = {
        'profiles':
            eta_i_profiles,
        'items':
            jnp.ones((config.num_samples_plot, len(config.num_forms_tuple))),
    }

    lalme_az_ = sample_lalme_az(
        state_list=state_list,
        batch=batch,
        smi_eta=smi_eta_,
        prng_key=next(prng_seq),
        config=config,
        lalme_dataset=lalme_dataset,
        include_gamma=show_basis_fields,
        num_samples_chunk=num_samples_chunk,
    )

    plot.lalme_plots_arviz(
        lalme_az=lalme_az_,
        lalme_dataset=lalme_dataset,
        step=state_list[0].step,
        show_mu=show_mu,
        show_zeta=show_zeta,
        show_basis_fields=show_basis_fields,
        show_W_items=show_W_items,
        show_a_items=show_a_items,
        lp_floating=lp_floating,
        lp_floating_traces=lp_floating_traces,
        lp_floating_grid10=lp_floating_grid10,
        lp_random_anchor=lp_random_anchor,
        lp_random_anchor_grid10=lp_random_anchor_grid10,
        lp_anchor_val=lp_anchor_val,
        lp_anchor_test=lp_anchor_test,
        loc_inducing=loc_inducing,
        workdir_png=workdir_png,
        summary_writer=summary_writer,
        suffix=f"_eta_floating_{float(eta_i):.3f}",
        scatter_kwargs={"alpha": 0.10},
    )

  ### Evaluation metrics ###
  if show_eval_metric:

    error_loc_dict = error_locations_vector_estimate(
        state_list=state_list,
        batch=batch,
        prng_key=next(prng_seq),
        config=config,
        eta_eval_grid=eta_eval_grid,
        num_samples=num_samples_chunk,
    )
    images = []

    if 'mean_dist_anchor_val' in error_loc_dict:
      plot_name = 'lalme_vmp_mean_dist_anchor_val'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['mean_dist_anchor_val'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean posterior distance')
      axs.set_title('Error distance for Anchor profiles (validation)\n' +
                    '(Mean difference posterior vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if 'dist_mean_anchor_val' in error_loc_dict:
      plot_name = 'lalme_vmp_dist_mean_anchor_val'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['dist_mean_anchor_val'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Distance to posterior mean')
      axs.set_title('Error distance for Anchor profiles (validation)\n' +
                    '(Posterior Mean vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if 'mean_dist_anchor_test' in error_loc_dict:
      plot_name = 'lalme_vmp_mean_dist_anchor_test'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['mean_dist_anchor_test'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean posterior distance')
      axs.set_title('Error distance for Anchor profiles (test)\n' +
                    '(Mean difference posterior vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if 'dist_mean_anchor_test' in error_loc_dict:
      plot_name = 'lalme_vmp_dist_mean_anchor_test'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['dist_mean_anchor_test'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Distance to posterior mean')
      axs.set_title('Error distance for Anchor profiles (test)\n' +
                    '(Posterior Mean vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    plot_name = 'lalme_vmp_mean_dist_floating'
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    # Plot distance as a function of eta
    axs.plot(eta_eval_grid, error_loc_dict['mean_dist_floating'])
    axs.set_xlabel('eta_floating')
    axs.set_ylabel('Mean distance')
    axs.set_title('Error distance for floating profiles\n' +
                  '(posterior vs. fit-technique)')
    fig.tight_layout()
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    if config.include_random_anchor:
      plot_name = 'lalme_vmp_mean_dist_random_anchor'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['mean_dist_random_anchor'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean posterior distance')
      axs.set_title('Error distance for anchor profiles\n' +
                    '(posterior vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if summary_writer:
      plot_name = 'lalme_vmp_distance'
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=state_list[0].step,
      )


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

  # For training, we need a Dictionary compatible with jit
  # we remove string vectors
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

  smi_eta_init = {
      'profiles':
          jnp.ones((config.num_samples_elbo, train_ds['num_profiles'])),
      'items':
          jnp.ones((config.num_samples_elbo, len(train_ds['num_forms_tuple']))),
  }

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
              'eta': smi_eta_init['profiles'],
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
      eta=smi_eta_init['profiles'],
  )['sample_base']

  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[1]}',
          forward_fn=hk.transform(q_distr_loc_floating),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'global_params_base_sample': global_sample_base_,
              'eta': smi_eta_init['profiles'],
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
              'eta': smi_eta_init['profiles'],
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
  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(flatten_dict(config))

  # Print a useful summary of the execution of the flows.
  logging.info('FLOW GLOBAL PARAMETERS:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda state, prng_key: hk.transform(q_distr_global).apply(
          state.params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          eta=smi_eta_init['profiles'],
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
          eta=smi_eta_init['profiles'],
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
                'eta': smi_eta_init['profiles'],
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
            eta=smi_eta_init['profiles'],
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

  profile_is_anchor = jnp.arange(
      train_ds['num_profiles']) < train_ds['num_profiles_anchor']

  @jax.jit
  def update_states_jit(state_list, batch, prng_key):
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
            'eta_sampling_a': config.eta_sampling_a,
            'eta_sampling_b': config.eta_sampling_b,
            'include_random_anchor': config.include_random_anchor,
            'prior_hparams': config.prior_hparams,
            'profile_is_anchor': profile_is_anchor,
            'kernel_name': config.kernel_name,
            'kernel_kwargs': config.kernel_kwargs,
            'num_samples_gamma_profiles': config.num_samples_gamma_profiles,
            'gp_jitter': config.gp_jitter,
        },
    )

  # globals().update(loss_fn_kwargs)

  # @jax.jit
  # def elbo_validation_jit(state_list, batch, prng_key):
  #   return elbo_estimate_along_eta(
  #       params_tuple=[state.params for state in state_list],
  #       batch=batch,
  #       prng_key=prng_key,
  #       num_samples=config.num_samples_eval,
  #       flow_name=config.flow_name,
  #       flow_kwargs=config.flow_kwargs,
  #       eta_sampling_a=1.0,
  #       eta_sampling_b=1.0,
  #       include_random_anchor=config.include_random_anchor,
  #       prior_hparams=config.prior_hparams,
  #       profile_is_anchor=profile_is_anchor,
  #       kernel_name=config.kernel_name,
  #       kernel_kwargs=config.kernel_kwargs,
  #       num_samples_gamma_profiles=config.num_samples_gamma_profiles,
  #       gp_jitter=config.gp_jitter,
  #   )

  # error_locations_estimate_jit = lambda state_list, batch, prng_key: error_locations_estimate(
  #     params_tuple=[state.params for state in state_list],
  #     batch=batch,
  #     prng_key=prng_key,
  #     num_samples=config.num_samples_eval,
  #     flow_name=config.flow_name,
  #     flow_kwargs=config.flow_kwargs,
  #     include_random_anchor=config.include_random_anchor,
  #     kernel_name=config.kernel_name,
  #     kernel_kwargs=config.kernel_kwargs,
  #     num_samples_gamma_profiles=config.num_samples_gamma_profiles,
  #     gp_jitter=config.gp_jitter,
  # )
  # # TODO: This doesn't work after jitting.
  # # error_locations_estimate_jit = jax.jit(error_locations_estimate_jit)

  save_last_checkpoint = False
  if state_list[0].step < config.training_steps:
    logging.info('Training variational posterior...')
    # Reset random keys
    prng_seq = hk.PRNGSequence(config.seed)
    save_last_checkpoint = True

  loss_ = []
  while state_list[0].step < config.training_steps:
    # step = 0

    # Plots to monitor training
    if config.log_img_steps > 0:
      if (state_list[0].step > 0) and (state_list[0].step %
                                       config.log_img_steps) == 0:
        logging.info("Logging plots...")
        log_images(
            state_list=state_list,
            prng_key=next(prng_seq),
            config=config,
            lalme_dataset=lalme_dataset,
            batch=train_ds,
            show_mu=True,
            show_zeta=True,
            lp_floating_grid10=config.lp_floating_grid10,
            lp_random_anchor_grid10=config.lp_random_anchor_10,
            show_eval_metric=True,
            eta_eval_grid=jnp.linspace(0, 1, 21),
            num_samples_chunk=config.num_samples_chunk_plot,
            summary_writer=summary_writer,
            workdir_png=workdir,
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
    )

    loss_.append(float(metrics['train_loss']))
    if len(loss_) >= config.max_steps_nan:
      loss_ = loss_[-config.max_steps_nan:]
      if jnp.isnan(jnp.array(loss_).astype(float)).all():
        logging.warning('Training stopped, %d steps with NaN loss',
                        config.max_steps_nan)
        break

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

      # # Multi-stage ELBO
      # elbo_dict_eval = elbo_validation_jit(
      #     state_list=state_list,
      #     batch=train_ds,
      #     prng_key=next(prng_seq),
      # )
      # for k, v in elbo_dict_eval.items():
      #   summary_writer.scalar(
      #       tag=f'elbo_{k}',
      #       value=v.mean(),
      #       step=state_list[0].step,
      #   )

      # Estimate posterior distance to true locations
      eta_eval_grid_ = jnp.linspace(0, 1, 21)
      # Each element is a vector across eta
      error_loc_all_eta_dict = error_locations_vector_estimate(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
          config=config,
          eta_eval_grid=eta_eval_grid_,
          num_samples=config.num_samples_eval,
      )

      # Summarize the distances
      error_loc_dict = {}
      for k, v in error_loc_all_eta_dict.items():
        error_loc_dict[k + '_min'] = jnp.min(v)
        error_loc_dict[k + '_min_eta'] = eta_eval_grid_[jnp.argmin(v)]
        error_loc_dict[k + '_max'] = jnp.max(v)
        error_loc_dict[k + '_max_eta'] = eta_eval_grid_[jnp.argmax(v)]

      for k, v in error_loc_dict.items():
        summary_writer.scalar(
            tag=k,
            value=float(v),
            step=state_list[0].step,
        )
        # Report the metric used by syne-tune
        if k == config.synetune_metric:
          synetune_report(**{k: float(v)})
          # synetune_report(**{k + '_max': float(jnp.max(v))})

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
    for state, state_name in zip(state_list, state_name_list):
      save_checkpoint(
          state=state,
          checkpoint_dir=f'{checkpoint_dir}/{state_name}',
          keep=config.checkpoints_keep,
      )
    del state

  # Estimate posterior distance to true locations
  eval_last = True
  if eval_last and summary_writer is not None:
    eta_eval_grid_ = jnp.linspace(0, 1, 21)
    # Each element is a vector across eta
    error_loc_all_eta_dict = error_locations_vector_estimate(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
        config=config,
        eta_eval_grid=eta_eval_grid_,
        num_samples=config.num_samples_eval,
    )

    # Summarize the distances
    error_loc_dict = {}
    for k, v in error_loc_all_eta_dict.items():
      error_loc_dict[k + '_min'] = jnp.min(v)
      error_loc_dict[k + '_min_eta'] = eta_eval_grid_[jnp.argmin(v)]
      error_loc_dict[k + '_max'] = jnp.max(v)
      error_loc_dict[k + '_max_eta'] = eta_eval_grid_[jnp.argmax(v)]

    for k, v in error_loc_dict.items():
      summary_writer.scalar(
          tag=k,
          value=float(v),
          step=state_list[0].step,
      )
      # Report the metric used by syne-tune
      if k == config.synetune_metric:
        synetune_report(**{k: float(v)})

  # Save samples from the last state
  for eta_i in config.eta_plot:
    eta_i_profiles = eta_i * jnp.ones(
        (config.num_samples_plot, config.num_profiles))

    smi_eta_ = {
        'profiles':
            jax.vmap(lambda eta_: jnp.where(
                profile_is_anchor,
                1.,
                eta_,
            ))(eta_i_profiles),
        'items':
            jnp.ones((config.num_samples_plot, len(config.num_forms_tuple))),
    }

    lalme_az_ = sample_lalme_az(
        state_list=state_list,
        batch=train_ds,
        smi_eta=smi_eta_,
        prng_key=next(prng_seq),
        config=config,
        lalme_dataset=lalme_dataset,
        include_gamma=False,
    )
    lalme_az_.to_netcdf(workdir + f'/lalme_az_eta_{float(eta_i):.3f}.nc')

  # Last plot of posteriors
  if config.log_img_at_end:
    logging.info("Plotting results...")
    log_images(
        state_list=state_list,
        prng_key=next(prng_seq),
        config=config,
        lalme_dataset=lalme_dataset,
        batch=train_ds,
        show_mu=True,
        show_zeta=True,
        show_basis_fields=True,
        # show_W_items=lalme_dataset['items'],
        # show_a_items=lalme_dataset['items'],
        lp_floating=lalme_dataset['LP'][lalme_dataset['num_profiles_anchor']:],
        lp_floating_traces=config.lp_floating_grid10,
        lp_floating_grid10=config.lp_floating_grid10,
        lp_random_anchor=(
            lalme_dataset['LP'][:lalme_dataset['num_profiles_anchor']]
            if config.include_random_anchor else None),
        lp_random_anchor_grid10=(config.lp_random_anchor_10
                                 if config.include_random_anchor else None),
        show_lp_anchor_val=(
            True if lalme_dataset['num_profiles_split'].lp_anchor_val > 0 else
            False),
        show_lp_anchor_test=(
            True if lalme_dataset['num_profiles_split'].lp_anchor_test > 0 else
            False),
        loc_inducing=train_ds['loc_inducing'],
        show_eval_metric=True,
        eta_eval_grid=jnp.linspace(0, 1, 21),
        num_samples_chunk=config.num_samples_chunk_plot,
        summary_writer=summary_writer,
        workdir_png=workdir,
    )
    logging.info("...done!")


# # For debugging
# config = get_config()
# workdir = str(pathlib.Path.home() / 'spatial-smi-output-exp/all_items/nsf/vmp_flow')
# # train_and_evaluate(config, workdir)

# config.checkpoint_steps = -1
# config.eval_steps = 5000
# config.kernel_kwargs.amplitude = 0.515
# config.kernel_kwargs.length_scale = 0.515
# config.log_img_at_end = False
# config.log_img_steps = -1
# config.optim_kwargs.lr_schedule_kwargs.decay_rate = 0.55
# config.optim_kwargs.lr_schedule_kwargs.peak_value = 0.00031622776601683783
# config.optim_kwargs.lr_schedule_kwargs.transition_steps = 10000
# config.synetune_metric = "mean_dist_anchor_val"
# config.training_steps = 30000
