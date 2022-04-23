"""Flow model for the LALME model."""

import math
import pathlib

from absl import logging

import numpy as np

# from clu import metric_writers
from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

from tensorflow_probability.substrates import jax as tfp

import data
import flows
import log_prob_fun
import plot

from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, issymmetric, initial_state_ckpt,
                          update_states, save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict,
                                      IntLike, List, Optional, PRNGKey,
                                      Sequence, SmiEta, SummaryWriter, Tuple,
                                      Union)

kernels = tfp.math.psd_kernels

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def load_data(config: ConfigDict) -> Dict[str, Array]:
  """Load LALME data."""

  data_dict = data.load_lalme(dataset_id=config.dataset_id)

  # Get locations bounds
  # loc_bounds = jnp.array([[0., 1.], [0., 1.]])
  loc_bounds = np.stack(
      [data_dict['loc'].min(axis=0), data_dict['loc'].max(axis=0)],
      axis=1).astype(np.float32)
  # Shift x's and y's to start on zero
  loc_bounds = loc_bounds - loc_bounds[:, [0]]
  # scale to the unit square, preserving relative size of axis
  loc_scale = 1 / loc_bounds.max()
  loc_bounds = loc_bounds * loc_scale

  # Process data
  data_dict = data.process_lalme(
      data_raw=data_dict,
      num_profiles_anchor_keep=config.num_profiles_anchor_keep,
      num_profiles_floating_keep=config.num_profiles_floating_keep,
      num_items_keep=config.num_items_keep,
      loc_bounds=loc_bounds,
      remove_empty_forms=config.remove_empty_forms,
  )

  return data_dict


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


def get_inducing_points(dataset: Batch, config: ConfigDict):
  """Define grid of inducing point for GPs."""
  dataset = dataset.copy()

  # Inducing points are defined as a grid on the unit square
  loc_inducing = jnp.meshgrid(
      jnp.linspace(0, 1, config.flow_kwargs.inducing_grid_shape[0]),
      jnp.linspace(0, 1, config.flow_kwargs.inducing_grid_shape[1]))
  loc_inducing = jnp.stack(loc_inducing, axis=-1).reshape(-1, 2)
  dataset['loc_inducing'] = loc_inducing
  # Compute GP covariance between inducing values
  dataset['cov_inducing'] = getattr(
      kernels, config.kernel_name)(**config.kernel_kwargs).matrix(
          x1=dataset['loc_inducing'],
          x2=dataset['loc_inducing'],
      )

  # Add jitter
  dataset['cov_inducing'] = dataset[
      'cov_inducing'] + config.gp_jitter * jnp.eye(config.num_inducing_points)
  # Check that the covarince is symmetric
  assert issymmetric(dataset['cov_inducing'])

  # Cholesky factor of covariance
  dataset['cov_inducing_chol'] = jnp.linalg.cholesky(dataset['cov_inducing'])

  # Inverse of covariance of inducing values
  # dataset['cov_inducing_inv'] = jnp.linalg.inv(dataset['cov_inducing'])
  cov_inducing_chol_inv = jax.scipy.linalg.solve_triangular(
      a=dataset['cov_inducing_chol'],
      b=jnp.eye(config.num_inducing_points),
      lower=True,
  )
  dataset['cov_inducing_inv'] = jnp.matmul(
      cov_inducing_chol_inv.T, cov_inducing_chol_inv, precision='highest')

  # Check that the inverse is symmetric
  assert issymmetric(dataset['cov_inducing_inv'])
  # Check that there are no NaNs
  assert ~jnp.any(jnp.isnan(dataset['cov_inducing_inv']))
  # Cross covariance between anchor and inducing values
  dataset['cov_anchor_inducing'] = getattr(
      kernels, config.kernel_name)(**config.kernel_kwargs).matrix(
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
  (global_params_sample, global_params_log_prob,
   global_params_base_sample) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(), sample_shape=sample_shape)

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = flows.split_flow_global_params(
      samples=global_params_sample,
      **flow_kwargs,
  )

  # Log_probabilities of the sample
  q_distr_out['global_params_log_prob'] = global_params_log_prob

  # Sample from base distribution are preserved
  # These are used for the posterior of profiles locations
  q_distr_out['global_params_base_sample'] = global_params_base_sample

  return q_distr_out


def q_distr_loc_floating(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    global_params_base_sample: Array,
    is_aux: bool,
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
  q_distr_locations = getattr(flows, flow_name + '_locations')(
      num_profiles=flow_kwargs['num_profiles_floating'], **flow_kwargs)

  # Sample from flow
  num_samples = global_params_base_sample.shape[0]
  (locations_sample,
   locations_log_prob) = q_distr_locations.sample_and_log_prob(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=global_params_base_sample,
   )

  # Split flow into model parameters
  # (and add to existing posterior_sample_dict)
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows.split_flow_locations(
          samples=locations_sample,
          num_profiles=flow_kwargs['num_profiles_floating'],
          is_aux=is_aux,
          name='loc_floating',
      ))

  # log P(beta,tau|sigma)
  q_distr_out['loc_floating_' + ('aux_' if is_aux else '') +
              'log_prob'] = locations_log_prob

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
  num_samples = global_params_base_sample.shape[0]

  q_distr_locations = getattr(flows, flow_name + '_locations')(
      num_profiles=flow_kwargs['num_profiles_anchor'], **flow_kwargs)
  # Sample from flow
  (locations_sample,
   locations_log_prob) = q_distr_locations.sample_and_log_prob(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=global_params_base_sample,
   )

  # Split flow into model parameters
  # (and add to existing posterior_sample_dict)
  q_distr_out['posterior_sample'] = flows.split_flow_locations(
      samples=locations_sample,
      num_profiles=flow_kwargs['num_profiles_anchor'],
      is_aux=False,
      name='loc_random_anchor',
  )

  # Log_probabilities of the sample
  q_distr_out['loc_random_anchor_log_prob'] = locations_log_prob

  return q_distr_out


def sample_all_flows(
    params_tuple: Tuple[hk.Params],
    batch: Optional[Batch],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    sample_shape: Union[IntLike, Sequence[IntLike]],
    include_random_anchor: bool,
    kernel_name: Optional[str],
    kernel_kwargs: Optional[Dict[str, Any]],
    num_samples_gamma_profiles: int = 0,
    gp_jitter: Optional[float] = None,
) -> Dict[str, Any]:

  prng_seq = hk.PRNGSequence(prng_key)

  # Global parameters
  q_distr_out = hk.transform(q_distr_global).apply(
      params_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=sample_shape,
  )
  # Floating profiles locations
  q_distr_out_loc_floating = hk.transform(q_distr_loc_floating).apply(
      params_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      global_params_base_sample=q_distr_out['global_params_base_sample'],
      is_aux=False,
  )
  q_distr_out['posterior_sample'].update(
      q_distr_out_loc_floating['posterior_sample'])
  q_distr_out['loc_floating_log_prob'] = q_distr_out_loc_floating[
      'loc_floating_log_prob']

  if flow_kwargs.is_smi:
    # Floating profiles locations
    q_distr_out_loc_floating = hk.transform(q_distr_loc_floating).apply(
        params_tuple[2],
        next(prng_seq),
        flow_name=flow_name,
        flow_kwargs=flow_kwargs,
        global_params_base_sample=q_distr_out['global_params_base_sample'],
        is_aux=True,
    )
    q_distr_out['posterior_sample'].update(
        q_distr_out_loc_floating['posterior_sample'])
    q_distr_out['loc_floating_aux_log_prob'] = q_distr_out_loc_floating[
        'loc_floating_aux_log_prob']

  # Anchor profiles locations
  if include_random_anchor:
    q_distr_out_loc_random_anchor = hk.transform(
        q_distr_loc_random_anchor).apply(
            params_tuple[3],
            next(prng_seq),
            flow_name=flow_name,
            flow_kwargs=flow_kwargs,
            global_params_base_sample=q_distr_out['global_params_base_sample'],
        )
    q_distr_out['posterior_sample'].update(
        q_distr_out_loc_random_anchor['posterior_sample'])
    q_distr_out['loc_random_anchor_log_prob'] = q_distr_out_loc_random_anchor[
        'loc_random_anchor_log_prob']

  if num_samples_gamma_profiles > 0:
    # Sample gamma_profiles
    # Such samples are not part of the posterior approximation q(Theta),
    # but we add them to the dictionary 'posterior_sample_dict' for convenience.
    gamma_sample_dict = log_prob_fun.sample_gamma_profiles_given_gamma_inducing(
        batch=batch,
        posterior_sample_dict=q_distr_out['posterior_sample'],
        prng_key=next(prng_seq),
        kernel_name=kernel_name,
        kernel_kwargs=kernel_kwargs,
        gp_jitter=gp_jitter,
        num_samples_gamma_profiles=num_samples_gamma_profiles,
        is_smi=flow_kwargs.is_smi,
        include_random_anchor=include_random_anchor,
    )
    q_distr_out['posterior_sample'].update(gamma_sample_dict)

  return q_distr_out


def elbo_estimate(
    params_tuple: Tuple[hk.Params],
    batch: Optional[Batch],
    prng_key: PRNGKey,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    smi_eta: Optional[SmiEta],
    include_random_anchor: bool,
    kernel_name: Optional[str] = None,
    kernel_kwargs: Optional[Dict[str, Any]] = None,
    num_samples_gamma_profiles: int = 0,
    gp_jitter: Optional[float] = None,
) -> Dict[str, Array]:
  # params_tuple = [state.params for state in state_list]

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      batch=batch,
      prng_key=prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples,),
      include_random_anchor=include_random_anchor,
      kernel_name=kernel_name,
      kernel_kwargs=kernel_kwargs,
      num_samples_gamma_profiles=num_samples_gamma_profiles,
      gp_jitter=gp_jitter,
  )

  is_smi = False if smi_eta is None else True

  shared_params_names = [
      'gamma_inducing',
      'gamma_anchor',
      'mixing_weights_list',
      'mixing_offset_list',
      'mu',
      'zeta',
  ]
  refit_params_names = [
      'gamma_floating',
      'loc_floating',
  ]

  # ELBO stage 1: Power posterior
  if is_smi:
    posterior_sample_dict_stg1 = {}
    for key in shared_params_names:
      posterior_sample_dict_stg1[key] = q_distr_out['posterior_sample'][key]
    for key in refit_params_names:
      posterior_sample_dict_stg1[key] = q_distr_out['posterior_sample'][key +
                                                                        '_aux']
    log_prob_joint_stg1 = log_prob_fun.log_prob_joint(
        batch=batch,
        posterior_sample_dict=posterior_sample_dict_stg1,
        smi_eta=smi_eta,
    )
    log_q_stg1 = (
        q_distr_out['global_params_log_prob'] +
        q_distr_out['loc_floating_aux_log_prob'])

    elbo_stg1 = log_prob_joint_stg1 - log_q_stg1
  else:
    elbo_stg1 = 0.

  # ELBO stage 2: Refit locations floating profiles
  posterior_sample_dict_stg2 = {}
  for key in shared_params_names:
    if is_smi:
      posterior_sample_dict_stg2[key] = jax.lax.stop_gradient(
          q_distr_out['posterior_sample'][key])
    else:
      posterior_sample_dict_stg2[key] = q_distr_out['posterior_sample'][key]
  for key in refit_params_names:
    posterior_sample_dict_stg2[key] = q_distr_out['posterior_sample'][key]

  log_prob_joint_stg2 = log_prob_fun.log_prob_joint(
      batch=batch,
      posterior_sample_dict=posterior_sample_dict_stg2,
      smi_eta=None,
  )
  if is_smi:
    log_q_stg2 = (
        jax.lax.stop_gradient(q_distr_out['global_params_log_prob']) +
        q_distr_out['loc_floating_log_prob'])
  else:
    log_q_stg2 = (
        q_distr_out['global_params_log_prob'] +
        q_distr_out['loc_floating_log_prob'])

  elbo_stg2 = log_prob_joint_stg2 - log_q_stg2

  # For model evaluation,
  if include_random_anchor:
    # ELBO stage 3: fit posteriors for locations of anchor profiles
    stop_grad_params_names = [
        'gamma_inducing',
        'mixing_weights_list',
        'mixing_offset_list',
        'mu',
        'zeta',
        'gamma_floating',
        'loc_floating',
    ]
    grad_params_names = [
        'gamma_random_anchor',
        'loc_random_anchor',
    ]
    posterior_sample_dict_stg3 = {}
    for key in stop_grad_params_names:
      posterior_sample_dict_stg3[key] = jax.lax.stop_gradient(
          q_distr_out['posterior_sample'][key])
    for key in grad_params_names:
      posterior_sample_dict_stg3[key] = q_distr_out['posterior_sample'][key]

    log_prob_joint_stg3 = log_prob_fun.log_prob_joint(
        batch=batch,
        posterior_sample_dict=posterior_sample_dict_stg3,
        smi_eta=None,
        random_anchor=True,
    )
    log_q_stg3 = (
        jax.lax.stop_gradient(q_distr_out['global_params_log_prob']) +
        q_distr_out['loc_random_anchor_log_prob'])

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


def log_images(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    config: ConfigDict,
    show_basis_fields: bool,
    show_linguistic_fields: bool,
    num_loc_random_anchor_plot: Optional[int],
    num_loc_floating_plot: Optional[int],
    show_mixing_weights: bool,
    show_loc_given_y: bool,
    use_gamma_anchor: bool = False,
    suffix: Optional[str] = None,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample from variational posterior
  # Sample from variational posteriors
  q_distr_out = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      batch=batch,
      prng_key=next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_plot,),
      include_random_anchor=config.include_random_anchor,
      kernel_name=config.kernel_name,
      kernel_kwargs=config.kernel_kwargs,
      num_samples_gamma_profiles=(config.num_samples_gamma_profiles
                                  if use_gamma_anchor else 0),
      gp_jitter=config.gp_jitter,
  )

  plot.posterior_samples(
      posterior_sample_dict=q_distr_out['posterior_sample'],
      batch=batch,
      prng_key=next(prng_seq),
      kernel_name=config.kernel_name,
      kernel_kwargs=config.kernel_kwargs,
      gp_jitter=config.gp_jitter,
      step=state_list[0].step,
      show_basis_fields=show_basis_fields,
      show_linguistic_fields=show_linguistic_fields,
      num_loc_random_anchor_plot=(num_loc_random_anchor_plot
                                  if config.include_random_anchor else None),
      num_loc_floating_plot=num_loc_floating_plot,
      show_mixing_weights=show_mixing_weights,
      show_loc_given_y=show_loc_given_y,
      suffix=suffix,
      summary_writer=summary_writer,
      workdir_png=workdir_png,
      use_gamma_anchor=use_gamma_anchor,
  )


def error_locations_estimate(
    params_tuple: Tuple[hk.Params],
    batch: Optional[Batch],
    prng_key: PRNGKey,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    include_random_anchor: bool,
    kernel_name: Optional[str] = None,
    kernel_kwargs: Optional[Dict[str, Any]] = None,
    num_samples_gamma_profiles: int = 0,
    gp_jitter: Optional[float] = None,
) -> Dict[str, Array]:
  """Compute average distance error."""

  # Sample from flow
  posterior_sample_dict = sample_all_flows(
      params_tuple=params_tuple,
      batch=batch,
      prng_key=prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples,),
      include_random_anchor=include_random_anchor,
      kernel_name=kernel_name,
      kernel_kwargs=kernel_kwargs,
      num_samples_gamma_profiles=num_samples_gamma_profiles,
      gp_jitter=gp_jitter,
  )['posterior_sample']

  error_loc_out = {}

  # Anchor profiles
  if 'loc_random_anchor' in posterior_sample_dict:
    predictions = posterior_sample_dict['loc_random_anchor']
    targets = jnp.broadcast_to(
        batch['loc'][:batch['num_profiles_anchor'], :],
        shape=posterior_sample_dict['loc_random_anchor'].shape)
    distances = jnp.linalg.norm(predictions - targets, ord=2, axis=-1)
    error_loc_out['distance_random_anchor'] = distances.mean()

  # Floating profiles
  predictions = posterior_sample_dict['loc_floating']
  targets = jnp.broadcast_to(
      batch['loc'][batch['num_profiles_anchor']:, :],
      shape=posterior_sample_dict['loc_floating'].shape)
  distances = jnp.linalg.norm(predictions - targets, ord=2, axis=-1)
  error_loc_out['distance_floating'] = distances.mean()

  return error_loc_out


def train_and_evaluate(config: ConfigDict, workdir: str) -> None:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)

  # Full dataset used everytime
  # No batching for now
  dataset = load_data(config=config)
  # Add some parameters to config
  config.num_profiles = dataset['num_profiles']
  config.num_profiles_anchor = dataset['num_profiles_anchor']
  config.num_profiles_floating = dataset['num_profiles_floating']
  config.num_forms_tuple = dataset['num_forms_tuple']
  config.num_inducing_points = math.prod(config.flow_kwargs.inducing_grid_shape)

  # For training, we need a Dictionary compatible with jit
  # we remove string vector
  train_ds = {k: v for k, v in dataset.items() if k not in ['items', 'forms']}

  # Compute GP covariance between anchor profiles
  train_ds['cov_anchor'] = getattr(
      kernels, config.kernel_name)(**config.kernel_kwargs).matrix(
          x1=train_ds['loc'][:train_ds['num_profiles_anchor'], :],
          x2=train_ds['loc'][:train_ds['num_profiles_anchor'], :],
      )

  train_ds = get_inducing_points(dataset=train_ds, config=config)

  smi_eta = dict(config.flow_kwargs.smi_eta)
  is_smi = any(v is not None for v in smi_eta.values())
  if is_smi:
    if 'profiles_floating' in smi_eta:
      smi_eta['profiles'] = jnp.where(
          jnp.arange(train_ds['num_profiles']) <
          train_ds['num_profiles_anchor'],
          1.,
          smi_eta['profiles_floating'],
      )
      del smi_eta['profiles_floating']
    smi_eta['items'] = jnp.ones(len(train_ds['num_forms_tuple']))

  # These parameters affect the dimension of the flow
  # so they are also part of the flow parameters
  config.flow_kwargs.num_profiles_anchor = dataset['num_profiles_anchor']
  config.flow_kwargs.num_profiles_floating = dataset['num_profiles_floating']
  config.flow_kwargs.num_forms_tuple = dataset['num_forms_tuple']
  config.flow_kwargs.num_inducing_points = int(
      math.prod(config.flow_kwargs.inducing_grid_shape))
  config.flow_kwargs.is_smi = is_smi
  # Get locations bounds
  # These define the range of values produced by the posterior of locations
  loc_bounds = np.stack(
      [dataset['loc'].min(axis=0), dataset['loc'].max(axis=0)],
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
  global_params_base_sample_init = hk.transform(q_distr_global).apply(
      state_list[0].params,
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_elbo,),
  )['global_params_base_sample']

  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[1]}',
          forward_fn=hk.transform(q_distr_loc_floating),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'global_params_base_sample': global_params_base_sample_init,
              'is_aux': False,
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
              'global_params_base_sample': global_params_base_sample_init,
              'is_aux': True,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0 and state_list[0].step < config.training_steps:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))
  else:
    summary_writer = None

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
          global_params_base_sample=global_params_base_sample_init,
          is_aux=False,
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
                'global_params_base_sample': global_params_base_sample_init,
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
            global_params_base_sample=global_params_base_sample_init,
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

  update_states_jit = lambda state_list, batch, prng_key, smi_eta: update_states(
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
          'kernel_name': config.kernel_name,
          'kernel_kwargs': config.kernel_kwargs,
          'num_samples_gamma_profiles': config.num_samples_gamma_profiles,
          'gp_jitter': config.gp_jitter,
      },
  )
  # globals().update(loss_fn_kwargs)
  update_states_jit = jax.jit(update_states_jit)

  elbo_validation_jit = lambda state_list, batch, prng_key, smi_eta: elbo_estimate(
      params_tuple=[state.params for state in state_list],
      batch=batch,
      prng_key=prng_key,
      num_samples=config.num_samples_eval,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      smi_eta=smi_eta,
      include_random_anchor=config.include_random_anchor,
      kernel_name=config.kernel_name,
      kernel_kwargs=config.kernel_kwargs,
      num_samples_gamma_profiles=config.num_samples_gamma_profiles,
      gp_jitter=config.gp_jitter,
  )
  elbo_validation_jit = jax.jit(elbo_validation_jit)

  error_locations_estimate_jit = lambda state_list, batch, prng_key: error_locations_estimate(
      params_tuple=[state.params for state in state_list],
      batch=batch,
      prng_key=prng_key,
      num_samples=config.num_samples_eval,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      include_random_anchor=config.include_random_anchor,
      kernel_name=config.kernel_name,
      kernel_kwargs=config.kernel_kwargs,
      num_samples_gamma_profiles=config.num_samples_gamma_profiles,
      gp_jitter=config.gp_jitter,
  )
  # TODO: This doesn't work after jitting.
  # error_locations_estimate_jit = jax.jit(error_locations_estimate_jit)

  if state_list[0].step < config.training_steps:
    logging.info('Training variational posterior...')
    # Reset random keys
    prng_seq = hk.PRNGSequence(config.seed)

  while state_list[0].step < config.training_steps:
    # step = 0

    # Plots to monitor training
    if (state_list[0].step == 0) or (state_list[0].step % config.log_img_steps
                                     == 0):
      # print("Logging images...\n")
      log_images(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
          config=config,
          show_basis_fields=config.show_basis_fields_during_training,
          show_linguistic_fields=config.show_linguistic_fields_during_training,
          num_loc_random_anchor_plot=5,
          num_loc_floating_plot=5,
          show_mixing_weights=False,
          show_loc_given_y=False,
          use_gamma_anchor=False,
          suffix=f"eta_floating_{float(smi_eta['profiles'][-1]):.3f}",
          summary_writer=summary_writer,
          workdir_png=workdir,
      )

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
      error_loc_dict = error_locations_estimate_jit(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
      )
      for k, v in error_loc_dict.items():
        summary_writer.scalar(
            tag=k,
            value=v.mean(),
            step=state_list[0].step,
        )

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
  # (in case training_steps is not multiple of checkpoint_steps)
  for state, state_name in zip(state_list, state_name_list):
    save_checkpoint(
        state=state,
        checkpoint_dir=f'{checkpoint_dir}/{state_name}',
        keep=config.checkpoints_keep,
    )
  del state

  # Last plot of posteriors
  log_images(
      state_list=state_list,
      batch=train_ds,
      prng_key=next(prng_seq),
      config=config,
      show_basis_fields=True,
      show_linguistic_fields=True,
      num_loc_random_anchor_plot=20,
      num_loc_floating_plot=20,
      show_mixing_weights=False,
      show_loc_given_y=False,
      use_gamma_anchor=False,
      suffix=f"eta_floating_{float(smi_eta['profiles'][-1]):.3f}",
      summary_writer=summary_writer,
      workdir_png=workdir,
  )


# # For debugging
# config = get_config()
# config.flow_kwargs.smi_eta.update({
#     'profiles_floating': 1.000,
# })
# # workdir = pathlib.Path.home() / 'spatial-smi/output/8_items/mean_field/eta_floating_1.000'
# workdir = pathlib.Path.home() / 'spatial-smi/output/8_items/spline/eta_floating_1.000'
# train_and_evaluate(config, workdir)
