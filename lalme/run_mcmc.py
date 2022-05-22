"""MCMC sampling for the LALME model."""

import time
import math

from absl import logging

import numpy as np

import jax
from jax import numpy as jnp

import haiku as hk
import distrax

from tensorflow_probability.substrates import jax as tfp

from flax.metrics import tensorboard

import log_prob_fun
import plot
from train_flow import load_data, get_inducing_points
from flows import (split_flow_global_params, split_flow_locations,
                   concat_samples_global_params, concat_samples_locations)

import modularbayes
from modularbayes import flatten_dict
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict,
                                      Mapping, Optional, OrderedDict, PRNGKey,
                                      Tuple)

tfd = tfp.distributions
tfb = tfp.bijectors
tfm = tfp.mcmc
kernels = tfp.math.psd_kernels

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def get_posterior_sample_init_stg1(
    num_forms_tuple: Tuple[int],
    num_basis_gps: int,
    num_inducing_points: int,
    num_profiles_floating: int,
    num_profiles_anchor: Optional[int] = None,
):
  """Get dictionary with parametes to initialize MCMC.

  The order of the parameters
  """

  num_samples = 1
  num_items = len(num_forms_tuple)

  global_params_names = [
      'gamma_inducing', 'mixing_weights_list', 'mixing_offset_list', 'mu',
      'zeta'
  ]
  all_params_names = global_params_names + ['loc_floating']
  posterior_sample = OrderedDict()
  for key in all_params_names:
    posterior_sample[key] = None

  ### Global parameters ###
  gamma_inducing_dim = num_basis_gps * num_inducing_points
  mixing_weights_dims = [
      num_basis_gps * num_forms_i for num_forms_i in num_forms_tuple
  ]
  mixing_weights_dim = sum(mixing_weights_dims)
  mixing_offset_dim = sum(num_forms_tuple)
  mu_dim = num_items
  zeta_dim = num_items
  flow_dim = (
      gamma_inducing_dim + mixing_weights_dim + mixing_offset_dim + mu_dim +
      zeta_dim)
  posterior_sample_global = split_flow_global_params(
      samples=jnp.zeros((num_samples, flow_dim)),
      num_forms_tuple=num_forms_tuple,
      num_basis_gps=num_basis_gps,
      num_inducing_points=num_inducing_points,
  )

  for key in global_params_names:
    posterior_sample[key] = posterior_sample_global[key]

  ### Location floating profiles ###
  posterior_sample_loc_floating = split_flow_locations(
      samples=jnp.zeros((num_samples, 2 * num_profiles_floating)),
      num_profiles=num_profiles_floating,
      name='loc_floating',
      is_aux=False,
  )

  posterior_sample['loc_floating'] = posterior_sample_loc_floating[
      'loc_floating']

  # Verify that the parameters are in the correct order
  assert list(posterior_sample.keys()) == all_params_names

  return posterior_sample


def get_kernel_bijector_stg1(
    num_forms_tuple: int,
    num_profiles_floating: int,
    num_basis_gps: int,
    inducing_grid_shape: Tuple[int],
    loc_x_range: Tuple[float],
    loc_y_range: Tuple[float],
):
  """Define kernel bijector for stage 1.

  Define bijectors for mapping values to parameter domain.
    -gamma goes to [-Inf,Inf]
    -mixing_weights go to [-Inf,Inf]
    -mixing_offset go to [-Inf,Inf]
    -mu goes to [0,Inf]
    -zeta goes to [0,1]
  """

  num_items = len(num_forms_tuple)

  ### Bijectors for Global Parameters ###
  num_inducing_points = math.prod(inducing_grid_shape)

  gamma_inducing_dim = num_basis_gps * num_inducing_points
  mixing_weights_dim = sum(
      [num_basis_gps * num_forms_i for num_forms_i in num_forms_tuple])
  mixing_offset_dim = sum(num_forms_tuple)
  mu_dim = num_items
  zeta_dim = num_items

  block_bijector_global = [
      # gamma: Identity [-Inf,Inf]
      tfb.Identity(),
      # mixing_weights: Identity [-Inf,Inf]
      tfb.Identity(),
      # mixing_offset: Identity [-Inf,Inf]
      tfb.Identity(),
      # mu: Softplus [0,Inf]
      tfb.Softplus(),
      # zeta: Sigmoid [0,1]
      tfb.Sigmoid(),
  ]
  block_sizes_global = [
      gamma_inducing_dim,
      mixing_weights_dim,
      mixing_offset_dim,
      mu_dim,
      zeta_dim,
  ]
  bijector_global = tfb.Blockwise(
      bijectors=block_bijector_global, block_sizes=block_sizes_global)

  ### Bijectors for Profile Locations ###
  bijector_loc_floating_layers = []
  # First, all locations to the [0,1] square
  bijector_loc_floating_layers.append(distrax.Block(distrax.Sigmoid(), 1))
  # profiles x's go to [0,loc_x_max]
  # profiles y's go to [0,loc_y_max]
  if loc_x_range == (0., 1.):
    loc_x_range_bijector = tfb.Identity()
  else:
    loc_x_range_bijector = tfb.Scale(scale=loc_x_range[1] - loc_x_range[0])
    # TODO(chrcarm): enable shift
    # loc_x_range_bijector = tfp.Shift(shift=loc_x_range[0])

  if loc_y_range == (0., 1.):
    loc_y_range_bijector = tfb.Identity()
  else:
    loc_y_range_bijector = tfb.Scale(scale=loc_y_range[1] - loc_y_range[0])
    # TODO(chrcarm): enable shift
    # loc_y_range_bijector = tfp.Shift(shift=loc_y_range[0])

  block_bijectors_loc_floating = [loc_x_range_bijector, loc_y_range_bijector]
  block_sizes_loc_floating = [num_profiles_floating, num_profiles_floating]
  bijector_loc_floating_layers.append(
      tfb.Blockwise(
          bijectors=block_bijectors_loc_floating,
          block_sizes=block_sizes_loc_floating))
  bijector_loc_floating = distrax.Chain(bijector_loc_floating_layers[::-1])

  bijector_loc_floating = tfb.Blockwise(
      bijectors=block_bijectors_loc_floating,
      block_sizes=block_sizes_loc_floating)

  kernel_bijector = tfb.Blockwise(
      bijectors=[bijector_global, bijector_loc_floating],
      block_sizes=[sum(block_sizes_global),
                   sum(block_sizes_loc_floating)])

  return kernel_bijector


def get_kernel_bijector_stg2(
    num_basis_gps: int,
    inducing_grid_shape: Tuple[int],
    num_forms_tuple: int,
    num_items: int,
):
  """Define kernel bijector for stage 1.

  Define bijectors for mapping values to parameter domain.
    -gamma goes to [-Inf,Inf]
    -mixing_weights go to [-Inf,Inf]
    -mixing_offset go to [-Inf,Inf]
    -mu goes to [0,Inf]
    -zeta goes to [0,1]
  """

  num_inducing_points = math.prod(inducing_grid_shape)

  gamma_inducing_dim = num_basis_gps * num_inducing_points
  mixing_weights_dim = sum(
      [num_basis_gps * num_forms_i for num_forms_i in num_forms_tuple])
  mixing_offset_dim = sum(num_forms_tuple)
  mu_dim = num_items
  zeta_dim = num_items

  block_bijectors = [
      tfb.Identity(),
      tfb.Identity(),
      tfb.Identity(),
      tfb.Softplus(),
      distrax.Sigmoid(),
  ]
  block_sizes = [
      gamma_inducing_dim,
      mixing_weights_dim,
      mixing_offset_dim,
      mu_dim,
      zeta_dim,
  ]

  kernel_bijector = tfb.Blockwise(
      bijectors=block_bijectors, block_sizes=block_sizes)

  return kernel_bijector


def concat_samples_stg1(samples_dict: Dict[str, Any],) -> Array:

  samples = []

  samples.append(concat_samples_global_params(samples_dict))
  samples.append(
      concat_samples_locations(
          samples_dict=samples_dict,
          is_aux=False,
          name='loc_floating',
      ))
  samples = jnp.concatenate(samples, axis=-1)

  return samples


def log_prob_fn(
    batch: Batch,
    prng_key: PRNGKey,
    posterior_sample_raw: Array,
    prior_hparams: Mapping[str, Any],
    kernel_name: str,
    kernel_kwargs: Mapping[str, Any],
    num_samples_gamma_profiles: int,
    smi_eta_profiles: Optional[Array],
    posterior_sample_dict_init: Mapping[str, Any],
    include_random_anchor=False,
    gp_jitter: float = 1e-5,
):
  """Log probability function for the random effects model."""

  # Use the tree structure in posterior_sample_dict to split the samples in
  # posterior_sample_raw and create a dictionary with named samples.
  leaves_init, treedef = jax.tree_util.tree_flatten(posterior_sample_dict_init)
  leaves = []
  for i in range(len(leaves_init) - 1):
    param_i, posterior_sample_raw = jnp.split(
        posterior_sample_raw, leaves_init[i].flatten().shape, axis=-1)
    leaves.append(param_i.reshape(leaves_init[i].shape))
  leaves.append(posterior_sample_raw.reshape(leaves_init[-1].shape))
  posterior_sample_dict = jax.tree_util.tree_unflatten(
      treedef=treedef, leaves=leaves)

  # Put Smi eta values into a dictionary.
  smi_eta = {
      'profiles': smi_eta_profiles
  } if smi_eta_profiles is not None else None
  smi_eta['items'] = jnp.ones(len(batch['num_forms_tuple']))

  # Sample the basis GPs on profiles locations conditional on GP values on the
  # inducing points.
  gamma_sample_dict = log_prob_fun.sample_gamma_profiles_given_gamma_inducing(
      batch=batch,
      posterior_sample_dict=posterior_sample_dict,
      prng_key=prng_key,
      kernel_name=kernel_name,
      kernel_kwargs=kernel_kwargs,
      gp_jitter=gp_jitter,
      num_samples_gamma_profiles=num_samples_gamma_profiles,
      is_smi=False,  # Do not sample the auxiliary gamma
      include_random_anchor=include_random_anchor,
  )
  posterior_sample_dict.update(gamma_sample_dict)

  # Compute the log probability function.
  log_prob = log_prob_fun.log_prob_joint(
      batch=batch,
      posterior_sample_dict=posterior_sample_dict,
      smi_eta=smi_eta,
      random_anchor=False,
      **prior_hparams,
  ).squeeze()
  # globals().update(prior_hparams)

  return log_prob


def sample_and_evaluate(config: ConfigDict, workdir: str) -> Mapping[str, Any]:
  """Sample and evaluate the random effects model."""

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
  config.num_inducing_points = math.prod(
      config.model_hparams.inducing_grid_shape)

  # For training, we need a Dictionary compatible with jit
  # we remove string vectors
  train_ds = {k: v for k, v in dataset.items() if k not in ['items', 'forms']}

  # Compute GP covariance between anchor profiles
  train_ds['cov_anchor'] = getattr(
      kernels, config.kernel_name)(**config.kernel_kwargs).matrix(
          x1=train_ds['loc'][:train_ds['num_profiles_anchor'], :],
          x2=train_ds['loc'][:train_ds['num_profiles_anchor'], :],
      )

  train_ds = get_inducing_points(
      dataset=train_ds,
      inducing_grid_shape=config.model_hparams.inducing_grid_shape,
      kernel_name=config.kernel_name,
      kernel_kwargs=config.kernel_kwargs,
      gp_jitter=config.gp_jitter,
  )

  smi_eta = dict(config.smi_eta)
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

  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))

  # Initilize the model parameters
  posterior_sample_dict_init = get_posterior_sample_init_stg1(
      num_forms_tuple=config.num_forms_tuple,
      num_basis_gps=config.model_hparams.num_basis_gps,
      num_inducing_points=config.num_inducing_points,
      num_profiles_floating=config.num_profiles_floating,
      num_profiles_anchor=config.num_profiles_anchor,
  )

  ### Sample First Stage ###

  logging.info("\t sampling stage 1...")

  times_data = {}
  times_data['start_sampling'] = time.perf_counter()

  posterior_sample_raw_init = concat_samples_stg1(
      samples_dict=posterior_sample_dict_init,)[0, :]

  prng_key_gamma = next(prng_seq)
  target_log_prob_fn_stg1_jit = lambda posterior_sample_raw: log_prob_fn(
      batch=train_ds,
      prng_key=prng_key_gamma,
      posterior_sample_raw=posterior_sample_raw,
      prior_hparams=config.prior_hparams,
      kernel_name=config.kernel_name,
      kernel_kwargs=config.kernel_kwargs,
      num_samples_gamma_profiles=config.num_samples_gamma_profiles,
      smi_eta_profiles=smi_eta['profiles'] if smi_eta is not None else None,
      posterior_sample_dict_init=posterior_sample_dict_init,
      include_random_anchor=False,
      gp_jitter=config.gp_jitter,
  )
  target_log_prob_fn_stg1_jit = jax.jit(target_log_prob_fn_stg1_jit)

  # Define bijector for samples
  bijector_stg1 = get_kernel_bijector_stg1(
      num_forms_tuple=config.num_forms_tuple,
      num_profiles_floating=config.num_profiles_floating,
      **config.model_hparams,
  )

  # Test the log probability function
  _ = target_log_prob_fn_stg1_jit(
      posterior_sample_raw=bijector_stg1.forward(posterior_sample_raw_init))

  # TODO(chris): Continue from here

  # Define sampling kernel
  kernel = tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=tfm.NoUTurnSampler(
          target_log_prob_fn=target_log_prob_fn_stg1_jit,
          step_size=config.mcmc_step_size,
      ),
      bijector=bijector_stg1)

  # Sample stage 1 chain
  times_data['start_mcmc_stg_1'] = time.perf_counter()
  posterior_sample = tfm.sample_chain(
      num_results=config.num_samples,
      num_burnin_steps=config.num_burnin_steps,
      kernel=kernel,
      current_state=bijector_stg1.forward(posterior_sample_raw_init),
      trace_fn=None,
      seed=next(prng_seq),
  )

  posterior_sample_dict = {}
  (posterior_sample_dict['sigma'], posterior_sample_dict['beta'],
   posterior_sample_dict['tau']) = jnp.split(
       posterior_sample, [config.num_groups, 2 * config.num_groups], axis=-1)

  logging.info("posterior means sigma %s",
               str(posterior_sample_dict['sigma'].mean(axis=0)))

  times_data['end_mcmc_stg_1'] = time.perf_counter()

  ### Sample Second Stage ###
  if smi_eta is not None:
    # Define parameters split for SMI
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
    for key in refit_params_names:
      posterior_sample_dict[key + '_aux'] = posterior_sample_dict[key]
      del posterior_sample_dict[key]

    logging.info("posterior means beta_aux %s",
                 str(posterior_sample_dict['beta_aux'].mean(axis=0)))

    logging.info("\t sampling stage 2...")

    def sample_stg2(
        sigma: Array,
        beta_init: Array,
        tau_init: Array,
        num_burnin_steps: int,
        prng_key: PRNGKey,
    ):
      target_log_prob_fn_stage2 = lambda state: log_prob_fn(
          batch=train_ds,
          posterior_sample_raw=jnp.concatenate([sigma, state], axis=-1),
          prior_hparams=config.prior_hparams,
          smi_eta_profiles=jnp.array(1.),
          posterior_sample_dict=posterior_sample_dict_init,
      )
      inner_kernel = tfm.NoUTurnSampler(
          target_log_prob_fn=target_log_prob_fn_stage2,
          step_size=config.mcmc_step_size,
      )
      block_bijectors = [
          tfb.Identity(),
          tfb.Softplus(),
      ]
      block_sizes = [
          config.num_groups,
          1,
      ]
      kernel_bijectors = [
          tfb.Blockwise(bijectors=block_bijectors, block_sizes=block_sizes)
      ]

      kernel = tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=inner_kernel, bijector=kernel_bijectors)

      posterior_sample = tfm.sample_chain(
          num_results=1,
          num_burnin_steps=num_burnin_steps,
          kernel=kernel,
          current_state=jnp.concatenate([beta_init, tau_init], axis=-1),
          trace_fn=None,
          seed=prng_key,
      )

      return posterior_sample

    # Get one sample of parameters in stage 2
    # sample_stg2(
    #     phi=posterior_sample_dict['phi'][0, :],
    #     theta_init=posterior_sample_dict['theta_aux'][0, :],
    #     num_burnin_steps=100,
    #     prng_key=next(prng_seq),
    # )

    # Define function to parallelize sample_stage2
    # TODO(chris): use pmap
    sample_stg2_vmap = jax.vmap(
        lambda sigma, beta_init, tau_init, prng_key: sample_stg2(
            sigma=sigma,
            beta_init=beta_init,
            tau_init=tau_init,
            num_burnin_steps=config.num_samples_subchain - 1,
            prng_key=prng_key,
        ))

    def _sample_stage2_loop(num_chunks):
      """Sequential sampling of stage 2.

      Improve performance of HMC by spplitting the total number of samples into
      num_chunks steps, using initialize the chains in values obtained from the
      previous step.
      """

      assert (config.num_samples % num_chunks) == 0

      # Initialize beta and tau
      beta = [posterior_sample_dict['beta_aux'][:num_chunks, :]]
      tau = [posterior_sample_dict['tau_aux'][:num_chunks, :]]

      for i in range(int(config.num_samples / num_chunks)):
        beta_tau_i = sample_stg2_vmap(
            posterior_sample_dict['sigma'][(i * num_chunks):((i + 1) *
                                                             num_chunks), :],
            beta[-1],
            tau[-1],
            jax.random.split(next(prng_seq), num_chunks),
        )
        beta_i, tau_i = jnp.split(
            beta_tau_i.squeeze(1), [config.num_groups], axis=-1)
        beta.append(beta_i)
        tau.append(tau_i)

      beta = jnp.concatenate(beta[1:], axis=0)
      tau = jnp.concatenate(tau[1:], axis=0)

      return beta, tau

    # Sample beta and tau
    times_data['start_mcmc_stg_2'] = time.perf_counter()
    posterior_sample_dict['beta'], posterior_sample_dict[
        'tau'] = _sample_stage2_loop(num_chunks=config.num_samples // 20)

  logging.info("posterior means beta %s",
               str(posterior_sample_dict['beta'].mean(axis=0)))

  times_data['end_mcmc_stg_2'] = time.perf_counter()

  times_data['end_sampling'] = time.perf_counter()

  logging.info("Sampling times:")
  logging.info("\t Total: %s",
               str(times_data['end_sampling'] - times_data['start_sampling']))
  logging.info(
      "\t Stg 1: %s",
      str(times_data['end_mcmc_stg_1'] - times_data['start_mcmc_stg_1']))
  if smi_eta is not None:
    logging.info(
        "\t Stg 2: %s",
        str(times_data['end_mcmc_stg_2'] - times_data['start_mcmc_stg_2']))

  logging.info("Plotting results...")
  ### Plot SMI samples ###
  plot.posterior_samples(
      posterior_sample_dict=posterior_sample_dict,
      batch=train_ds,
      prng_key=next(prng_seq),
      kernel_name=config.kernel_name,
      kernel_kwargs=config.kernel_kwargs,
      gp_jitter=config.gp_jitter,
      step=0,
      show_basis_fields=True,
      show_linguistic_fields=True,
      num_loc_random_anchor_plot=(20 if config.include_random_anchor else None),
      num_loc_floating_plot=5,
      show_mixing_weights=False,
      show_loc_given_y=False,
      suffix=config.plot_suffix,
      summary_writer=summary_writer,
      workdir_png=workdir,
      use_gamma_anchor=False,
  )

  # j = 1
  # fig, axs = plt.subplots(2, 1)
  # axs[0].hist(posterior_sample_dict['beta'][:, j], 30)
  # axs[1].plot(posterior_sample_dict['beta'][:, j])

  return posterior_sample_dict


# # For debugging
# config = get_config()
# config.smi_eta.update({
#     'profiles_floating': 1.000,
# })
# import pathlib
# workdir = pathlib.Path.home() / 'spatial-smi/output/8_items/mcmc/eta_floating_1.000'
# sample_and_evaluate(config, workdir)
