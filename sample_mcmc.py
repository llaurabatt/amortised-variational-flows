"""MCMC sampling for the LALME model."""

import os

import time
import math

from absl import logging

import numpy as np

import jax
from jax import numpy as jnp

import haiku as hk

from tensorflow_probability.substrates import jax as tfp

from flax.metrics import tensorboard

import log_prob_fun
import plot
from train_flow import load_data, get_inducing_points
from flows import (split_flow_global_params, split_flow_locations,
                   concat_samples_global_params, concat_samples_locations,
                   get_global_params_dim)

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


def split_samples(
    samples: Array,
    num_forms_tuple: Tuple[int],
    num_basis_gps: int,
    num_inducing_points: int,
    num_profiles_floating: int,
):
  """Get dictionary with parametes to initialize MCMC.

  The order of the parameters
  """

  global_params_names = [
      'gamma_inducing', 'mixing_weights_list', 'mixing_offset_list', 'mu',
      'zeta'
  ]
  all_params_names = global_params_names + ['loc_floating']
  posterior_sample = OrderedDict()
  for key in all_params_names:
    posterior_sample[key] = None

  global_params_dim = get_global_params_dim(
      num_forms_tuple=num_forms_tuple,
      num_basis_gps=num_basis_gps,
      num_inducing_points=num_inducing_points,
  )
  assert samples.shape[-1] == global_params_dim + 2 * num_profiles_floating

  ### Global parameters ###
  samples_global, samples_loc_floating = jnp.split(
      samples, [global_params_dim], axis=-1)

  posterior_sample_global = split_flow_global_params(
      samples=samples_global,
      num_forms_tuple=num_forms_tuple,
      num_basis_gps=num_basis_gps,
      num_inducing_points=num_inducing_points,
  )

  for key in global_params_names:
    posterior_sample[key] = posterior_sample_global[key]

  ### Location floating profiles ###
  posterior_sample_loc_floating = split_flow_locations(
      samples=samples_loc_floating,
      num_profiles=num_profiles_floating,
      name='loc_floating',
      is_aux=False,
  )

  posterior_sample['loc_floating'] = posterior_sample_loc_floating[
      'loc_floating']

  # Verify that the parameters are in the correct order
  assert list(posterior_sample.keys()) == all_params_names

  return posterior_sample


def concat_samples(samples_dict: Dict[str, Any]) -> Array:

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


def get_posterior_sample_init_stg1(
    num_forms_tuple: Tuple[int],
    num_basis_gps: int,
    num_inducing_points: int,
    num_profiles_floating: int,
):
  """Get dictionary with parametes to initialize MCMC.

  The order of the parameters
  """

  num_samples = 1
  global_params_names = [
      'gamma_inducing', 'mixing_weights_list', 'mixing_offset_list', 'mu',
      'zeta'
  ]
  all_params_names = global_params_names + ['loc_floating']
  posterior_sample = OrderedDict()
  for key in all_params_names:
    posterior_sample[key] = None

  ### Global parameters ###
  samples_dim = get_global_params_dim(
      num_forms_tuple=num_forms_tuple,
      num_basis_gps=num_basis_gps,
      num_inducing_points=num_inducing_points,
  ) + 2 * num_profiles_floating

  posterior_sample = split_samples(
      samples=jnp.zeros((num_samples, samples_dim)),
      num_forms_tuple=num_forms_tuple,
      num_basis_gps=num_basis_gps,
      num_inducing_points=num_inducing_points,
      num_profiles_floating=num_profiles_floating,
  )

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
  bijector_loc_floating_layers.append(tfb.Sigmoid())
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
  bijector_loc_floating = tfb.Chain(bijector_loc_floating_layers[::-1])

  kernel_bijector = tfb.Blockwise(
      bijectors=[bijector_global, bijector_loc_floating],
      block_sizes=[sum(block_sizes_global),
                   sum(block_sizes_loc_floating)])

  return kernel_bijector


def get_kernel_bijector_stg2(
    num_profiles_floating: int,
    loc_x_range: Tuple[float],
    loc_y_range: Tuple[float],
    **kwargs,
):
  """Define kernel bijector for stage 2."""

  ### Bijectors for Profile Locations ###
  bijector_loc_floating_layers = []
  # First, all locations to the [0,1] square
  bijector_loc_floating_layers.append(tfb.Sigmoid())
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
  bijector_loc_floating = tfb.Chain(bijector_loc_floating_layers[::-1])

  return bijector_loc_floating


def log_prob_fn(
    batch: Batch,
    prng_key: PRNGKey,
    posterior_sample_raw_1d: Array,
    prior_hparams: Mapping[str, Any],
    kernel_name: str,
    kernel_kwargs: Mapping[str, Any],
    num_samples_gamma_profiles: int,
    smi_eta_profiles: Optional[Array],
    num_forms_tuple: int,
    num_basis_gps: int,
    num_inducing_points: int,
    num_profiles_floating: int,
    include_random_anchor: bool = False,
    gp_jitter: float = 1e-5,
):

  # Split the array of raw samples to create a dictionary with named samples.
  posterior_sample_dict = split_samples(
      samples=jnp.expand_dims(posterior_sample_raw_1d, axis=0),
      num_forms_tuple=num_forms_tuple,
      num_basis_gps=num_basis_gps,
      num_inducing_points=num_inducing_points,
      num_profiles_floating=num_profiles_floating,
  )

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

  samples_path_stg1 = workdir + '/posterior_sample_dict_stg1.npz'
  samples_path_stg2 = workdir + '/posterior_sample_dict_stg2.npz'
  samples_path = workdir + '/posterior_sample_dict.npz'

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

  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))

  # Initilize the model parameters
  posterior_sample_dict_init = get_posterior_sample_init_stg1(
      num_forms_tuple=config.num_forms_tuple,
      num_basis_gps=config.model_hparams.num_basis_gps,
      num_inducing_points=config.num_inducing_points,
      num_profiles_floating=config.num_profiles_floating,
  )

  times_data = {}
  times_data['start_sampling'] = time.perf_counter()

  ### Sample First Stage ###
  if os.path.exists(samples_path_stg1):
    logging.info("\t Loading samples for stage 1...")
    posterior_sample_dict = np.load(
        str(samples_path_stg1), allow_pickle=True)['arr_0'].item()
  else:
    logging.info("\t sampling stage 1...")

    posterior_sample_raw_init = concat_samples(
        samples_dict=posterior_sample_dict_init)[0, :]

    # Verify that split_samples is the inverse of concat_samples
    assert all([
        v for v in jax.tree_map(
            lambda x, y: (x == y).all(),
            posterior_sample_dict_init,
            split_samples(
                samples=jnp.expand_dims(posterior_sample_raw_init, 0),
                num_forms_tuple=config.num_forms_tuple,
                num_basis_gps=config.model_hparams.num_basis_gps,
                num_inducing_points=config.num_inducing_points,
                num_profiles_floating=config.num_profiles_floating,
            ),
        ).values()
    ])

    prng_key_gamma = next(prng_seq)

    @jax.jit
    def target_log_prob_fn_stg1(posterior_sample_raw):
      return log_prob_fn(
          batch=train_ds,
          prng_key=prng_key_gamma,
          posterior_sample_raw_1d=posterior_sample_raw,
          prior_hparams=config.prior_hparams,
          kernel_name=config.kernel_name,
          kernel_kwargs=config.kernel_kwargs,
          num_samples_gamma_profiles=config.num_samples_gamma_profiles,
          smi_eta_profiles=smi_eta['profiles'] if smi_eta is not None else None,
          num_forms_tuple=config.num_forms_tuple,
          num_basis_gps=config.model_hparams.num_basis_gps,
          num_inducing_points=config.num_inducing_points,
          num_profiles_floating=config.num_profiles_floating,
          include_random_anchor=False,
          gp_jitter=config.gp_jitter,
      )

    # Define bijector for samples
    bijector_stg1 = get_kernel_bijector_stg1(
        num_forms_tuple=config.num_forms_tuple,
        num_profiles_floating=config.num_profiles_floating,
        **config.model_hparams,
    )

    # Test the log probability function
    _ = target_log_prob_fn_stg1(
        posterior_sample_raw=bijector_stg1.forward(posterior_sample_raw_init))

    # Define sampling kernel
    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfm.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn_stg1,
            step_size=config.mcmc_step_size,
        ),
        bijector=bijector_stg1)

    # Sample stage 1 chain

    times_data['start_mcmc_stg_1'] = time.perf_counter()
    posterior_sample_stg1 = tfm.sample_chain(
        num_results=config.num_samples,
        num_burnin_steps=config.num_burnin_steps_stg1,
        kernel=kernel,
        current_state=bijector_stg1.forward(posterior_sample_raw_init),
        trace_fn=None,
        seed=next(prng_seq),
    )

    posterior_sample_dict = split_samples(
        samples=posterior_sample_stg1,
        num_forms_tuple=config.num_forms_tuple,
        num_basis_gps=config.model_hparams.num_basis_gps,
        num_inducing_points=config.num_inducing_points,
        num_profiles_floating=config.num_profiles_floating,
    )
    # Verify that concat_samples is the inverse of split_samples
    assert (posterior_sample_stg1 == concat_samples(
        samples_dict=posterior_sample_dict)).all()
    global_params_sample, loc_floating_aux_sample = jnp.split(
        posterior_sample_stg1, [
            get_global_params_dim(
                num_forms_tuple=config.num_forms_tuple,
                num_basis_gps=config.model_hparams.num_basis_gps,
                num_inducing_points=config.num_inducing_points,
            )
        ],
        axis=-1)
    assert loc_floating_aux_sample.shape == (config.num_samples,
                                             2 * config.num_profiles_floating)

    logging.info("posterior means mu %s",
                 str(posterior_sample_dict['mu'].mean(axis=0)))

    times_data['end_mcmc_stg_1'] = time.perf_counter()

    # Save MCMC samples from stage 1
    np.savez_compressed(samples_path_stg1, posterior_sample_dict)

  ### Sample Second Stage ###

  if os.path.exists(samples_path_stg2):
    logging.info("\t Loading samples for stage 2...")

    posterior_sample_dict = np.load(
        str(samples_path_stg2), allow_pickle=True)['arr_0'].item()
  else:

    if smi_eta is not None:

      logging.info("\t sampling stage 2...")

      # Define parameters split for SMI
      shared_params_names = [
          'gamma_inducing',
          'mixing_weights_list',
          'mixing_offset_list',
          'mu',
          'zeta',
      ]
      refit_params_names = [
          'loc_floating',
      ]
      for key in refit_params_names:
        posterior_sample_dict[key + '_aux'] = posterior_sample_dict[key]
        del posterior_sample_dict[key]

      logging.info("posterior means loc_floating_aux %s",
                   str(posterior_sample_dict['loc_floating_aux'].mean(axis=0)))
      # Define bijector for samples
      bijector_stg2 = get_kernel_bijector_stg2(
          num_profiles_floating=config.num_profiles_floating,
          **config.model_hparams,
      )

      @jax.jit
      def target_log_prob_fn_stg2(global_params, loc_floating_sample):
        return log_prob_fn(
            batch=train_ds,
            prng_key=prng_key_gamma,
            posterior_sample_raw_1d=jnp.concatenate(
                [global_params, loc_floating_sample], axis=-1),
            prior_hparams=config.prior_hparams,
            kernel_name=config.kernel_name,
            kernel_kwargs=config.kernel_kwargs,
            num_samples_gamma_profiles=config.num_samples_gamma_profiles,
            smi_eta_profiles=None,
            num_forms_tuple=config.num_forms_tuple,
            num_basis_gps=config.model_hparams.num_basis_gps,
            num_inducing_points=config.num_inducing_points,
            num_profiles_floating=config.num_profiles_floating,
            include_random_anchor=False,
            gp_jitter=config.gp_jitter,
        )

      # Test the log probability function
      _ = target_log_prob_fn_stg2(
          global_params_sample[0],
          bijector_stg2.forward(5 *
                                jnp.zeros(2 * config.num_profiles_floating)),
      )

      def sample_stg2(
          global_params: Array,
          loc_floating_init: Array,
          num_samples_subchain_stg2: int,
          prng_key: PRNGKey,
      ):

        inner_kernel = tfm.NoUTurnSampler(
            target_log_prob_fn=lambda x: target_log_prob_fn_stg2(
                global_params, x),
            step_size=config.mcmc_step_size,
        )

        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=inner_kernel, bijector=bijector_stg2)

        posterior_sample = tfm.sample_chain(
            num_results=1,
            num_burnin_steps=num_samples_subchain_stg2 - 1,
            kernel=kernel,
            current_state=loc_floating_init,
            trace_fn=None,
            seed=prng_key,
        )

        return posterior_sample

      # # Get one sample of parameters in stage 2
      # loc_floating_sample_ = sample_stg2(
      #     global_params=global_params_sample[0],
      #     loc_floating_init=loc_floating_aux_sample[0],
      #     num_samples_subchain_stg2=5,
      #     prng_key=next(prng_seq),
      # )

      # Define function to parallelize sample_stage2
      # TODO(chris): use pmap?
      sample_stg2_vmap = jax.vmap(
          lambda global_params, loc_floating_init, prng_key: sample_stg2(
              global_params=global_params,
              loc_floating_init=loc_floating_init,
              num_samples_subchain_stg2=config.num_samples_subchain_stg2,
              prng_key=prng_key,
          ))

      def _sample_stage2_loop(chunk_size):
        """Sequential sampling of stage 2.

        Improve performance of nested HMC by splitting the total number of samples
        into several steps, initialising the chains of each step in values
        obtained from the previous step.
        """

        assert (config.num_samples % chunk_size) == 0

        # Initialize loc_floating
        loc_floating = [loc_floating_aux_sample[:chunk_size, :]]

        for i in range(config.num_samples // chunk_size):
          # i=0
          loc_floating_i = sample_stg2_vmap(
              global_params=global_params_sample[(i *
                                                  chunk_size):((i + 1) *
                                                               chunk_size), :],
              loc_floating_init=loc_floating[-1],
              prng_key=jax.random.split(next(prng_seq), chunk_size),
          )
          loc_floating.append(loc_floating_i.squeeze(1))

        loc_floating = jnp.concatenate(loc_floating[1:], axis=0)

        return loc_floating

      # Sample loc_floating
      times_data['start_mcmc_stg_2'] = time.perf_counter()
      loc_floating_sample = _sample_stage2_loop(chunk_size=config.num_samples //
                                                config.num_chunks_stg2)
      posterior_sample_dict_stg2 = split_flow_locations(
          samples=loc_floating_sample,
          num_profiles=config.num_profiles_floating,
          name='loc_floating',
          is_aux=False,
      )
      posterior_sample_dict.update(posterior_sample_dict_stg2)

    logging.info("posterior means loc_floating %s",
                 str(posterior_sample_dict['loc_floating'].mean(axis=0)))

    times_data['end_mcmc_stg_2'] = time.perf_counter()

    # Save MCMC samples from stage 2
    np.savez_compressed(samples_path_stg2, posterior_sample_dict)

  times_data['end_sampling'] = time.perf_counter()

  logging.info("Sampling times:")
  logging.info("\t Total: %s",
               str(times_data['end_sampling'] - times_data['start_sampling']))
  if ('start_mcmc_stg_1' in times_data) and ('end_mcmc_stg_1' in times_data):
    logging.info(
        "\t Stg 1: %s",
        str(times_data['end_mcmc_stg_1'] - times_data['start_mcmc_stg_1']))
  if ('start_mcmc_stg_2' in times_data) and ('end_mcmc_stg_2' in times_data):
    logging.info(
        "\t Stg 2: %s",
        str(times_data['end_mcmc_stg_2'] - times_data['start_mcmc_stg_2']))

  # Save final MCMC samples
  np.savez_compressed(samples_path, posterior_sample_dict)

  # Load samples to compare MCMC vs Variational posteriors
  if config.path_variational_samples != '':
    logging.info("Loading variational samples for comparison...")
    posterior_sample_dict_variational = np.load(
        str(config.path_variational_samples),
        allow_pickle=True)['arr_0'].item()
  else:
    posterior_sample_dict_variational = None

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
      profiles_id=dataset['LP'],
      items_id=dataset['items'],
      forms_id=dataset['forms'],
      show_basis_fields=True,
      show_linguistic_fields=True,
      num_loc_random_anchor_plot=None,
      num_loc_floating_plot=dataset['num_profiles_floating'],
      show_mixing_weights=False,
      show_loc_given_y=False,
      suffix=f"eta_floating_{float(config.eta_profiles_floating):.3f}",
      summary_writer=summary_writer,
      workdir_png=workdir,
      use_gamma_anchor=False,
      posterior_sample_dict_2=posterior_sample_dict_variational,
  )

  # j = 1
  # fig, axs = plt.subplots(2, 1)
  # axs[0].hist(posterior_sample_dict['beta'][:, j], 30)
  # axs[1].plot(posterior_sample_dict['beta'][:, j])

  return posterior_sample_dict


# # For debugging
# config = get_config()
# config.num_samples = 20
# config.num_burnin_steps_stg1 = 5
# config.num_samples_subchain_stg2 = 5
# config.num_chunks_stg2 = 5
# import pathlib
# workdir = str(pathlib.Path.home() / 'spatial-smi-output/8_items/mcmc/eta_floating_0.001')
# sample_and_evaluate(config, workdir)
