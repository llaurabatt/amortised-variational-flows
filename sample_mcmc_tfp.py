"""MCMC sampling for the LALME model."""

import os

import time
import math

from collections import namedtuple

from absl import logging

import numpy as np

import arviz as az

import jax
from jax import numpy as jnp

import haiku as hk
import distrax

from tensorflow_probability.substrates import jax as tfp

from flax.metrics import tensorboard

from modularbayes import flatten_dict
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict,
                                      Mapping, Optional, PRNGKey, Tuple)

# import log_prob_fun
import log_prob_fun_2
from log_prob_fun_2 import ModelParamsGlobal, ModelParamsLocations

import plot
from train_flow import load_data, get_inducing_points
from flows import (split_flow_global_params, split_flow_locations,
                   concat_global_params, concat_locations,
                   get_global_params_shapes)

ModelParamsStg1 = namedtuple("modelparams_stg1", [
    'gamma_inducing',
    'mixing_weights_list',
    'mixing_offset_list',
    'mu',
    'zeta',
    'loc_floating_aux',
])
ModelParamsStg2 = namedtuple("modelparams_stg2", [
    'loc_floating',
])

tfd = tfp.distributions
tfb = tfp.bijectors
tfm = tfp.mcmc
kernels = tfp.math.psd_kernels

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def split_params_stg1(
    params_concat: Array,
    num_forms_tuple: Tuple[int, ...],
    num_basis_gps: int,
    num_inducing_points: int,
    num_profiles_floating: int,
) -> ModelParamsStg1:
  """Get dictionary with parametes to initialize MCMC.

  The order of the parameters
  """

  ### Global parameters ###
  global_dim_ = params_concat.shape[-1] - 2 * num_profiles_floating
  params_global, params_loc_floating_aux = jnp.split(
      params_concat, [global_dim_], axis=-1)

  model_params_global = split_flow_global_params(
      samples=params_global[None, ...],
      num_forms_tuple=num_forms_tuple,
      num_basis_gps=num_basis_gps,
      num_inducing_points=num_inducing_points,
  )
  model_params_global = jax.tree_map(lambda x: x[0], model_params_global)

  ### Location floating profiles ###
  model_params_locations = split_flow_locations(
      samples=params_loc_floating_aux[None, ...],
      num_profiles=num_profiles_floating,
      name='loc_floating_aux',
  )
  model_params_locations = jax.tree_map(lambda x: x[0], model_params_locations)

  model_params_stg1 = ModelParamsStg1(
      gamma_inducing=model_params_global.gamma_inducing,
      mixing_weights_list=model_params_global.mixing_weights_list,
      mixing_offset_list=model_params_global.mixing_offset_list,
      mu=model_params_global.mu,
      zeta=model_params_global.zeta,
      loc_floating_aux=model_params_locations.loc_floating_aux,
  )

  return model_params_stg1


def concat_params_stg1(model_params_stg1: ModelParamsStg1) -> Array:

  params_concat = []

  params_concat.append(
      concat_global_params(
          model_params_global=ModelParamsGlobal(
              gamma_inducing=model_params_stg1.gamma_inducing,
              mixing_weights_list=model_params_stg1.mixing_weights_list,
              mixing_offset_list=model_params_stg1.mixing_offset_list,
              mu=model_params_stg1.mu,
              zeta=model_params_stg1.zeta,
          )))
  params_concat.append(
      concat_locations(
          model_params_locations=ModelParamsLocations(
              loc_floating_aux=model_params_stg1.loc_floating_aux),
          name='loc_floating_aux',
      ))
  params_concat = jnp.concatenate(params_concat, axis=-1)

  return params_concat


def init_param_fn_stg1(
    prng_key: PRNGKey,
    num_forms_tuple: Tuple[int, ...],
    num_basis_gps: int,
    num_inducing_points: int,
    num_profiles_floating: int,
):
  """Get dictionary with parametes to initialize MCMC.

  This function produce unbounded values, i.e. before bijectors to map into the
  domain of the model parameters.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  # Dictionary with shapes of model parameters in stage 1
  samples_shapes = get_global_params_shapes(
      num_forms_tuple=num_forms_tuple,
      num_basis_gps=num_basis_gps,
      num_inducing_points=num_inducing_points,
  )
  samples_shapes['loc_floating_aux'] = (num_profiles_floating, 2)

  # Get a sample for all parameters
  model_params_stg1_ = jax.tree_map(
      lambda shape_i: distrax.Normal(0., 1.).sample(
          seed=next(prng_seq), sample_shape=shape_i),
      tree=samples_shapes,
      is_leaf=lambda x: isinstance(x, Tuple),
  )

  # Define the named tuple to preserve the order of the parameters
  model_params_stg1 = ModelParamsStg1(**model_params_stg1_)

  return model_params_stg1


def get_kernel_bijector_stg1(
    num_forms_tuple: Tuple[int, ...],
    num_profiles_floating: int,
    num_basis_gps: int,
    inducing_grid_shape: Tuple[int, int],
    loc_x_range: Tuple[float, float],
    loc_y_range: Tuple[float, float],
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

  block_bijectors_loc_floating = [loc_x_range_bijector, loc_y_range_bijector
                                 ] * num_profiles_floating
  block_sizes_loc_floating = [1, 1] * num_profiles_floating
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
    loc_x_range: Tuple[float, float],
    loc_y_range: Tuple[float, float],
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

  block_bijectors_loc_floating = [loc_x_range_bijector, loc_y_range_bijector
                                 ] * num_profiles_floating
  block_sizes_loc_floating = [1, 1] * num_profiles_floating
  bijector_loc_floating_layers.append(
      tfb.Blockwise(
          bijectors=block_bijectors_loc_floating,
          block_sizes=block_sizes_loc_floating))
  bijector_loc_floating = tfb.Chain(bijector_loc_floating_layers[::-1])

  return bijector_loc_floating


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
      lambda key_: log_prob_fun_2.sample_gamma_profiles_given_gamma_inducing(
          batch=batch,
          model_params_global=model_params_global,
          model_params_locations=model_params_locations,
          prng_key=key_,
          kernel_name=kernel_name,
          kernel_kwargs=kernel_kwargs,
          gp_jitter=gp_jitter,
          include_random_anchor=False,  # Do not sample gamma for random anchor locations
      ))(
          jax.random.split(prng_key, num_samples_gamma_profiles))

  # Average joint logprob across samples of gamma_profiles
  log_prob = jax.vmap(lambda gamma_profiles_, gamma_profiles_logprob_:
                      log_prob_fun_2.logprob_joint(
                          batch=batch,
                          model_params_global=model_params_global,
                          model_params_locations=model_params_locations,
                          model_params_gamma_profiles=gamma_profiles_,
                          gamma_profiles_logprob=gamma_profiles_logprob_,
                          smi_eta=smi_eta,
                          random_anchor=False,
                          **prior_hparams,
                      ))(model_params_gamma_profiles_sample,
                         gamma_profiles_logprob_sample)
  log_prob = jnp.mean(log_prob)

  return log_prob


def sample_and_evaluate(config: ConfigDict, workdir: str) -> Mapping[str, Any]:
  """Sample and evaluate the random effects model."""

  # Remove trailing slash
  workdir = workdir.rstrip("/")

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)

  # Full dataset used everytime
  # No batching for now
  lalme_dataset = load_data(config=config)
  # Add some parameters to config
  config.num_profiles = lalme_dataset['num_profiles']
  config.num_profiles_anchor = lalme_dataset['num_profiles_anchor']
  config.num_profiles_floating = lalme_dataset['num_profiles_floating']
  config.num_forms_tuple = lalme_dataset['num_forms_tuple']
  config.num_inducing_points = math.prod(
      config.model_hparams.inducing_grid_shape)

  samples_path_stg1 = workdir + '/lalme_stg1_az.nc'
  samples_path = workdir + '/lalme_az.nc'

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

  if os.path.exists(samples_path):
    logging.info("\t Loading final samples")
    lalme_az = az.from_netcdf(samples_path)
  else:
    times_data = {}
    times_data['start_sampling'] = time.perf_counter()

    ### Sample First Stage ###
    if os.path.exists(samples_path_stg1):
      logging.info("\t Loading samples for stage 1...")
      lalme_stg1_az = az.from_netcdf(samples_path_stg1)
    else:
      logging.info("\t Stage 1...")

      # Define target logdensity function
      prng_key_gamma = next(prng_seq)

      @jax.jit
      def logdensity_fn_stg1(model_params_concat):
        model_params_stg1 = split_params_stg1(
            params_concat=model_params_concat,
            num_forms_tuple=config.num_forms_tuple,
            num_basis_gps=config.model_hparams.num_basis_gps,
            num_inducing_points=config.num_inducing_points,
            num_profiles_floating=config.num_profiles_floating,
        )
        model_params_global = ModelParamsGlobal(
            gamma_inducing=model_params_stg1.gamma_inducing,
            mixing_weights_list=model_params_stg1.mixing_weights_list,
            mixing_offset_list=model_params_stg1.mixing_offset_list,
            mu=model_params_stg1.mu,
            zeta=model_params_stg1.zeta,
        )
        model_params_locations = ModelParamsLocations(
            loc_floating=model_params_stg1.loc_floating_aux,)
        logprob_ = logprob_lalme(
            batch=train_ds,
            prng_key=prng_key_gamma,
            model_params_global=model_params_global,
            model_params_locations=model_params_locations,
            prior_hparams=config.prior_hparams,
            kernel_name=config.kernel_name,
            kernel_kwargs=config.kernel_kwargs,
            num_samples_gamma_profiles=config.num_samples_gamma_profiles,
            smi_eta_profiles=smi_eta['profiles'],
            gp_jitter=config.gp_jitter,
        )
        return logprob_

      # initial positions of model parameters
      model_params_stg1_unb_init = init_param_fn_stg1(
          prng_key=next(prng_seq),
          num_forms_tuple=config.num_forms_tuple,
          num_basis_gps=config.model_hparams.num_basis_gps,
          num_inducing_points=config.num_inducing_points,
          num_profiles_floating=config.num_profiles_floating,
      )

      posterior_sample_raw_init = concat_params_stg1(model_params_stg1_unb_init)

      # Verify that split_samples is the inverse of concat_samples
      assert all(
          v for v in jax.tree_map(
              lambda x, y: (x == y).all(),
              model_params_stg1_unb_init,
              split_params_stg1(
                  params_concat=posterior_sample_raw_init,
                  num_forms_tuple=config.num_forms_tuple,
                  num_basis_gps=config.model_hparams.num_basis_gps,
                  num_inducing_points=config.num_inducing_points,
                  num_profiles_floating=config.num_profiles_floating,
              ),
          )._asdict().values())

      # Define bijector for samples
      bijector_stg1 = get_kernel_bijector_stg1(
          num_forms_tuple=config.num_forms_tuple,
          num_profiles_floating=config.num_profiles_floating,
          **config.model_hparams,
      )

      # Test the log probability function
      _ = logdensity_fn_stg1(
          model_params_concat=bijector_stg1.forward(posterior_sample_raw_init))

      # Define sampling kernel
      kernel = tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=tfm.NoUTurnSampler(
              target_log_prob_fn=logdensity_fn_stg1,
              step_size=config.mcmc_step_size,
          ),
          bijector=bijector_stg1)

      # Sample stage 1 chain
      times_data['start_mcmc_stg_1'] = time.perf_counter()
      model_params_stg1_samples_concat = tfm.sample_chain(
          num_results=config.num_samples,
          num_burnin_steps=config.num_burnin_steps_stg1,
          kernel=kernel,
          current_state=bijector_stg1.forward(posterior_sample_raw_init),
          trace_fn=None,
          seed=next(prng_seq),
      )

      # Save samples from stage 1
      model_params_stg1_samples = jax.vmap(lambda x: split_params_stg1(
          params_concat=x,
          num_forms_tuple=config.num_forms_tuple,
          num_basis_gps=config.model_hparams.num_basis_gps,
          num_inducing_points=config.num_inducing_points,
          num_profiles_floating=config.num_profiles_floating,
      ))(
          model_params_stg1_samples_concat)

      # Verify that concat_samples is the inverse of split_samples
      assert (model_params_stg1_samples_concat == jax.vmap(
          lambda x: concat_params_stg1(model_params_stg1=x))(
              model_params_stg1_samples)).all()

      model_params_global_samples = ModelParamsGlobal(
          gamma_inducing=model_params_stg1_samples.gamma_inducing,
          mixing_weights_list=model_params_stg1_samples.mixing_weights_list,
          mixing_offset_list=model_params_stg1_samples.mixing_offset_list,
          mu=model_params_stg1_samples.mu,
          zeta=model_params_stg1_samples.zeta,
      )

      # Create InferenceData object
      lalme_stg1_az = plot.lalme_az_from_samples(
          lalme_dataset=lalme_dataset,
          model_params_global=jax.tree_map(lambda x: x[None, ...],
                                           model_params_global_samples),
          model_params_locations=ModelParamsLocations(
              loc_floating=None,
              loc_floating_aux=model_params_stg1_samples.loc_floating_aux[None,
                                                                          ...],
              loc_random_anchor=None,
          ),
          model_params_gamma=None,
      )
      # Save InferenceData object from stage 1
      lalme_stg1_az.to_netcdf(samples_path_stg1)

      logging.info("\t\t posterior means mu:  %s",
                   str(jnp.array(lalme_stg1_az.posterior.mu).mean(axis=[0, 1])))

      times_data['end_mcmc_stg_1'] = time.perf_counter()

    ### Sample Second Stage ###
    logging.info("\t Stage 2...")

    # Extract global parameters from stage 1 samples
    model_params_global_samples = ModelParamsGlobal(
        gamma_inducing=jnp.array(lalme_stg1_az.posterior.gamma_inducing),
        mixing_weights_list=[
            jnp.array(lalme_stg1_az.posterior[f'W_{i}'])
            for i in range(len(config.num_forms_tuple))
        ],
        mixing_offset_list=[
            jnp.array(lalme_stg1_az.posterior[f'a_{i}'])
            for i in range(len(config.num_forms_tuple))
        ],
        mu=jnp.array(lalme_stg1_az.posterior.mu),
        zeta=jnp.array(lalme_stg1_az.posterior.zeta),
    )
    model_params_global_samples = jax.tree_map(lambda x: x[0],
                                               model_params_global_samples)

    # Define bijector for samples
    bijector_stg2 = get_kernel_bijector_stg2(
        num_profiles_floating=config.num_profiles_floating,
        **config.model_hparams,
    )

    # Define target logdensity function
    prng_key_gamma = next(prng_seq)

    @jax.jit
    def logdensity_fn_stg2(model_params_concat, conditioner):
      model_params_locations = split_flow_locations(
          samples=model_params_concat[None, ...],
          num_profiles=config.num_profiles_floating,
          name='loc_floating',
      )
      model_params_locations = jax.tree_map(lambda x: x[0],
                                            model_params_locations)
      logprob_ = logprob_lalme(
          batch=train_ds,
          prng_key=prng_key_gamma,
          model_params_global=conditioner,
          model_params_locations=model_params_locations,
          prior_hparams=config.prior_hparams,
          kernel_name=config.kernel_name,
          kernel_kwargs=config.kernel_kwargs,
          num_samples_gamma_profiles=config.num_samples_gamma_profiles,
          smi_eta_profiles=smi_eta['profiles'],
          gp_jitter=config.gp_jitter,
      )
      return logprob_

    # Test the log probability function
    _ = logdensity_fn_stg2(
        model_params_concat=bijector_stg2.forward(
            5 * jnp.zeros(2 * config.num_profiles_floating)),
        conditioner=jax.tree_map(lambda x: x[0], model_params_global_samples))

    def sample_stg2(
        loc_floating_init: Array,
        model_params_global: ModelParamsGlobal,
        num_samples_subchain_stg2: int,
        prng_key: PRNGKey,
    ):

      inner_kernel = tfm.NoUTurnSampler(
          target_log_prob_fn=lambda x: logdensity_fn_stg2(
              x, conditioner=model_params_global),
          step_size=config.mcmc_step_size,
      )

      kernel = tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=inner_kernel, bijector=bijector_stg2)

      loc_floating_concat_init = concat_locations(
          model_params_locations=loc_floating_init,
          name='loc_floating',
      )
      posterior_sample = tfm.sample_chain(
          num_results=1,
          num_burnin_steps=num_samples_subchain_stg2 - 1,
          kernel=kernel,
          current_state=loc_floating_concat_init,
          trace_fn=None,
          seed=prng_key,
      ).squeeze(0)

      return posterior_sample

    model_params_locations_init = ModelParamsLocations(
        loc_floating=jax.tree_map(
            lambda x: x[:config.num_samples_perchunk_stg2, ...],
            jnp.array(lalme_stg1_az.posterior.loc_floating_aux[0])))

    # # Get one sample of parameters in stage 2
    # loc_floating_sample_ = sample_stg2(
    #     loc_floating_init=jax.tree_map(lambda x: x[0],
    #                                    model_params_locations_init),
    #     model_params_global=jax.tree_map(lambda x: x[0],
    #                                      model_params_global_samples),
    #     num_samples_subchain_stg2=5,
    #     prng_key=next(prng_seq),
    # )

    # Define function to parallelize sample_stage2
    sample_stg2_vmap = jax.vmap(
        lambda loc_floating_concat, global_, key_: sample_stg2(
            loc_floating_init=loc_floating_concat,
            model_params_global=global_,
            num_samples_subchain_stg2=config.num_samples_subchain_stg2,
            prng_key=key_,
        ))

    # The number of samples is large and often it does not fit into GPU memory
    # we split the sampling of stage 2 into chunks
    assert config.num_samples % config.num_samples_perchunk_stg2 == 0
    num_chunks_stg2 = config.num_samples // config.num_samples_perchunk_stg2

    # Initialize loc_floating
    initial_position_i = model_params_locations_init
    chunks_positions = []
    for i in range(num_chunks_stg2):
      cond_i = jax.tree_map(
          lambda x: x[(i * config.num_samples_perchunk_stg2):
                      ((i + 1) * config.num_samples_perchunk_stg2), ...],
          model_params_global_samples)

      model_params_concat_i = sample_stg2_vmap(
          initial_position_i,
          cond_i,
          jax.random.split(next(prng_seq), config.num_samples_perchunk_stg2),
      )

      model_params_i = split_flow_locations(
          samples=model_params_concat_i,
          num_profiles=config.num_profiles_floating,
          name='loc_floating',
      )
      chunks_positions.append(model_params_i)
      initial_position_i = model_params_i

    model_params_stg2_samples = jax.tree_map(  # pylint: disable=no-value-for-parameter
        lambda *x: jnp.concatenate(x, axis=0), *chunks_positions)

    times_data['end_mcmc_stg_2'] = time.perf_counter()

    lalme_az = plot.lalme_az_from_samples(
        lalme_dataset=lalme_dataset,
        model_params_global=jax.tree_map(lambda x: x[None, ...],
                                         model_params_global_samples),
        model_params_locations=ModelParamsLocations(
            loc_floating=model_params_stg2_samples.loc_floating[None, ...],
            loc_floating_aux=jnp.array(
                lalme_stg1_az.posterior.loc_floating_aux),
            loc_random_anchor=None,
        ),
        model_params_gamma=None,
    )

    logging.info(
        "\t\t posterior means loc_floating (before transform): %s",
        str(jnp.array(lalme_az.posterior.loc_floating).mean(axis=[0, 1])))

    # Save InferenceData object
    lalme_az.to_netcdf(samples_path)

    logging.info(
        "\t\t posterior means loc_floating: %s",
        str(jnp.array(lalme_az.posterior.loc_floating).mean(axis=[0, 1])))

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

  # Get a sample of the basis GPs on profiles locations
  # conditional on values at the inducing locations.
  model_params_global_samples_ = ModelParamsGlobal(
      gamma_inducing=jnp.array(lalme_az.posterior.gamma_inducing),
      mixing_weights_list=[
          jnp.array(lalme_az.posterior[f'W_{i}'])
          for i in range(len(config.num_forms_tuple))
      ],
      mixing_offset_list=[
          jnp.array(lalme_az.posterior[f'a_{i}'])
          for i in range(len(config.num_forms_tuple))
      ],
      mu=jnp.array(lalme_az.posterior.mu),
      zeta=jnp.array(lalme_az.posterior.zeta),
  )
  model_params_locations_samples_ = ModelParamsLocations(
      loc_floating=jnp.array(lalme_az.posterior.loc_floating),
      loc_floating_aux=jnp.array(lalme_az.posterior.loc_floating_aux),
      loc_random_anchor=None,
  )
  model_params_gamma_samples_, _ = jax.vmap(
      jax.vmap(lambda key_, global_, locations_: log_prob_fun_2.
               sample_gamma_profiles_given_gamma_inducing(
                   batch=train_ds,
                   model_params_global=global_,
                   model_params_locations=locations_,
                   prng_key=key_,
                   kernel_name=config.kernel_name,
                   kernel_kwargs=config.kernel_kwargs,
                   gp_jitter=config.gp_jitter,
                   include_random_anchor=False,
               )))(
                   jax.random.split(next(prng_seq),
                                    1 * config.num_samples).reshape(
                                        (1, config.num_samples, 2)),
                   model_params_global_samples_,
                   model_params_locations_samples_,
               )
  lalme_az_with_gamma = plot.lalme_az_from_samples(
      lalme_dataset=lalme_dataset,
      model_params_global=model_params_global_samples_,
      model_params_locations=model_params_locations_samples_,
      model_params_gamma=model_params_gamma_samples_,
  )

  ### Posterior visualisation with Arviz

  logging.info("Plotting results...")

  plot.lalme_plots_arviz(
      lalme_az=lalme_az_with_gamma,
      lalme_dataset=lalme_dataset,
      step=0,
      show_mu=True,
      show_zeta=True,
      show_basis_fields=True,
      show_W_items=lalme_dataset['items'],
      show_a_items=lalme_dataset['items'],
      lp_floating=lalme_dataset['LP'][lalme_dataset['num_profiles_anchor']:],
      lp_floating_traces=config.lp_floating_grid10,
      lp_floating_grid10=config.lp_floating_grid10,
      loc_inducing=train_ds['loc_inducing'],
      workdir_png=workdir,
      summary_writer=summary_writer,
      suffix=f"_eta_floating_{float(config.eta_profiles_floating):.3f}",
      scatter_kwargs={"alpha": 0.05},
  )
  logging.info("...done!")

  # Load samples to compare MCMC vs Variational posteriors
  if (config.path_variational_samples != '') and (os.path.exists(
      config.path_variational_samples)):
    logging.info("Plotting comparison MCMC and Variational...")
    lalme_az_variational = az.from_netcdf(config.path_variational_samples)

    plot.posterior_samples_compare(
        lalme_az_1=lalme_az_with_gamma,
        lalme_az_2=lalme_az_variational,
        lalme_dataset=lalme_dataset,
        step=0,
        lp_floating_grid10=config.lp_floating_grid10,
        summary_writer=summary_writer,
        workdir_png=workdir,
        suffix=f"_eta_floating_{float(config.eta_profiles_floating):.3f}",
        scatter_kwargs={"alpha": 0.05},
    )
    logging.info("...done!")


# # For debugging
# config = get_config()
# eta = 1.000
# import pathlib
# workdir = str(pathlib.Path.home() / f'spatial-smi-output-exp/8_items/mcmc/eta_floating_{eta:.3f}')
# config.path_variational_samples = str(pathlib.Path.home() / f'spatial-smi-output/8_items/nsf/vmp_flow/lalme_az_eta_{eta:.3f}.nc')
# config.eta_profiles_floating = eta
# # config.num_samples = 100
# # config.num_burnin_steps_stg1 = 5
# # config.num_samples_subchain_stg2 = 5
# # config.num_samples_perchunk_stg2 = 10
# # config.mcmc_step_size = 0.001
# # sample_and_evaluate(config, workdir)
