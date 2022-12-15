"""MCMC sampling for the LALME model."""

import os

import time
import math

from collections import namedtuple

from absl import logging

import numpy as np

import jax
from jax import numpy as jnp

import haiku as hk
import distrax

import blackjax
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.nuts import NUTSInfo
from blackjax.types import PyTree

from tensorflow_probability.substrates import jax as tfp

from flax.metrics import tensorboard

from modularbayes import flatten_dict
from modularbayes._src.typing import (Any, Array, Batch, Callable, ConfigDict,
                                      Dict, Mapping, Optional, PRNGKey, Tuple)

import plot
from train_flow import load_data, get_inducing_points
from flows import get_global_params_shapes
import log_prob_fun_2
from log_prob_fun_2 import ModelParamsGlobal, ModelParamsLocations

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

tfb = tfp.bijectors
kernels = tfp.math.psd_kernels

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def call_warmup(
    prng_key: PRNGKey,
    logprob_fn: Callable,
    model_params: PyTree,
    num_steps: int,
) -> Tuple:
  warmup = blackjax.window_adaptation(
      algorithm=blackjax.nuts,
      logprob_fn=logprob_fn,
  )
  initial_states, _, hmc_params = warmup.run(
      rng_key=prng_key,
      position=model_params,
      num_steps=num_steps,
  )
  return initial_states, hmc_params


def inference_loop_one_chain(
    prng_key: PRNGKey,
    initial_state: HMCState,
    hmc_params: Dict[str, Array],
    logprob_fn: Callable,
    num_samples: int,
) -> Tuple[HMCState, NUTSInfo]:

  def one_step(state, rng_key):
    kernel_fn = lambda state_, key_, hmc_param_: blackjax.nuts.kernel()(
        rng_key=key_,
        state=state_,
        logprob_fn=logprob_fn,
        step_size=hmc_param_['step_size'],
        inverse_mass_matrix=hmc_param_['inverse_mass_matrix'],
    )
    state_new, info_new = kernel_fn(state, rng_key, hmc_params)
    return state_new, (state_new, info_new)

  keys = jax.random.split(prng_key, num_samples)
  _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

  return states, infos


def inference_loop_stg1(
    prng_key: PRNGKey,
    initial_states: HMCState,
    hmc_params: Dict[str, Array],
    logprob_fn: Callable,
    num_samples: int,
    num_chains: int,
) -> Tuple[HMCState, NUTSInfo]:

  def one_step(states, rng_keys):
    kernel_fn_multichain = jax.vmap(
        lambda state_, hmc_param_, key_nuts_, key_gamma_: blackjax.nuts.kernel(
        )(
            rng_key=key_nuts_,
            state=state_,
            logprob_fn=lambda x: logprob_fn(x, key_gamma_),
            step_size=hmc_param_['step_size'],
            inverse_mass_matrix=hmc_param_['inverse_mass_matrix'],
        ))
    keys_nuts_, key_gamma_ = jnp.split(rng_keys, 2, axis=-2)
    states_new, infos_new = kernel_fn_multichain(
        states,
        hmc_params,
        keys_nuts_.squeeze(-2),
        key_gamma_.squeeze(-2),
    )
    return states_new, (states_new, infos_new)

  keys = jax.random.split(prng_key, num_samples * num_chains * 2).reshape(
      (num_samples, num_chains, 2, 2))

  # one_step(initial_states, keys[0])

  _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

  return states, infos


def inference_loop_stg2(
    prng_key: PRNGKey,
    initial_states: HMCState,
    hmc_params: Dict[str, Array],
    logprob_fn_conditional: Callable,
    conditioner_logprob: ModelParamsGlobal,
    num_samples_stg1: int,
    num_samples_stg2: int,
    num_chains: int,
):

  # We only need to keep the last sample of the subchains
  def one_step(states, rng_keys):
    kernel_fn_multichain = jax.vmap(
        lambda state, cond, hmc_param, key_nuts_, key_gamma_: blackjax.nuts.
        kernel()(
            rng_key=key_nuts_,
            state=state,
            logprob_fn=lambda param_: logprob_fn_conditional(
                model_params=param_,
                conditioner=cond,
                prng_key_gamma=key_gamma_,
            ),
            step_size=hmc_param['step_size'],
            inverse_mass_matrix=hmc_param['inverse_mass_matrix'],
        ))
    kernel_fn_multicond_multichain = jax.vmap(
        lambda states_, conds_, keys_nuts_, keys_gamma_: kernel_fn_multichain(
            states_,
            conds_,
            hmc_params,
            keys_nuts_,
            keys_gamma_,
        ))
    keys_nuts_, key_gamma_ = jnp.split(rng_keys, 2, axis=-2)

    states_new, _ = kernel_fn_multicond_multichain(
        states,
        conditioner_logprob,
        keys_nuts_.squeeze(-2),
        key_gamma_.squeeze(-2),
    )
    return states_new, None

  keys = jax.random.split(
      prng_key, num_samples_stg2 * num_samples_stg1 * num_chains * 2).reshape(
          (num_samples_stg2, num_samples_stg1, num_chains, 2, 2))

  # one_step(initial_states, keys[0])

  last_state, _ = jax.lax.scan(one_step, initial_states, keys)

  return last_state


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
  samples_ = jax.tree_map(
      lambda shape_i: distrax.Normal(0., 1.).sample(
          seed=next(prng_seq), sample_shape=shape_i),
      tree=samples_shapes,
      is_leaf=lambda x: isinstance(x, Tuple),
  )

  # Define the named tuple
  model_params_stg1 = ModelParamsStg1(**samples_)

  return model_params_stg1


def transform_model_params(
    model_params_global_unb: ModelParamsGlobal,
    model_params_locations_unb: ModelParamsLocations,
) -> Tuple[ModelParamsGlobal, ModelParamsLocations, Array]:
  """Apply transformations to map into model parameters domain."""

  bijectors_ = {
      'mu': distrax.Block(tfb.Softplus(), 1),
      'zeta': distrax.Block(distrax.Sigmoid(), 1),
      'loc_floating': distrax.Block(distrax.Sigmoid(), 2),
      'loc_floating_aux': distrax.Block(distrax.Sigmoid(), 2),
      'loc_random_anchor': distrax.Block(distrax.Sigmoid(), 2),
  }

  model_params_global = ModelParamsGlobal(
      gamma_inducing=model_params_global_unb.gamma_inducing,
      mixing_weights_list=model_params_global_unb.mixing_weights_list,
      mixing_offset_list=model_params_global_unb.mixing_offset_list,
      mu=bijectors_['mu'].forward(model_params_global_unb.mu),
      zeta=bijectors_['zeta'].forward(model_params_global_unb.zeta),
  )

  model_params_locations = ModelParamsLocations(
      loc_floating=(None if model_params_locations_unb.loc_floating is None else
                    bijectors_['loc_floating'].forward(
                        model_params_locations_unb.loc_floating)),
      loc_floating_aux=(None
                        if model_params_locations_unb.loc_floating_aux is None
                        else bijectors_['loc_floating_aux'].forward(
                            model_params_locations_unb.loc_floating_aux)),
      loc_random_anchor=(None
                         if model_params_locations_unb.loc_random_anchor is None
                         else bijectors_['loc_random_anchor'].forward(
                             model_params_locations_unb.loc_random_anchor)),
  )

  # Adjust log probability for the transformations.
  log_det_jacob_transformed = 0
  log_det_jacob_transformed += bijectors_['mu'].forward_log_det_jacobian(
      model_params_global_unb.mu)
  log_det_jacob_transformed += bijectors_['zeta'].forward_log_det_jacobian(
      model_params_global_unb.zeta)
  if model_params_locations_unb.loc_floating is not None:
    log_det_jacob_transformed += bijectors_[
        'loc_floating'].forward_log_det_jacobian(
            model_params_locations_unb.loc_floating)
  if model_params_locations_unb.loc_floating_aux is not None:
    log_det_jacob_transformed += bijectors_[
        'loc_floating_aux'].forward_log_det_jacobian(
            model_params_locations_unb.loc_floating_aux)
  if model_params_locations_unb.loc_random_anchor is not None:
    log_det_jacob_transformed += bijectors_[
        'loc_random_anchor'].forward_log_det_jacobian(
            model_params_locations_unb.loc_random_anchor)

  return model_params_global, model_params_locations, log_det_jacob_transformed


def log_prob_lalme(
    batch: Batch,
    prng_key: PRNGKey,
    model_params_global_unb: ModelParamsGlobal,
    model_params_locations_unb: ModelParamsLocations,
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

  (model_params_global, model_params_locations,
   log_det_jacob_transformed) = transform_model_params(
       model_params_global_unb=model_params_global_unb,
       model_params_locations_unb=model_params_locations_unb,
   )

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
          is_smi=False,  # Do not sample the auxiliary gamma
          include_random_anchor=False,  # Do not sample gamma for random anchor locations
      ))(
          jax.random.split(prng_key, num_samples_gamma_profiles))

  # Average joint logprob across samples of gamma_profiles
  log_prob = jax.vmap(lambda gamma_profiles_, gamma_profiles_logprob_:
                      log_prob_fun_2.log_prob_joint(
                          batch=batch,
                          model_params_global=model_params_global,
                          model_params_locations=model_params_locations,
                          model_params_gamma_profiles=gamma_profiles_,
                          gamma_profiles_logprob=gamma_profiles_logprob_,
                          smi_eta=smi_eta,
                          **prior_hparams,
                      ))(model_params_gamma_profiles_sample,
                         gamma_profiles_logprob_sample)
  log_prob = jnp.mean(log_prob)

  # globals().update(dict(prior_hparams))
  # model_params_gamma_profiles = jax.tree_map(lambda x: x[0], model_params_gamma_profiles_sample)
  # gamma_profiles_logprob = jax.tree_map(lambda x: x[0], gamma_profiles_logprob_sample)

  return log_prob + log_det_jacob_transformed


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

  samples_path_stg1 = workdir + '/model_params_stg1_unb_samples.npz'
  samples_path_stg2 = workdir + '/model_params_stg2_unb_samples.npz'
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

  times_data = {}
  times_data['start_sampling'] = time.perf_counter()

  ### Sample First Stage ###
  if os.path.exists(samples_path_stg1):
    logging.info("\t Loading samples for stage 1...")
    aux_ = np.load(str(samples_path_stg1), allow_pickle=True)['arr_0']
    model_params_stg1_unb_samples = ModelParamsStg1(*aux_)
    # model_params_stg1_unb_samples = jax.tree_map(lambda x: x.swapaxes(0, 1),
    #                                              model_params_stg1_unb_samples)
  else:
    logging.info("\t Stage 1...")

    # Define target logprob function
    @jax.jit
    def logprob_fn_stg1(model_params, prng_key_gamma):
      model_params_global_unb = ModelParamsGlobal(
          gamma_inducing=model_params.gamma_inducing,
          mixing_weights_list=model_params.mixing_weights_list,
          mixing_offset_list=model_params.mixing_offset_list,
          mu=model_params.mu,
          zeta=model_params.zeta,
      )
      model_params_locations_unb = ModelParamsLocations(
          loc_floating=model_params.loc_floating_aux,
          loc_floating_aux=None,
          loc_random_anchor=None,
      )

      log_prob = log_prob_lalme(
          batch=train_ds,
          prng_key=prng_key_gamma,
          model_params_global_unb=model_params_global_unb,
          model_params_locations_unb=model_params_locations_unb,
          prior_hparams=config.prior_hparams,
          kernel_name=config.kernel_name,
          kernel_kwargs=config.kernel_kwargs,
          num_samples_gamma_profiles=config.num_samples_gamma_profiles,
          smi_eta_profiles=smi_eta['profiles'] if smi_eta is not None else None,
          gp_jitter=config.gp_jitter,
      )
      return log_prob

    # initial positions of model parameters
    # (vmap to produce one for each MCMC chains)
    model_params_stg1_unb_init = jax.vmap(lambda prng_key: init_param_fn_stg1(
        prng_key=prng_key,
        num_forms_tuple=config.num_forms_tuple,
        num_basis_gps=config.model_hparams.num_basis_gps,
        num_inducing_points=config.num_inducing_points,
        num_profiles_floating=config.num_profiles_floating,
    ))(
        jax.random.split(next(prng_seq), config.num_chains))

    # Tune HMC parameters automatically
    logging.info('\t tuning HMC parameters stg1...')
    key_gamma_ = next(prng_seq)
    initial_states_stg1, hmc_params_stg1 = jax.vmap(
        lambda prng_key, model_params: call_warmup(
            prng_key=prng_key,
            logprob_fn=lambda x: logprob_fn_stg1(x, key_gamma_),
            model_params=model_params,
            num_steps=config.num_steps_call_warmup,
        ))(
            jax.random.split(next(prng_seq), config.num_chains),
            model_params_stg1_unb_init,
        )

    # Sampling loop stage 1
    logging.info('\t sampling stage 1...')
    states_stg1, _ = inference_loop_stg1(
        prng_key=next(prng_seq),
        initial_states=initial_states_stg1,
        hmc_params=hmc_params_stg1,
        logprob_fn=logprob_fn_stg1,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
    )

    # Save samples from stage 1
    # swap position axes to have shape (num_chains, num_samples, ...)
    model_params_stg1_unb_samples = jax.tree_map(lambda x: x.swapaxes(0, 1),
                                                 states_stg1.position)

    # Save MCMC samples from stage 1
    np.savez_compressed(samples_path_stg1, model_params_stg1_unb_samples)

    logging.info("\t\t posterior means mu (before transform) %s",
                 str(model_params_stg1_unb_samples.mu.mean(axis=[0, 1])))

    times_data['end_mcmc_stg_1'] = time.perf_counter()

  model_params_global_unb_samples = ModelParamsGlobal(
      gamma_inducing=model_params_stg1_unb_samples.gamma_inducing,
      mixing_weights_list=model_params_stg1_unb_samples.mixing_weights_list,
      mixing_offset_list=model_params_stg1_unb_samples.mixing_offset_list,
      mu=model_params_stg1_unb_samples.mu,
      zeta=model_params_stg1_unb_samples.zeta,
  )

  ### Sample Second Stage ###

  if os.path.exists(samples_path_stg2):
    logging.info("\t Loading samples for stage 2...")
    aux_ = np.load(str(samples_path_stg2), allow_pickle=True)['arr_0']
    model_params_stg2_unb_samples = ModelParamsLocations(*aux_)

  else:

    logging.info("\t Stage 2...")

    # Define target logprob function

    @jax.jit
    def logprob_fn_stg2(model_params, conditioner, prng_key_gamma):
      log_prob = log_prob_lalme(
          batch=train_ds,
          prng_key=prng_key_gamma,
          model_params_global_unb=conditioner,
          model_params_locations_unb=model_params,
          prior_hparams=config.prior_hparams,
          kernel_name=config.kernel_name,
          kernel_kwargs=config.kernel_kwargs,
          num_samples_gamma_profiles=config.num_samples_gamma_profiles,
          smi_eta_profiles=None,
          gp_jitter=config.gp_jitter,
      )
      return log_prob

    # Tune HMC parameters automatically
    logging.info('\t tuning HMC parameters stg2...')

    # We tune the HMC for one sample in stage 1
    # tune HMC parameters, vmap across chains
    key_gamma_ = next(prng_seq)
    _, hmc_params_stg2 = jax.vmap(lambda key, param, cond: call_warmup(
        prng_key=key,
        logprob_fn=lambda param_: logprob_fn_stg2(
            conditioner=cond,
            model_params=param_,
            prng_key_gamma=key_gamma_,
        ),
        model_params=param,
        num_steps=config.num_steps_call_warmup,
    ))(
        jax.random.split(next(prng_seq), config.num_chains),
        ModelParamsLocations(
            loc_floating=model_params_stg1_unb_samples.loc_floating_aux[:, 0,
                                                                        ...],
            loc_floating_aux=None,
            loc_random_anchor=None,
        ),
        jax.tree_map(lambda x: x[:, 0, ...], model_params_global_unb_samples),
    )

    # When the number of samples is large,
    # we split the sampling of stage 2 into chunks
    assert config.num_samples % config.num_samples_perchunk_stg2 == 0
    num_chunks_stg2 = config.num_samples // config.num_samples_perchunk_stg2

    # Initialize stage 1 using loc_floating_aux
    # we use the tuned HMC parameters from above
    # Note: vmap is first applied to the chains, then to samples from conditioner
    #    this requires swap axes 0 and 1 in a few places
    init_fn_multichain = jax.vmap(lambda param, cond, hmc_param: blackjax.nuts(
        logprob_fn=lambda param_: logprob_fn_stg2(
            conditioner=cond,
            model_params=param_,
            prng_key_gamma=key_gamma_,
        ),
        step_size=hmc_param['step_size'],
        inverse_mass_matrix=hmc_param['inverse_mass_matrix'],
    ).init(position=param))
    init_fn_multicond_multichain = jax.vmap(
        lambda param_, cond_: init_fn_multichain(
            param=param_,
            cond=cond_,
            hmc_param=hmc_params_stg2,
        ))

    # The initial position for loc_floating in the first chunk is the location
    # of loc_floating_aux from stage 1
    initial_position_i = ModelParamsLocations(
        loc_floating=model_params_stg1_unb_samples
        .loc_floating_aux[:, :config.num_samples_perchunk_stg2, ...].swapaxes(
            0, 1),  # swap axes to have (num_samples, num_chains, ...)
        loc_floating_aux=None,
        loc_random_anchor=None,
    )

    logging.info('\t sampling stage 2...')
    chunks_positions = []
    for i in range(num_chunks_stg2):
      cond_i = jax.tree_map(
          lambda x: x[:, (i * config.num_samples_perchunk_stg2):(
              (i + 1) * config.num_samples_perchunk_stg2), ...].swapaxes(0, 1),
          model_params_global_unb_samples)

      initial_state_i = init_fn_multicond_multichain(initial_position_i, cond_i)

      # Sampling loop stage 2
      states_stg2_i = inference_loop_stg2(
          prng_key=next(prng_seq),
          initial_states=initial_state_i,
          hmc_params=hmc_params_stg2,
          logprob_fn_conditional=logprob_fn_stg2,
          conditioner_logprob=cond_i,
          num_samples_stg1=config.num_samples_perchunk_stg2,
          num_samples_stg2=config.num_samples_subchain_stg2,
          num_chains=config.num_chains,
      )

      chunks_positions.append(states_stg2_i.position)

      # Subsequent chunks initialise in last position of the previous chunk
      initial_position_i = states_stg2_i.position

    # Concatenate samples from each chunk, across samples dimension
    model_params_stg2_unb_samples = jax.tree_map(  # pylint: disable=no-value-for-parameter
        lambda *x: jnp.concatenate(x, axis=0), *chunks_positions)
    # swap axes to have shape (num_chains, num_samples, ...)
    model_params_stg2_unb_samples = jax.tree_map(lambda x: x.swapaxes(0, 1),
                                                 model_params_stg2_unb_samples)

    logging.info(
        "\t\t posterior means loc_floating (before transform) %s",
        str(model_params_stg2_unb_samples.loc_floating.mean(axis=[0, 1])))

    times_data['end_mcmc_stg_2'] = time.perf_counter()

    # Save MCMC samples from stage 1
    np.savez_compressed(samples_path_stg2, model_params_stg2_unb_samples)

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

  # Transform unbounded parameters to model parameters
  (model_params_global_samples, model_params_locations_samples,
   _) = transform_model_params(
       model_params_global_unb=model_params_global_unb_samples,
       model_params_locations_unb=ModelParamsLocations(
           loc_floating=model_params_stg2_unb_samples.loc_floating,
           loc_floating_aux=model_params_stg1_unb_samples.loc_floating_aux,
           loc_random_anchor=None,
       ),
   )

  # Get a sample of the basis GPs on profiles locations
  # conditional on values at the inducing locations.
  model_params_gamma_samples, _ = jax.vmap(
      jax.vmap(
          lambda key_, global_, locations_: log_prob_fun_2.
          sample_gamma_profiles_given_gamma_inducing(
              batch=train_ds,
              model_params_global=global_,
              model_params_locations=locations_,
              prng_key=key_,
              kernel_name=config.kernel_name,
              kernel_kwargs=config.kernel_kwargs,
              gp_jitter=config.gp_jitter,
              is_smi=False,  # Do not sample on loc_floating_aux
              include_random_anchor=False,  # Do not sample gamma for random anchor locations
          )))(
              jax.random.split(
                  next(prng_seq),
                  config.num_chains * config.num_samples).reshape(
                      (config.num_chains, config.num_samples, 2)),
              model_params_global_samples,
              model_params_locations_samples,
          )

  ### Posterior visualisation with Arviz

  logging.info("Plotting results...")

  # Create InferenceData object
  lalme_az = plot.lalme_az_from_samples(
      model_params_global=model_params_global_samples,
      model_params_locations=model_params_locations_samples,
      model_params_gamma=model_params_gamma_samples,
      lalme_dataset=dataset,
  )

  plot.lalme_plots_arviz(
      lalme_az=lalme_az,
      lalme_dataset=dataset,
      step=0,
      show_mu=True,
      show_zeta=True,
      show_basis_fields=True,
      show_W_items=dataset['items'],
      show_a_items=dataset['items'],
      lp_floating=dataset['LP'][dataset['num_profiles_anchor']:],
      lp_floating_traces=config.lp_floating_10,
      lp_floating_grid10=config.lp_floating_10,
      loc_inducing=train_ds['loc_inducing'],
      workdir_png=workdir,
      summary_writer=summary_writer,
      suffix=f"_eta_floating_{float(config.eta_profiles_floating):.3f}",
      scatter_kwargs={"alpha": 0.05},
  )


# # For debugging
# config = get_config()
# eta = 1.000
# import pathlib
# workdir = str(pathlib.Path.home() / f'spatial-smi-output-exp/8_items/mcmc/eta_floating_{eta:.3f}')
# config.path_variational_samples = str(pathlib.Path.home() / f'spatial-smi-output-exp/8_items/nsf/eta_floating_{eta:.3f}/posterior_sample_dict.npz')
# # config.num_samples = 100
# # config.num_samples_subchain_stg2 = 10
# # config.num_samples_perchunk_stg2 = 50
# # config.num_steps_call_warmup = 10
# # # sample_and_evaluate(config, workdir)
