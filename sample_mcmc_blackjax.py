"""MCMC sampling for the LALME model."""

import os

import time
import math

from collections import namedtuple

from absl import logging

import numpy as np

from matplotlib import pyplot as plt

import jax
from jax import numpy as jnp

import haiku as hk
import distrax

import blackjax
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.nuts import NUTSInfo
from blackjax.types import PyTree
import arviz as az

from tensorflow_probability.substrates import jax as tfp

from flax.metrics import tensorboard

import plot
from train_flow import load_data, get_inducing_points
from flows import get_global_params_shapes
import log_prob_fun_2
from log_prob_fun_2 import ModelParamsGlobal, ModelParamsLocations
from modularbayes import flatten_dict
from modularbayes._src.typing import (Any, Array, Batch, Callable, ConfigDict,
                                      Dict, List, Mapping, Optional, PRNGKey,
                                      Tuple)

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

  prng_keys = jax.random.split(prng_key, num_samples)
  _, (states, infos) = jax.lax.scan(one_step, initial_state, prng_keys)

  return states, infos


def inference_loop_multiple_chains(
    prng_key: PRNGKey,
    initial_states: HMCState,
    hmc_params: Dict[str, Array],
    logprob_fn: Callable,
    num_samples: int,
    num_chains: int,
) -> Tuple[HMCState, NUTSInfo]:

  def one_step(states, rng_key):
    rng_keys = jax.random.split(rng_key, num_chains)
    kernel_fn_multichain = jax.vmap(
        lambda state_, key_, hmc_param_: blackjax.nuts.kernel()(
            rng_key=key_,
            state=state_,
            logprob_fn=logprob_fn,
            step_size=hmc_param_['step_size'],
            inverse_mass_matrix=hmc_param_['inverse_mass_matrix'],
        ))
    states_new, infos_new = kernel_fn_multichain(states, rng_keys, hmc_params)
    return states_new, (states_new, infos_new)

  prng_keys = jax.random.split(prng_key, num_samples)
  _, (states, infos) = jax.lax.scan(one_step, initial_states, prng_keys)

  return states, infos


def inference_loop_stg2(
    prng_key: PRNGKey,
    initial_states: HMCState,
    hmc_params: Dict[str, Array],
    logprob_fn: Callable,
    model_params_global_unb_samples: ModelParamsGlobal,
    num_samples_stg1: int,
    num_samples_stg2: int,
    num_chains: int,
):

  # We only need to keep the last sample of the subchains
  # so instead of using jax.lax.scan, we use jax.lax.fori_loop
  # This allow us to save memory
  def one_step_fori(_, states_infos_key):
    states, _, key_ = states_infos_key

    kernel_fn_multichain = jax.vmap(
        lambda state, cond, key, hmc_param: blackjax.nuts.kernel()(
            rng_key=key,
            state=state,
            logprob_fn=lambda param_: logprob_fn(
                conditioning=cond,
                model_params=param_,
            ),
            step_size=hmc_param['step_size'],
            inverse_mass_matrix=hmc_param['inverse_mass_matrix'],
        ))
    kernel_fn_multicond_multichain = jax.vmap(
        lambda states_, conds_, keys_: kernel_fn_multichain(
            states_,
            conds_,
            keys_,
            hmc_params,
        ))

    key_out, key_aux = jax.random.split(key_)
    keys = jax.random.split(key_aux, num_samples_stg1 * num_chains).reshape(
        (num_samples_stg1, num_chains, 2))
    states_new, infos_new = kernel_fn_multicond_multichain(
        states,
        model_params_global_unb_samples,
        keys,
    )

    return states_new, infos_new, key_out

  initial_states_infos_key = one_step_fori(None,
                                           (initial_states, None, prng_key))

  states, infos, _ = jax.lax.fori_loop(0, num_samples_stg2, one_step_fori,
                                       initial_states_infos_key)

  return states, infos


def arviz_trace_from_samples(
    position: PyTree,
    burn_in: int = 0,
    info: Optional[NUTSInfo] = None,
):
  position = position._asdict()

  samples = {}
  for key, param_ in position.items():
    # k='mu'; v_ = position[k]
    param_list_ = isinstance(param_, List)
    for i in range(1 if not param_list_ else len(param_)):
      if param_list_:
        param = param_[i]
      else:
        param = param_

      ndims = len(param.shape)
      if ndims >= 2:
        samples[key + (f"_{str(i)}" if param_list_ else "")] = jnp.swapaxes(
            param, 0, 1)[:, burn_in:]  # swap n_samples and n_chains
        if info:
          divergence = jnp.swapaxes(info.is_divergent[burn_in:], 0, 1)

      if ndims == 1:
        samples[key] = param[burn_in:]
        if info:
          divergence = info.is_divergent[burn_in:]

  trace_posterior = az.convert_to_inference_data(samples)

  if info:
    trace_sample_stats = az.convert_to_inference_data({"diverging": divergence},
                                                      group="sample_stats")
    trace = az.concat(trace_posterior, trace_sample_stats)

  return trace


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
      'mu': distrax.Sigmoid(),
      'zeta': distrax.Sigmoid(),
      'loc_floating': distrax.Sigmoid(),
      'loc_floating_aux': distrax.Sigmoid(),
      'loc_random_anchor': distrax.Sigmoid(),
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
      model_params_global_unb.mu).sum()
  log_det_jacob_transformed += bijectors_['zeta'].forward_log_det_jacobian(
      model_params_global_unb.zeta).sum()
  if model_params_locations_unb.loc_floating is not None:
    log_det_jacob_transformed += bijectors_[
        'loc_floating'].forward_log_det_jacobian(
            model_params_locations_unb.loc_floating).sum()
  if model_params_locations_unb.loc_floating_aux is not None:
    log_det_jacob_transformed += bijectors_[
        'loc_floating_aux'].forward_log_det_jacobian(
            model_params_locations_unb.loc_floating_aux).sum()
  if model_params_locations_unb.loc_random_anchor is not None:
    log_det_jacob_transformed += bijectors_[
        'loc_random_anchor'].forward_log_det_jacobian(
            model_params_locations_unb.loc_random_anchor).sum()

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
       model_params_global_unb, model_params_locations_unb)

  # Sample the basis GPs on profiles locations conditional on GP values on the
  # inducing points.
  model_params_gamma_profiles = log_prob_fun_2.sample_gamma_profiles_given_gamma_inducing(
      batch=batch,
      model_params_global=model_params_global,
      model_params_locations=model_params_locations,
      prng_key=prng_key,
      kernel_name=kernel_name,
      kernel_kwargs=kernel_kwargs,
      gp_jitter=gp_jitter,
      num_samples_gamma_profiles=num_samples_gamma_profiles,
      is_smi=False,  # Do not sample the auxiliary gamma
      include_random_anchor=False,  # Do not sample gamma for random anchor locations
  )

  # Compute the log probability function.
  log_prob = log_prob_fun_2.log_prob_joint(
      batch=batch,
      model_params_global=model_params_global,
      model_params_locations=model_params_locations,
      model_params_gamma_profiles=model_params_gamma_profiles,
      smi_eta=smi_eta,
      **prior_hparams,
  )

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

  times_data = {}
  times_data['start_sampling'] = time.perf_counter()

  ### Sample First Stage ###
  if os.path.exists(samples_path_stg1):
    logging.info("\t Loading samples for stage 1...")
    aux_ = np.load(str(samples_path_stg1), allow_pickle=True)['arr_0']
    model_params_stg1_unb_samples = ModelParamsStg1(*aux_)
  else:
    logging.info("\t Stage 1...")

    # Define target log_prob as a function of the model parameters only
    prng_key_gamma = next(prng_seq)

    @jax.jit
    def log_prob_fn_stg1(model_params):
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

    # Sanity check stg1
    # Sample one chain
    sanity_check_stg1 = False
    if sanity_check_stg1:
      logging.info("\t sanity check stg 1...")
      model_params_stg1_unb_init_ = init_param_fn_stg1(
          prng_key=next(prng_seq),
          num_forms_tuple=config.num_forms_tuple,
          num_basis_gps=config.model_hparams.num_basis_gps,
          num_inducing_points=config.num_inducing_points,
          num_profiles_floating=config.num_profiles_floating,
      )
      log_prob_fn_stg1(model_params_stg1_unb_init_)

      initial_state_, hmc_params_ = call_warmup(
          prng_key=next(prng_seq),
          logprob_fn=log_prob_fn_stg1,
          model_params=model_params_stg1_unb_init_,
          num_steps=config.num_steps_call_warmup,
      )

      # 5 NUTS samples on a single chain
      inference_loop_one_chain(
          prng_key=next(prng_seq),
          initial_state=initial_state_,
          hmc_params=hmc_params_,
          logprob_fn=log_prob_fn_stg1,
          num_samples=5,
      )
      del model_params_stg1_unb_init_, initial_state_, hmc_params_
      logging.info("\t Sanity check stg1 finished succesfully.")

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
    initial_states_stg1, hmc_params_stg1 = jax.vmap(
        lambda prng_key, model_params: call_warmup(
            prng_key=prng_key,
            logprob_fn=log_prob_fn_stg1,
            model_params=model_params,
            num_steps=config.num_steps_call_warmup,
        ))(
            jax.random.split(next(prng_seq), config.num_chains),
            model_params_stg1_unb_init,
        )

    # Sampling loop stage 1
    logging.info('\t sampling stage 1...')
    states_stg1, infos_stg1 = inference_loop_multiple_chains(
        prng_key=next(prng_seq),
        initial_states=initial_states_stg1,
        hmc_params=hmc_params_stg1,
        logprob_fn=log_prob_fn_stg1,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
    )

    model_params_stg1_unb_samples = states_stg1.position

    # Save MCMC samples from stage 1
    np.savez_compressed(samples_path_stg1, model_params_stg1_unb_samples)

    logging.info("\t\t posterior means mu %s",
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
    model_params_stg2_unb_samples = ModelParamsStg1(*aux_)

  else:

    logging.info("\t Stage 2...")

    @jax.jit
    def log_prob_fn_stg2(model_params, conditioning):
      log_prob = log_prob_lalme(
          batch=train_ds,
          prng_key=prng_key_gamma,
          model_params_global_unb=conditioning,
          model_params_locations_unb=model_params,
          prior_hparams=config.prior_hparams,
          kernel_name=config.kernel_name,
          kernel_kwargs=config.kernel_kwargs,
          num_samples_gamma_profiles=config.num_samples_gamma_profiles,
          smi_eta_profiles=None,
          gp_jitter=config.gp_jitter,
      )
      return log_prob

    # Sanity check stg2
    # Sample one sub-chain
    sanity_check_stg2 = False
    if sanity_check_stg2:
      logging.info("\t sanity check stg 2...")

      # Each subchain requires:
      #   1) Conditioning fixed values of the global parameters, and
      #   2) an initial state for the floating locations sampling
      model_params_global_unb_cond_ = jax.tree_map(
          lambda x: x[0, 0], model_params_global_unb_samples)
      model_params_stg2_unb_init_ = ModelParamsLocations(
          loc_floating=model_params_stg1_unb_samples.loc_floating_aux[0,
                                                                      0, :, :],
          loc_floating_aux=None,
          loc_random_anchor=None,
      )
      initial_state_stg2_, hmc_params_stg2_ = call_warmup(
          prng_key=next(prng_seq),
          logprob_fn=lambda param_: log_prob_fn_stg2(
              model_params=param_,
              conditioning=model_params_global_unb_cond_,
          ),
          model_params=model_params_stg2_unb_init_,
          num_steps=10,
      )

      # The target logprob fixes the global parameters
      def one_step_(state, rng_key):
        logprob_fn_cond = lambda param_: log_prob_fn_stg2(
            model_params=param_,
            conditioning=model_params_global_unb_cond_,
        )
        state_new, info_new = blackjax.nuts.kernel()(
            rng_key=rng_key,
            state=state,
            logprob_fn=logprob_fn_cond,
            step_size=hmc_params_stg2_['step_size'],
            inverse_mass_matrix=hmc_params_stg2_['inverse_mass_matrix'],
        )
        return state_new, (state_new, info_new)

      one_step_(initial_state_stg2_, next(prng_seq))

      del (model_params_global_unb_cond_, model_params_stg2_unb_init_,
           initial_state_stg2_, hmc_params_stg2_, one_step_)

      logging.info("\t sanity check stg2 finished succesfully.")

    # Tune HMC parameters automatically
    logging.info('\t tuning HMC parameters stg2...')

    # We tune the HMC for one sample in stage 1
    # tune HMC parameters, vmap across chains
    initial_states_stg2, hmc_params_stg2 = jax.vmap(
        lambda key, param, cond: call_warmup(
            prng_key=key,
            logprob_fn=lambda param_: log_prob_fn_stg2(
                conditioning=cond,
                model_params=param_,
            ),
            model_params=param,
            num_steps=config.num_steps_call_warmup,
        ))(
            jax.random.split(next(prng_seq), config.num_chains),
            ModelParamsLocations(
                loc_floating=model_params_stg1_unb_samples.loc_floating_aux[0],
                loc_floating_aux=None,
                loc_random_anchor=None,
            ),
            jax.tree_map(lambda x: x[0], model_params_global_unb_samples),
        )
    # Copy these tuned HMC parameters to use it in all other samples
    initial_states_stg2_expanded = jax.tree_map(
        lambda x: jnp.broadcast_to(x, (config.num_samples,) + x.shape),
        initial_states_stg2)

    # Sampling loop stage 2
    logging.info('\t sampling stage 2...')
    states_stg2, infos_stg2 = inference_loop_stg2(
        prng_key=next(prng_seq),
        initial_states=initial_states_stg2_expanded,
        hmc_params=hmc_params_stg2,
        logprob_fn=log_prob_fn_stg2,
        model_params_global_unb_samples=model_params_global_unb_samples,
        num_samples_stg1=config.num_samples,
        num_samples_stg2=config.num_samples_stg2,
        num_chains=config.num_chains,
    )

    model_params_stg2_unb_samples = states_stg2.position

    # Save MCMC samples from stage 1
    np.savez_compressed(samples_path_stg2, model_params_stg2_unb_samples)

    logging.info(
        "\t\t posterior means loc_floating %s",
        str(model_params_stg2_unb_samples.loc_floating.mean(axis=[0, 1])))

    times_data['end_mcmc_stg_2'] = time.perf_counter()

  # Transform unbounded parameters to model parameters
  (model_params_global, model_params_locations, _) = transform_model_params(
      model_params_global_unb=model_params_global_unb_samples,
      model_params_locations_unb=ModelParamsLocations(
          loc_floating=model_params_stg2_unb_samples.loc_floating,
          loc_floating_aux=model_params_stg1_unb_samples.loc_floating_aux,
          loc_random_anchor=None,
      ),
  )

  # make arviz trace from states
  trace_global = arviz_trace_from_samples(
      position=model_params_global,
      info=infos_stg1,
      burn_in=config.num_burnin_steps_stg1,
  )
  trace_locations = arviz_trace_from_samples(
      position=model_params_locations,
      info=infos_stg2,
      burn_in=config.num_burnin_steps_stg1,
  )

  az.summary(trace_global)
  az.summary(trace_locations)

  az.plot_trace(trace_global)
  plt.tight_layout()

  az.plot_trace(trace_locations)
  plt.tight_layout()


# # For debugging
# config = get_config()
# config.num_samples = 21
# config.num_burnin_steps_stg1 = 5
# config.num_samples_subchain_stg2 = 7
# config.num_chunks_stg2 = 5
# config.mcmc_step_size = 0.001
# eta = 1.000
# import pathlib
# workdir = str(pathlib.Path.home() / f'spatial-smi-output-exp/8_items/mcmc/eta_floating_{eta:.3f}')
# config.path_variational_samples = str(pathlib.Path.home() / f'spatial-smi-output-exp/8_items/nsf/eta_floating_{eta:.3f}/posterior_sample_dict.npz')
# # sample_and_evaluate(config, workdir)
