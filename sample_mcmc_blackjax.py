"""MCMC sampling for the LALME model."""

import os
os.environ["TP_CPP_MIN_LOG_LEVEL"]="0" #TP_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=bfc_allocator=1
os.environ["TF_CPP_VMODULE"]="bfc_allocator=1"

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

import blackjax
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.nuts import NUTSInfo
# from blackjax.types import PyTree
from jax.experimental import host_callback
import ot
from tensorflow_probability.substrates import jax as tfp

from flax.metrics import tensorboard

from modularbayes import flatten_dict
from modularbayes._src.typing import (Any, Array, Batch, Callable, ConfigDict,
                                      Dict, Mapping, Optional, PRNGKey, Tuple)

import plot
from train_flow import load_data, get_inducing_points
from flows import get_global_params_shapes
import log_prob_fun
from log_prob_fun import ModelParamsGlobal, ModelParamsLocations


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
    logdensity_fn: Callable,
    model_params, #: PyTree,
    num_steps: int,
) -> Tuple:
  """
    Run a warm-up phase for Hamiltonian Monte Carlo (HMC) sampling.
    Parameters:
    -----------
    prng_key : jax.random.PRNGKey
        A pseudo-random number generator key for reproducible random number generation.

    logdensity_fn : callable
        A function that computes the log-density of the target distribution. This function guides the sampling process.

    model_params : jax.numpy.ndarray
        The model parameters, representing the current state of the probabilistic model.

    num_steps : int
        The number of warm-up steps to perform in the HMC algorithm. More steps may lead to better adaptation but can be computationally expensive.

    Returns:
    --------
    Tuple[jax.numpy.ndarray, dict]
        A tuple containing the results of the warm-up phase:
        - initial_states : jax.numpy.ndarray
            The initial states of the HMC algorithm after warm-up. These represent samples from the posterior distribution.
        - hmc_params : dict
            Parameters of the HMC algorithm after adaptation, including information about step size and mass matrix.
    
    Notes:
    ------
    - Warm-up is a crucial step in Bayesian sampling to adapt the HMC algorithm for efficient posterior exploration.
    - This function uses the `window_adaptation` and `warmup.run` functions from a HMC library to perform warm-up.
    - `window_adaptation` is responsible for adapting the step size and mass matrix of the HMC algorithm.
    - `warmup.run` executes the warm-up phase of HMC sampling with adaptive parameters.
    
    Example:
    --------
    # Define the log-density function and model parameters
    def log_density(x):
        # Compute the log-density of the target distribution
        ...

    initial_params = ...

    # Run warm-up with 1000 steps
    initial_states, hmc_params = call_warmup(rng_key, log_density, initial_params, num_steps=1000)

    # Use initial_states and hmc_params for posterior sampling
    ...
    """
  warmup = blackjax.window_adaptation(
      algorithm=blackjax.nuts,
      logdensity_fn=logdensity_fn,
  )
  (initial_states, hmc_params), _ = warmup.run( #they were in.., _, hmc.. before
      rng_key=prng_key,
      position=model_params,
      num_steps=num_steps,
  )
  return initial_states, hmc_params


def inference_loop_one_chain(
    prng_key: PRNGKey,
    initial_state: HMCState,
    hmc_params: Dict[str, Array],
    logdensity_fn: Callable,
    num_samples: int,
) -> Tuple[HMCState, NUTSInfo]:

  def one_step(state, rng_key):
    kernel_fn = lambda state_, key_, hmc_param_: blackjax.nuts.build_kernel()(
        rng_key=key_,
        state=state_,
        logdensity_fn=logdensity_fn,
        step_size=hmc_param_['step_size'],
        inverse_mass_matrix=hmc_param_['inverse_mass_matrix'],
    )
    state_new, info_new = kernel_fn(state, rng_key, hmc_params)
    return state_new, (state_new, info_new)

  keys = jax.random.split(prng_key, num_samples)
  _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

  return states, infos



def _print_consumer(arg, transforms):
    i, n_iter = arg
    print(f"Iteration {i}/{n_iter}")

@jax.jit
def progress_bar(arg, result):
    "Print progress of loop only if iteration number is a multiple of the print_rate"
    i, n_iter, print_rate = arg
    result = jax.lax.cond(
        i%print_rate==0,
        lambda _: host_callback.id_tap(_print_consumer, (i, n_iter), result=result), 
        lambda _: result,
        operand=None)
    return result

# @jax.jit
# def jax_loop(a):
#     """
#     Jax loop that increments `a` 100 times
#     """
#     n_iter, print_rate = 100, 10
#     def body(carry, x):
#         carry = progress_bar((x, n_iter, print_rate), carry)
#         carry += 1
#         return carry, None
#     carry, _ = jax.lax.scan(body, a, jax.numpy.arange(n_iter))
#     return carry


# @jax.jit
def inference_loop_stg1_init(
    prng_key: PRNGKey,
    initial_states: HMCState,
    hmc_params: Dict[str, Array],
    logdensity_fn: Callable,
    num_samples: int,
    num_chains: int,
) -> Tuple[HMCState, NUTSInfo]:
  jax.debug.print("Inference loop stage 1 started")
  logging.info("Inference loop stage 1 started")
  kernel_fn_multichain = jax.vmap(
        lambda state_, hmc_param_, key_nuts_, key_gamma_: blackjax.nuts.build_kernel(
        )(
            rng_key=key_nuts_,
            state=state_,
            logdensity_fn=lambda x: logdensity_fn(x, key_gamma_),
            step_size=hmc_param_['step_size'],
            inverse_mass_matrix=hmc_param_['inverse_mass_matrix'],
        ))
  
  def one_step(states, i):#rng_keys):
    # if i%5==0:
    jax.debug.print(f"{i} Step starting")
    logging.info(f"{i} Step starting!") # prints at compiling stage
    rng_keys = jax.random.split(jax.random.PRNGKey(i), num_chains * 2).reshape(
      (num_chains, 2, 2))
    keys_nuts_, key_gamma_ = jnp.split(rng_keys, 2, axis=-2)
    # states = progress_bar((i, num_samples, 5), states)
    states_new, infos_new = kernel_fn_multichain(
        states,
        hmc_params,
        keys_nuts_.squeeze(-2),
        key_gamma_.squeeze(-2),
    )
    return states_new, (states_new, None) # infos_new

  # Not using this for being able to use progress bar
  keys = jax.random.split(prng_key, num_samples * num_chains * 2).reshape(
      (num_samples, num_chains, 2, 2))

  # one_step(initial_states, keys[0])

  _, (states, infos) = jax.lax.scan(one_step, initial_states, jnp.arange(num_samples)) # keys)

  return states, infos

def inference_loop_stg1(
    prng_key: PRNGKey,
    initial_states: HMCState,
    hmc_params: Dict[str, Array],
    logdensity_fn: Callable,
    num_samples: int,
    num_chains: int,
) -> Tuple[HMCState, NUTSInfo]:

  def one_step(states, rng_keys):
    # loop over chains
    kernel_fn_multichain1 = jax.vmap(
        lambda state_, hmc_param_, key_nuts_, key_gamma_: blackjax.nuts.build_kernel(
        )(
            rng_key=key_nuts_,
            state=state_,
            logdensity_fn=lambda x: logdensity_fn(x, key_gamma_),
            step_size=hmc_param_['step_size'],
            inverse_mass_matrix=hmc_param_['inverse_mass_matrix'],
        ))
    ######################################

    # loop over inner samples (here 1?)
    keys_nuts_, keys_gamma_ = jnp.split(rng_keys, 2, axis=-2)

    kernel_fn_multichain2 = jax.vmap(
        lambda states_: kernel_fn_multichain1(
            states_,
            hmc_params,
            keys_nuts_.squeeze(-2), # get rid of the 1 dim in pos -2
            keys_gamma_.squeeze(-2),
        ))
    ######################################
    
    states_new, infos_new = kernel_fn_multichain2(
        states,
        # hmc_params,
        # keys_nuts_.squeeze(-2),
        # key_gamma_.squeeze(-2),
    )
    return states_new, (states_new, infos_new)

  keys = jax.random.split(
    prng_key, num_samples * num_chains * 2).reshape(
      (num_samples, num_chains, 2, 2))

  # one_step(initial_states, keys[0])

  # scan will apply f starting from init state and looping over first dimension of keys, so n_samples
  _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

  return states, infos


def inference_loop_stg2(
    prng_key: PRNGKey,
    initial_states: HMCState,
    hmc_params: Dict[str, Array],
    logdensity_fn_conditional: Callable,
    conditioner_logprob: ModelParamsGlobal,
    num_samples_stg1: int,
    num_samples_stg2: int,
    num_chains: int,
):
  kernel_fn_multichain = jax.vmap(
        lambda state, cond, hmc_param, key_nuts_, key_gamma_: blackjax.nuts.
        build_kernel()(
            rng_key=key_nuts_,
            state=state,
            logdensity_fn=lambda param_: logdensity_fn_conditional(
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
  # We only need to keep the last sample of the subchains
  def one_step(states, rng_keys):
    jax.debug.print(f"Step starting")

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


def logprob_lalme(
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
      lambda key_: log_prob_fun.sample_gamma_profiles_given_gamma_inducing(
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
                      log_prob_fun.logprob_joint(
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

  # globals().update(dict(prior_hparams))
  # model_params_gamma_profiles = jax.tree_map(lambda x: x[0], model_params_gamma_profiles_sample)
  # gamma_profiles_logprob = jax.tree_map(lambda x: x[0], gamma_profiles_logprob_sample)

  return log_prob + log_det_jacob_transformed


def sample_and_evaluate(config: ConfigDict, workdir: str) -> Mapping[str, Any]:
  """Sample and evaluate the random effects model."""

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
  config.num_inducing_points = math.prod(
      config.model_hparams.inducing_grid_shape)

  samples_path_stg1 = workdir + '/lalme_stg1_unb_az_10_000s'+ (f'_thinning{config.thinning}' if config.thinning!=1 else "")+ '.nc'
  samples_path = workdir + '/lalme_az_10_000s' + (f'_thinning{config.thinning}' if config.thinning!=1 else "")+ '.nc'

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
    lalme_stg1_unb_az = None
  else:
    times_data = {}
    times_data['start_sampling'] = time.perf_counter()

    ### Sample First Stage ###
    if os.path.exists(samples_path_stg1):
      logging.info("\t Loading samples for stage 1...")
      lalme_stg1_unb_az = az.from_netcdf(samples_path_stg1)
    else:
      logging.info("\t Stage 1...")

      # Define target logdensity function
      @jax.jit
      def logdensity_fn_stg1(model_params, prng_key_gamma):
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

        logprob_ = logprob_lalme(
            batch=train_ds,
            prng_key=prng_key_gamma,
            model_params_global_unb=model_params_global_unb,
            model_params_locations_unb=model_params_locations_unb,
            prior_hparams=config.prior_hparams,
            kernel_name=config.kernel_name,
            kernel_kwargs=config.kernel_kwargs,
            num_samples_gamma_profiles=config.num_samples_gamma_profiles,
            smi_eta_profiles=smi_eta['profiles'],
            gp_jitter=config.gp_jitter,
        )
        return logprob_

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
      #initial_states_stg1 will have shape (n_chains, ....)
      logging.info('\t tuning HMC parameters stg1...')
      key_gamma_ = next(prng_seq)
      initial_states_stg1, hmc_params_stg1 = jax.vmap( 
          lambda prng_key, model_params: call_warmup(
              prng_key=prng_key,
              logdensity_fn=lambda x: logdensity_fn_stg1(x, key_gamma_),
              model_params=model_params,
              num_steps=config.num_steps_call_warmup,
          ))(
              jax.random.split(next(prng_seq), config.num_chains),
              model_params_stg1_unb_init,
          )

####################################################################################################
    #  # Sampling loop stage 1 (with chunks)
    #  # The number of samples is large and often it does not fit into GPU memory
    #   # we split the sampling of stage 1 into chunks
    #   assert config.num_samples % config.num_samples_perchunk_stg1 == 0
    #   num_chunks_stg1 = config.num_samples // config.num_samples_perchunk_stg1

    #   # Initialize stage 1 using loc_floating_aux
    #   # we use the tuned HMC parameters from above
    #   # Note: vmap is first applied to the chains, then to samples from
    #   #   conditioner this requires swap axes 0 and 1 in a few places

    #   # multichain1 seems to imply that also hmc_params is of shape (n_chains, ...)
    #   # and that multichain1 expects to vmap param over chains too (but wasn't first dim of params n_samples?)
    #   # also n_samples is n MCMC samples and logdensity_fn_stg1 wants one sample one chain at a time I think
    #   init_fn_multichain1 = jax.vmap(lambda param, hmc_param: blackjax.nuts(
    #       logdensity_fn=lambda param_: logdensity_fn_stg1(
    #           model_params=param_,
    #           prng_key_gamma=key_gamma_,
    #       ),
    #       step_size=hmc_param['step_size'],
    #       inverse_mass_matrix=hmc_param['inverse_mass_matrix'],
    #   ).init(position=param))

    #   # ok so this acts first, and loops over (n_samples,...) of params, and then above loops over n_chains of both?
    #   init_fn_multichain2 = jax.vmap(
    #       lambda param_: init_fn_multichain1(
    #           param=param_,
    #           hmc_param=hmc_params_stg1,
    #       ))

    #   # The initial position for loc_floating in the first chunk is the location
    #   # of loc_floating_aux from stage 1
    # #   initial_model_params_i = ModelParamsStg1(
    # #         gamma_inducing=(initial_states_stg1.position.gamma_inducing)[:, :config.num_samples_perchunk_stg1,
    # #        ...].swapaxes(0,
    # #                      1),
    # #         mixing_weights_list=[(w)[:, :config.num_samples_perchunk_stg1,
    # #        ...].swapaxes(0,
    # #                      1) for w in initial_states_stg1.position.mixing_weights_list],
    # #         mixing_offset_list=[(o)[:, :config.num_samples_perchunk_stg1,
    # #        ...].swapaxes(0,
    # #                      1) for o in initial_states_stg1.position.mixing_offset_list],
    # #         mu=(initial_states_stg1.position.mu)[:, :config.num_samples_perchunk_stg1,
    # #        ...].swapaxes(0,
    # #                      1),
    # #         zeta=(initial_states_stg1.position.zeta)[:, :config.num_samples_perchunk_stg1,
    # #        ...].swapaxes(0,
    # #                      1),
    # #         loc_floating_aux=(initial_states_stg1.position.loc_floating_aux)[:, :config.num_samples_perchunk_stg1,
    # #        ...].swapaxes(0,
    # #                      1),
    # #     )
    #   initial_model_params_i = ModelParamsStg1(
    #         gamma_inducing=initial_states_stg1.position.gamma_inducing,
    #         mixing_weights_list=initial_states_stg1.position.mixing_weights_list,
    #         mixing_offset_list=initial_states_stg1.position.mixing_offset_list,
    #         mu=initial_states_stg1.position.mu,
    #         zeta=initial_states_stg1.position.zeta,
    #         loc_floating_aux=initial_states_stg1.position.loc_floating_aux,
    #     )

    #   logging.info('\t sampling stage 1...')

    #   #initial_states_stg1 will have shape (n_chains, ....)
    #   initial_state_i = initial_states_stg1 #
    #   chunks_positions = []
    #   for i in range(num_chunks_stg1):
    #     print(i)
    #     if i!=0:
    #         initial_state_i = init_fn_multichain2(initial_model_params_i) #, hmc_param=hmc_params_stg1)

    #         # Sampling loop stage 2 after first round
    #         # I think states_stg1_i will have shape (n_samples, n_chains, ....)
    #         states_stg1_i, _ = inference_loop_stg1(
    #             prng_key=next(prng_seq),
    #             initial_states=initial_state_i,
    #             hmc_params=hmc_params_stg1,
    #             logdensity_fn=logdensity_fn_stg1,
    #             num_samples=config.num_samples_perchunk_stg1,
    #             num_chains=config.num_chains,
    #         )
    #     else:
    #         # Sampling loop stage 2 init
    #         # I think states_stg1_i will have shape (n_samples, n_chains, ....)
    #         states_stg1_i, _ = inference_loop_stg1_init(
    #             prng_key=next(prng_seq),
    #             initial_states=initial_state_i,
    #             hmc_params=hmc_params_stg1,
    #             logdensity_fn=logdensity_fn_stg1,
    #             num_samples=config.num_samples_perchunk_stg1,
    #             num_chains=config.num_chains,
    #         )

    #     chunks_positions.append(states_stg1_i.position)

    #     # Subsequent chunks initialise in last position of the previous chunk
    #     initial_model_params_i = states_stg1_i.position

    #   times_data['end_mcmc_stg_1'] = time.perf_counter()

    #   # Concatenate samples from each chunk, across samples dimension
    #   model_params_stg1_unb_samples = jax.tree_map(  # pylint: disable=no-value-for-parameter
    #       lambda *x: jnp.concatenate(x, axis=0), *chunks_positions)
    #   # swap axes to have shape (num_chains, num_samples, ...)
    #   model_params_stg1_unb_samples = jax.tree_map(lambda x: x.swapaxes(0, 1),
    #                                                model_params_stg1_unb_samples)

####################################################################################################
      #Sampling loop stage 1 (no chunks)
      logging.info('\t sampling stage 1...')
      states_stg1, _ = inference_loop_stg1_init(
          prng_key=next(prng_seq),
          initial_states=initial_states_stg1,
          hmc_params=hmc_params_stg1,
          logdensity_fn=logdensity_fn_stg1,
          num_samples=config.num_samples,
          num_chains=config.num_chains,
      )

      # Save samples from stage 1
      # swap position axes to have shape (num_chains, num_samples, ...)
      model_params_stg1_unb_samples = jax.tree_map(lambda x: x.swapaxes(0, 1),
                                                   states_stg1.position)
####################################################################################################
      jax.debug.print("Create InferenceData object") # {x}",x=var_name)
      logging.info("Create InferenceData object")
      # Create InferenceData object
      lalme_stg1_unb_az = plot.lalme_az_from_samples(
          lalme_dataset=lalme_dataset,
          thinning=config.thinning,
          model_params_global=ModelParamsGlobal(
              gamma_inducing=model_params_stg1_unb_samples.gamma_inducing,
              mixing_weights_list=model_params_stg1_unb_samples
              .mixing_weights_list,
              mixing_offset_list=model_params_stg1_unb_samples
              .mixing_offset_list,
              mu=model_params_stg1_unb_samples.mu,
              zeta=model_params_stg1_unb_samples.zeta,
          ),
          model_params_locations=ModelParamsLocations(
              loc_floating=None,
              loc_floating_aux=model_params_stg1_unb_samples.loc_floating_aux,
              loc_random_anchor=None,
          ),
          model_params_gamma=None,
      )
      # Save InferenceData object from stage 1
      lalme_stg1_unb_az.to_netcdf(samples_path_stg1)

      logging.info(
          "\t\t posterior means mu (before transform):  %s",
          str(jnp.array(lalme_stg1_unb_az.posterior.mu).mean(axis=[0, 1])))

      times_data['end_mcmc_stg_1'] = time.perf_counter()

    ### Sample Second Stage ###
    logging.info("\t Stage 2...")
    jax.debug.print("\t Stage 2... jax debug")

    # Extract global parameters from stage 1 samples
    model_params_global_unb_samples = ModelParamsGlobal(
        gamma_inducing=jnp.array(lalme_stg1_unb_az.posterior.gamma_inducing),
        mixing_weights_list=[
            jnp.array(lalme_stg1_unb_az.posterior[f'W_{i}'])
            for i in range(len(config.num_forms_tuple))
        ],
        mixing_offset_list=[
            jnp.array(lalme_stg1_unb_az.posterior[f'a_{i}'])
            for i in range(len(config.num_forms_tuple))
        ],
        mu=jnp.array(lalme_stg1_unb_az.posterior.mu),
        zeta=jnp.array(lalme_stg1_unb_az.posterior.zeta),
    )
    # ADDED
    model_params_loc_floating_aux_samples = jnp.array(lalme_stg1_unb_az.posterior.loc_floating_aux)
    del lalme_stg1_unb_az
    # ADDED END
    
    # Define target logdensity function
    @jax.jit
    def logdensity_fn_stg2(model_params, conditioner, prng_key_gamma):
      logprob_ = logprob_lalme(
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
      return logprob_

    # Tune HMC parameters automatically
    logging.info('\t tuning HMC parameters stg2...')

    # We tune the HMC for one sample in stage 1
    # tune HMC parameters, vmap across chains
    key_gamma_ = next(prng_seq)
    _, hmc_params_stg2 = jax.vmap(lambda key, param, cond: call_warmup(
        prng_key=key,
        logdensity_fn=lambda param_: logdensity_fn_stg2(
            conditioner=cond,
            model_params=param_,
            prng_key_gamma=key_gamma_,
        ),
        model_params=param,
        num_steps=config.num_steps_call_warmup,
    ))(
        jax.random.split(next(prng_seq), config.num_chains),
        ModelParamsLocations(
            loc_floating=model_params_loc_floating_aux_samples[:, 0, ...],#jnp.array(lalme_stg1_unb_az.posterior.loc_floating_aux)[:, 0, ...],
            loc_floating_aux=None,
            loc_random_anchor=None,
        ),
        jax.tree_map(lambda x: x[:, 0, ...], model_params_global_unb_samples),
    )

    # The number of samples is large and often it does not fit into GPU memory
    # we split the sampling of stage 2 into chunks
    num_effective_samples = int(config.num_samples/config.thinning)
    assert num_effective_samples % config.num_samples_perchunk_stg2 == 0
    num_chunks_stg2 = num_effective_samples// config.num_samples_perchunk_stg2

    # Initialize stage 1 using loc_floating_aux
    # we use the tuned HMC parameters from above
    # Note: vmap is first applied to the chains, then to samples from
    #   conditioner this requires swap axes 0 and 1 in a few places
    # my_logdensity_fn=lambda param_: logdensity_fn_stg2(
    #         conditioner=cond,
    #         model_params=param_,
    #         prng_key_gamma=key_gamma_,
    #     )
    # logdensity, logdensity_grad = jax.value_and_grad(my_logdensity_fn)(model_params_global_unb_samples)
    init_fn_multichain = jax.vmap(lambda param, cond, hmc_param: blackjax.nuts(
        logdensity_fn=lambda param_: logdensity_fn_stg2(
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
        loc_floating=model_params_loc_floating_aux_samples#jnp.array(lalme_stg1_unb_az.posterior.loc_floating_aux)
        [:, :config.num_samples_perchunk_stg2,
         ...].swapaxes(0,
                       1),  # swap axes to have (num_samples, num_chains, ...)
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
          logdensity_fn_conditional=logdensity_fn_stg2,
          conditioner_logprob=cond_i,
          num_samples_stg1=config.num_samples_perchunk_stg2,
          num_samples_stg2=config.num_samples_subchain_stg2,
          num_chains=config.num_chains,
      )

      chunks_positions.append(states_stg2_i.position)

      # Subsequent chunks initialise in last position of the previous chunk
      initial_position_i = states_stg2_i.position

    times_data['end_mcmc_stg_2'] = time.perf_counter()

    # Concatenate samples from each chunk, across samples dimension
    model_params_stg2_unb_samples = jax.tree_map(  # pylint: disable=no-value-for-parameter
        lambda *x: jnp.concatenate(x, axis=0), *chunks_positions)
    # swap axes to have shape (num_chains, num_samples, ...)
    model_params_stg2_unb_samples = jax.tree_map(lambda x: x.swapaxes(0, 1),
                                                 model_params_stg2_unb_samples)

    # Transform unbounded parameters to model parameters
    (model_params_global_samples, model_params_locations_samples,
     _) = transform_model_params(
         model_params_global_unb=model_params_global_unb_samples,
         model_params_locations_unb=ModelParamsLocations(
             loc_floating=model_params_stg2_unb_samples.loc_floating,
             loc_floating_aux=model_params_loc_floating_aux_samples,#jnp.array(lalme_stg1_unb_az.posterior.loc_floating_aux),
             loc_random_anchor=None,
         ),
     )

    # Create InferenceData object
    lalme_az = plot.lalme_az_from_samples(
        lalme_dataset=lalme_dataset,
        model_params_global=model_params_global_samples,
        model_params_locations=model_params_locations_samples,
        model_params_gamma=None,
    )
    # Save InferenceData object
    lalme_az.to_netcdf(samples_path)

    logging.info(
        "\t\t posterior means loc_floating (before transform): %s",
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
      jax.vmap(lambda key_, global_, locations_: log_prob_fun.
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
                   jax.random.split(
                       next(prng_seq),
                       config.num_chains * int((config.num_samples/config.thinning))).reshape(
                           (config.num_chains, int((config.num_samples/config.thinning)), 2)),
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
  # Load samples to compare MCMC vs Variational posteriors
  if (config.path_variational_samples != ''):
    Wass_dict = {'LP_order':config.lp_floating_grid10}
    loc_samples_mcmc = jnp.vstack(jnp.array(lalme_az_with_gamma.posterior[f'loc_floating'].reindex(LP_floating=config.lp_floating_grid10)))
    loc_samples_mcmc = loc_samples_mcmc[1000:] # just last 500 samples
    n_samples_mcmc = loc_samples_mcmc.shape[0]
    if n_samples_mcmc > config.max_wass_samples:
        idxs = jax.random.choice(key=next(prng_seq), a=n_samples_mcmc, shape=(config.max_wass_samples,))
        loc_samples_mcmc = loc_samples_mcmc[idxs]
        n_samples_mcmc = config.max_wass_samples
    VI_path_dict = eval(config.path_variational_samples)
    for VI_name, VI_path in VI_path_dict.items():
      if os.path.exists(VI_path):
        logging.info("Plotting comparison MCMC and Variational...")
        lalme_az_variational = az.from_netcdf(VI_path)
        loc_samples_VI = jnp.array(lalme_az_variational.posterior[f'loc_floating'].reindex(LP_floating=config.lp_floating_grid10).squeeze()) # assume (n_samples, n_locs, 2)
        n_samples_VI = loc_samples_VI.shape[0]
        if n_samples_VI > config.max_wass_samples:
            idxs = jax.random.choice(key=next(prng_seq), a=n_samples_VI, shape=(config.max_wass_samples,))
            loc_samples_VI = loc_samples_VI[idxs]
            n_samples_VI = config.max_wass_samples
        a_mcmc = jnp.ones((n_samples_mcmc,)) / n_samples_mcmc
        b_VI = jnp.ones((n_samples_VI,)) / n_samples_VI
        
        # Need to swap axis from (n_samples, n_locations, ...) to (n_locations, n_samples, ...)
        Wass_dict[VI_name] = jnp.array([ot.emd2(a_mcmc, b_VI, ot.dist(x, y, metric='euclidean')) 
                                        for (x,y) in zip(loc_samples_mcmc.swapaxes(0,1), loc_samples_VI.swapaxes(0,1))])

            
        plot.posterior_samples_compare(
            lalme_az_1=lalme_az_with_gamma,
            lalme_az_2=lalme_az_variational,
            lalme_dataset=lalme_dataset,
            step=0,
            lp_floating_grid10=config.lp_floating_grid10,
            # show_mu=(lalme_dataset['num_items'] <= 8),
            # show_zeta=(lalme_dataset['num_items'] <= 8),
            wass_dists={lp_: wass for lp_, wass in zip(Wass_dict['LP_order'], Wass_dict[VI_name])},
            summary_writer=summary_writer,
            workdir_png=workdir,
            suffix=f"_{VI_name}_eta_floating_{float(config.eta_profiles_floating):.3f}",
            scatter_kwargs={"alpha": 0.3},#0.03
            data_labels=["MCMC", "VI"],
        )
    def generate_latex_wd_table(data):
        latex_code = """
            \\begin{table}[ht]
            \\centering
            \\begin{tabular}{lccc}
            \\toprule
            & \\multicolumn{3}{c}{Models} \\\\
            \\cmidrule(lr){2-4}
            & VMP & VP & Additive-VMP \\\\
            \\midrule
            WD"""
                
        for model in ['VMP', 'VP', 'Additive-VMP']:
            wd_value = data[model]['WD']
            ci_lower, ci_upper = data[model]['CI']
            latex_code += f" & {wd_value:.2f}"
            latex_code += "\\\\\n& " if model == "Additive-VMP" else " "
        latex_code += "& " + " & ".join(f"({ci_lower:.2f}, {ci_upper:.2f})" for model in ['VMP', 'VP', 'Additive-VMP'] for ci_lower, ci_upper in [data[model]['CI']])
        
        latex_code += """
            \\bottomrule
            \\end{tabular}
            \\caption{Wasserstein Distances (WD) and Confidence Intervals (CIs) for different models.}
            \\label{tab:WD_CIs}
            \\end{table}
            """
        return latex_code

    # Example data
    latex_data = {
        'VMP': {'WD': Wass_dict['VMP'].mean(), 
                'CI': (Wass_dict['VMP'].mean()-Wass_dict['VMP'].std()*1.96/jnp.sqrt(len(config.lp_floating_grid10)), 
                       Wass_dict['VMP'].mean()+Wass_dict['VMP'].std()*1.96/jnp.sqrt(len(config.lp_floating_grid10)))},
        'VP': {'WD': Wass_dict['ADDITIVE-VMP'].mean(), 
                'CI': (Wass_dict['VMP'].mean()-Wass_dict['ADDITIVE-VMP'].std()*1.96/jnp.sqrt(len(config.lp_floating_grid10)), 
                       Wass_dict['VMP'].mean()+Wass_dict['ADDITIVE-VMP'].std()*1.96/jnp.sqrt(len(config.lp_floating_grid10)))},
        'Additive-VMP': {'WD': Wass_dict['VP'].mean(), 
                'CI': (Wass_dict['VMP'].mean()-Wass_dict['VP'].std()*1.96/jnp.sqrt(len(config.lp_floating_grid10)), 
                       Wass_dict['VMP'].mean()+Wass_dict['VP'].std()*1.96/jnp.sqrt(len(config.lp_floating_grid10)))}
    }

    # Generate LaTeX table code
    latex_table_code = generate_latex_wd_table(latex_data)

    # Print the LaTeX code
    print(latex_table_code)

    logging.info("...done!")

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

#   del lalme_az

  if config.plot_floating_aux:
    #   if not lalme_stg1_unb_az:
    #     logging.info("\t Loading samples for stage 1 for aux plot...")
    #     lalme_stg1_unb_az = az.from_netcdf(samples_path_stg1)

      plot.lalme_plots_arviz(

            lalme_az=lalme_az_with_gamma, #lalme_stg1_unb_az,
            lalme_dataset=lalme_dataset,
            step=0,
            lp_floating_aux_traces=config.lp_floating_grid10,
            lp_floating_aux_grid10=config.lp_floating_grid10,
            workdir_png=workdir,
            summary_writer=summary_writer,
            suffix=f"_eta_floating_{float(config.eta_profiles_floating):.3f}",
            scatter_kwargs={"alpha": 0.05},
    )
  logging.info("...done!")







# # For debugging
# config = get_config()
# eta = 1.000
# import pathlib
# workdir = str(pathlib.Path.home() / f'spatial-smi-output-exp/8_items/mcmc/eta_floating_{eta:.3f}')
# config.path_variational_samples = str(pathlib.Path.home() / f'spatial-smi-output-exp/8_items/nsf/eta_floating_{eta:.3f}/lalme_az.nc')
# # config.num_samples = 100
# # config.num_samples_subchain_stg2 = 10
# # config.num_samples_perchunk_stg2 = 50
# # config.num_steps_call_warmup = 10
# # # sample_and_evaluate(config, workdir)
