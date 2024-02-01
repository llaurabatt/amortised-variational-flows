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

  samples_path_stg1 = workdir + '/lalme_stg1_unb_az.nc'
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

  logging.info("\t Loading samples for stage 1...")
  lalme_stg1_unb_az = az.from_netcdf(samples_path_stg1)
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
    
    # Define target logdensity function

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
            loc_floating=jnp.array(
                lalme_stg1_unb_az.posterior.loc_floating_aux)[:, 0, ...],
            loc_floating_aux=None,
            loc_random_anchor=None,
        ),
        jax.tree_map(lambda x: x[:, 0, ...], model_params_global_unb_samples),
    )

    # The number of samples is large and often it does not fit into GPU memory
    # we split the sampling of stage 2 into chunks
  assert config.num_samples % config.num_samples_perchunk_stg2 == 0
  num_chunks_stg2 = config.num_samples // config.num_samples_perchunk_stg2

    # Initialize stage 1 using loc_floating_aux
    # we use the tuned HMC parameters from above
    # Note: vmap is first applied to the chains, then to samples from
    #   conditioner this requires swap axes 0 and 1 in a few places
  init_fn_multichain = jax.vmap(lambda param, cond, hmc_param: blackjax.nuts(
        logdensity_fn=lambda param_: logdensity_fn_stg2(
            conditioner=cond,
            model_params=param_,
            prng_key_gamma=key_gamma_,
        ),
        step_size=hmc_param['step_size'],
        inverse_mass_matrix=hmc_param['inverse_mass_matrix'],
    ).init(position=param))
