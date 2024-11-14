"""A simple example of a flow model trained on Epidemiology data."""

import pathlib
import sys
import os
import warnings
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256) 

from absl import logging

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import ot
import pickle
import pandas as pd
import seaborn as sns
import scipy
import time

from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax
import distrax

import flows_all_integrated
import log_prob_fun_integrated
import plot_all
from train_flow import load_dataset, make_optimizer

from modularbayes._src.utils.training import TrainState
from modularbayes import (plot_to_image, normalize_images, flatten_dict, initial_state_ckpt, 
update_state, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict, List,
                                      Optional, PRNGKey, SmiEta, SummaryWriter,
                                      Tuple, Mapping)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)

def make_optimizer_eta(learning_rate: float) -> optax.GradientTransformation:
  optimizer = optax.adabelief(learning_rate=learning_rate)
  return optimizer


def q_distr_phi(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    hp: Array,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  # Define normalizing flows
  q_distr = getattr(flows_all_integrated, flow_name + '_phi')(**flow_kwargs)

  num_samples = hp.shape[0]

  # Sample from flows
  (phi_sample, phi_log_prob_posterior,
   phi_base_sample) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=[hp, None],
   )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows_all_integrated.split_flow_phi(
          samples=phi_sample,
          **flow_kwargs,
      ))

  # sample from base distribution that generated phi
  q_distr_out['phi_base_sample'] = phi_base_sample

  # log P(phi)
  q_distr_out['phi_log_prob'] = phi_log_prob_posterior

  return q_distr_out


def q_distr_theta(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    phi_base_sample: Array,
    hp: Array,
    is_aux: bool,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  num_samples = phi_base_sample.shape[0]

  # Define normalizing flows
  q_distr = getattr(flows_all_integrated, flow_name + '_theta')(**flow_kwargs)

  # Sample from flow
  (theta_sample, theta_log_prob_posterior) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
    #   context=[hp[:,2:], phi_base_sample],
      context=[hp, phi_base_sample],
  )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows_all_integrated.split_flow_theta(
          samples=theta_sample,
        #   is_aux=False,
          is_aux=is_aux,
          **flow_kwargs,
      ))

  # log P(theta|phi)
  q_distr_out['theta_' + ('aux_' if is_aux else '') +
              'log_prob'] = theta_log_prob_posterior

  return q_distr_out


def sample_all_flows(
    params_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    hp: Array,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  prng_seq = hk.PRNGSequence(prng_key)

  # phi
  q_distr_out = hk.transform(q_distr_phi).apply(
      params_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      hp=hp,
  )

#   hp_theta = hp.at[:,1].set(1.)

  # theta
  q_distr_out_theta = hk.transform(q_distr_theta).apply(
      params_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      phi_base_sample=q_distr_out['phi_base_sample'],
      hp=hp,
      is_aux=False,
  )
  q_distr_out['posterior_sample'].update(q_distr_out_theta['posterior_sample'])
  q_distr_out['theta_log_prob'] = q_distr_out_theta['theta_log_prob']

  q_distr_out_theta_aux = hk.transform(q_distr_theta).apply(
      params_tuple[2],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      phi_base_sample=q_distr_out['phi_base_sample'],
      hp=hp,
      is_aux=True,
  )
  q_distr_out['posterior_sample'].update(
      q_distr_out_theta_aux['posterior_sample'])
  q_distr_out['theta_aux_log_prob'] = q_distr_out_theta_aux[
      'theta_aux_log_prob']

  return q_distr_out


def elbo_estimate_along_eta(
    params_tuple: Tuple[hk.Params],
    batch: Batch,
    prng_key: PRNGKey,
    mask_Y: Array,
    mask_Z: Array,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta_sampling_a: float,
    eta_sampling_b: float,
    betahp_sampling_a: float,
    betahp_sampling_b: float,
    smi:bool,
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample beta concentration values
#   betahp_elbo = jax.random.gamma(
#       key=next(prng_seq),
#       a=betahp_sampling_a, 
#       shape=(num_samples, 2),
#   )/betahp_sampling_b
  betahp_elbo = jax.random.uniform(
        key=next(prng_seq),
        shape=(num_samples, 2),
        minval=betahp_sampling_a,
        maxval=betahp_sampling_b,
  )


  if smi:
    # Sample eta values (only for Y module)
    etas_elbo = jax.random.beta(
        key=next(prng_seq),
        a=eta_sampling_a,
        b=eta_sampling_b,
        shape=(num_samples, 1),
    )

    # Set eta_z=1

    etas_elbo_ext = jnp.concatenate([jnp.ones_like(etas_elbo), etas_elbo], axis=-1)
    hp_elbo = jnp.concatenate([etas_elbo_ext, betahp_elbo], axis=-1)
    cond_values = jnp.concatenate([etas_elbo, betahp_elbo], axis=-1)
  else:
    etas_elbo_ext = jnp.ones((num_samples, 2))
    hp_elbo = jnp.concatenate([etas_elbo_ext, betahp_elbo], axis=-1)
    cond_values = betahp_elbo

  
#   hp_elbo = jnp.ones(shape=(num_samples, 4))

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      prng_key=next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      hp=cond_values,
  )

  shared_params_names = [
      'phi',
  ]
  refit_params_names = [
      'theta',
  ]

  # ELBO stage 1: Power posterior
  posterior_sample_dict_stg1 = {}
  for key in shared_params_names:
    posterior_sample_dict_stg1[key] = q_distr_out['posterior_sample'][key]
  for key in refit_params_names:
    posterior_sample_dict_stg1[key] = q_distr_out['posterior_sample'][key +
                                                                      '_aux']

  log_prob_joint_stg1 = jax.vmap(
      lambda posterior_sample_i, hp_i: log_prob_fun_integrated.log_prob_joint(
          batch=batch,
          posterior_sample_dict=posterior_sample_i,
          mask_Z=mask_Z,
          mask_Y=mask_Y,
          hp=hp_i,
          is_smi=True,
      ))(
          jax.tree_map(lambda x: jnp.expand_dims(x, 1),
                       posterior_sample_dict_stg1),
          hp_elbo
      )

  log_q_stg1 = q_distr_out['phi_log_prob'] + q_distr_out['theta_aux_log_prob']

  #TODO: check reshape
  elbo_stg1 = log_prob_joint_stg1.reshape(-1) - log_q_stg1

  # ELBO stage 2: Refit theta
  posterior_sample_dict_stg2 = {}
  for key in shared_params_names:
    posterior_sample_dict_stg2[key] = jax.lax.stop_gradient(
        q_distr_out['posterior_sample'][key])
  for key in refit_params_names:
    posterior_sample_dict_stg2[key] = q_distr_out['posterior_sample'][key]

  log_prob_joint_stg2 = jax.vmap(
      lambda posterior_sample_i, hp_i: log_prob_fun_integrated.log_prob_joint(
          batch=batch,
          posterior_sample_dict=posterior_sample_i,
          mask_Z=mask_Z,
          mask_Y=mask_Y,
          hp=hp_i,
          is_smi=False,
      ))(
          jax.tree_map(lambda x: jnp.expand_dims(x, 1),
                       posterior_sample_dict_stg2),
        hp_elbo,                   
    )

  log_q_stg2 = (
      jax.lax.stop_gradient(q_distr_out['phi_log_prob']) +
      q_distr_out['theta_log_prob'])

  # TODO: check reshape
  elbo_stg2 = log_prob_joint_stg2.reshape(-1) - log_q_stg2

  elbo_dict = {'stage_1': elbo_stg1, 'stage_2': elbo_stg2}

  return elbo_dict


def loss(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_estimate_along_eta(
      params_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


########################################################################################################################
def compute_lpd(
    posterior_sample_dict: Dict[str, Any],
    batch: Batch,
    mask_neg_Z: Array,
    mask_neg_Y: Array,
) -> Mapping[str, Array]:
  """Compute ELPD.

  Estimates the ELPD based on two Monte Carlo approximations:
    1) Using WAIC.
  Masking is turned off.

  Args:
    posterior_sample_dict: Dictionary of posterior samples.
    batch: Batch of data (the one that was for training).

  Returns:
    Dictionary of ELPD estimates, with keys:
      - 'elpd_waic_pointwise': WAIC-based estimate.
  """

  # Initialize dictionary for output
  lpd_out = {}

  num_samples, phi_dim = posterior_sample_dict['phi'].shape # TOCHECK: is it correct?
  num_obs = batch['Z'].shape[0]

  ### WAIC ###
#   # Compute LPD

  loglik_pointwise_insample = log_prob_fun_integrated.log_lik_vectorised(
      mask_neg_Z,
      mask_neg_Y,
      batch['Z'],
      batch['Y'],
      batch['N'],
      batch['T'],
      posterior_sample_dict['phi'],
      posterior_sample_dict['theta'],
  ).sum(2)

  lpd_pointwise = jax.scipy.special.logsumexp(
      loglik_pointwise_insample, axis=0) - jnp.log(num_samples)
  lpd_out['lpd_pointwise'] = lpd_pointwise


  loglik_pointwise_y_insample = log_prob_fun_integrated.log_lik_y_vectorised(
      mask_neg_Y,
      batch['Y'],
      batch['T'],
      posterior_sample_dict['phi'],
      posterior_sample_dict['theta'],
  )
  lpd_pointwise_y = jax.scipy.special.logsumexp(
      loglik_pointwise_y_insample, axis=0) - jnp.log(num_samples)
  lpd_out['lpd_pointwise_y'] = lpd_pointwise_y

  loglik_pointwise_z_insample = log_prob_fun_integrated.log_lik_z_vectorised(
      mask_neg_Z,
      batch['Z'],
      batch['N'],
      posterior_sample_dict['phi'],
  )
  lpd_pointwise_z = jax.scipy.special.logsumexp(
      loglik_pointwise_z_insample, axis=0) - jnp.log(num_samples)
  lpd_out['lpd_pointwise_z'] = lpd_pointwise_z

  return lpd_out


compute_lpd_jit = jax.jit(compute_lpd)


def lpd_estimate_pointwise(
    hp_params: Array,
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    num_samples: int,
    mask_neg_Z:Array,
    mask_neg_Y:Array,
    # eta: Array,
    # betahp: Array,
):

#   assert len(hp_params) == 4
  q_distr_out_i = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=prng_key,  # same key to reduce variance of posterior along eta
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      hp=jnp.broadcast_to(hp_params, (num_samples, len(hp_params))) # num modules
  )
  lpd_dict = compute_lpd_jit(
      posterior_sample_dict=q_distr_out_i['posterior_sample'],
      batch=batch,
      mask_neg_Z=mask_neg_Z,
      mask_neg_Y=mask_neg_Y,
  )
  return lpd_dict

def pytrees_stack(pytrees, axis=0):
    results = jax.tree_map(
        lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results


def pytrees_vmap(fn):
    def g(pytrees, *args):
        stacked = pytrees_stack(pytrees)
        results = jax.vmap(fn)(stacked, *args)
        return results
    return g

def elpd_loocv_estimate(
      hp_params: Array,
      prng_key: PRNGKey,
      states_lists: List[List[TrainState]],
      batch_predict: Batch,
      flow_name: str,
      flow_kwargs: Dict[str, Any],
      num_samples: int,
):
    n_obs = batch_predict['Z'].shape[0]
    all_mask_neg_Z = jnp.eye(n_obs)
    all_mask_neg_Y = jnp.eye(n_obs)
    elpd_loocv_fixed_hp_params = pytrees_vmap(lambda state_list, key, mask_neg_Z, mask_neg_Y:  lpd_estimate_pointwise(
        hp_params=hp_params,
        state_list=state_list,
        batch=batch_predict,
        prng_key=key,
        flow_name=flow_name,
        flow_kwargs=flow_kwargs,
        num_samples=num_samples,
        mask_neg_Z=mask_neg_Z,
        mask_neg_Y=mask_neg_Y,
        ))
    
    elpd_loocv_fixed_hp_params_jit = jax.jit(elpd_loocv_fixed_hp_params)

    n_obs = batch_predict['Z'].shape[0]
    keys = jax.random.split(
      prng_key, n_obs)


    return elpd_loocv_fixed_hp_params_jit(states_lists, keys, all_mask_neg_Z, all_mask_neg_Y)


########################################################################################################################

def compute_elpd(
    posterior_sample_dict: Dict[str, Any],
    batch: Batch,
) -> Mapping[str, Array]:
  """Compute ELPD.

  Estimates the ELPD based on two Monte Carlo approximations:
    1) Using WAIC.
  
  Masking is turned off.

  Args:
    posterior_sample_dict: Dictionary of posterior samples.
    batch: Batch of data (the one that was for training).

  Returns:
    Dictionary of ELPD estimates, with keys:
      - 'elpd_waic_pointwise': WAIC-based estimate.
  """

  # Initialize dictionary for output
  elpd_out = {}

  num_samples, phi_dim = posterior_sample_dict['phi'].shape # TOCHECK: is it correct?

  ### WAIC ###
#   # Compute LPD

  loglik_pointwise_insample = log_prob_fun_integrated.log_lik_vectorised(
      jnp.array([1]*phi_dim),
      jnp.array([1]*phi_dim),
      batch['Z'],
      batch['Y'],
      batch['N'],
      batch['T'],
      posterior_sample_dict['phi'],
      posterior_sample_dict['theta'],
  ).sum(2)

  lpd_pointwise = jax.scipy.special.logsumexp(
      loglik_pointwise_insample, axis=0) - jnp.log(num_samples)
  elpd_out['lpd_pointwise'] = lpd_pointwise

  p_waic_pointwise = jnp.var(loglik_pointwise_insample, axis=0)
  elpd_out['p_waic_pointwise'] = p_waic_pointwise

  elpd_waic_pointwise = lpd_pointwise - p_waic_pointwise
  elpd_out['elpd_waic_pointwise'] = elpd_waic_pointwise


  loglik_pointwise_y_insample = log_prob_fun_integrated.log_lik_y_vectorised(
      jnp.array([1]*phi_dim),
      batch['Y'],
      batch['T'],
      posterior_sample_dict['phi'],
      posterior_sample_dict['theta'],
  )
  lpd_pointwise_y = jax.scipy.special.logsumexp(
      loglik_pointwise_y_insample, axis=0) - jnp.log(num_samples)
  elpd_out['lpd_pointwise_y'] = lpd_pointwise_y

  p_waic_pointwise_y = jnp.var(loglik_pointwise_y_insample, axis=0)
  elpd_out['p_waic_pointwise_y'] = p_waic_pointwise_y

  elpd_waic_pointwise_y = lpd_pointwise_y - p_waic_pointwise_y
  elpd_out['elpd_waic_pointwise_y'] = elpd_waic_pointwise_y


  loglik_pointwise_z_insample = log_prob_fun_integrated.log_lik_z_vectorised(
      jnp.array([1]*phi_dim),
      batch['Z'],
      batch['N'],
      posterior_sample_dict['phi'],
  )
  lpd_pointwise_z = jax.scipy.special.logsumexp(
      loglik_pointwise_z_insample, axis=0) - jnp.log(num_samples)
  elpd_out['lpd_pointwise_z'] = lpd_pointwise_z

  p_waic_pointwise_z = jnp.var(loglik_pointwise_z_insample, axis=0)
  elpd_out['p_waic_pointwise_z'] = p_waic_pointwise_z

  elpd_waic_pointwise_z = lpd_pointwise_z - p_waic_pointwise_z
  elpd_out['elpd_waic_pointwise_z'] = elpd_waic_pointwise_z

  return elpd_out


compute_elpd_jit = jax.jit(compute_elpd)


def elpd_estimate_pointwise(
    hp_params: Array,
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    num_samples: int,
    # eta: Array,
    # betahp: Array,
):

#   assert len(hp_params) == 4
  q_distr_out_i = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=prng_key,  # same key to reduce variance of posterior along eta
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      hp=jnp.broadcast_to(hp_params, (num_samples, len(hp_params))) # num modules
  )
  elpd_dict = compute_elpd_jit(
      posterior_sample_dict=q_distr_out_i['posterior_sample'],
      batch=batch,
  )
  return elpd_dict

def elpd_surface_points(
    state_list: List[TrainState],
    batch: Batch, # train_ds
    prng_key: PRNGKey,
    config: ConfigDict,
    eta_grid: Array,
    use_vmap: bool = True,
):
  """Visualize ELPD surface as function of eta."""

  assert eta_grid.ndim == 3

  num_groups, *grid_shape = eta_grid.shape

  # TODO: vmap implementation produces RuntimeError: RESOURCE_EXHAUSTED
  lpd_pointwise_all_eta = []
  p_waic_pointwise_all_eta = []
  elpd_waic_pointwise_all_eta = []

  lpd_pointwise_y_all_eta = []
  p_waic_pointwise_y_all_eta = []
  elpd_waic_pointwise_y_all_eta = []

  lpd_pointwise_z_all_eta = []
  p_waic_pointwise_z_all_eta = []
  elpd_waic_pointwise_z_all_eta = []

  if use_vmap:
    # Faster option: using vmap
    # Sometimes fail due to RuntimeError: RESOURCE_EXHAUSTED
    eta_grid_reshaped = eta_grid.reshape(num_groups, -1).T

    elpd_dict_all = jax.vmap(lambda hp_params_i: elpd_estimate_pointwise(
        hp_params=hp_params_i,
        state_list=state_list,
        batch=batch,
        prng_key=prng_key,
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        num_samples=config.num_samples_elpd,
        # eta=eta_i,
        # betahp=betahp_i,
    ))(eta_grid_reshaped)

    print('vmap done')
        

    lpd_pointwise_all_eta = elpd_dict_all['lpd_pointwise']
    p_waic_pointwise_all_eta = elpd_dict_all['p_waic_pointwise']
    elpd_waic_pointwise_all_eta = elpd_dict_all['elpd_waic_pointwise']

    lpd_pointwise_y_all_eta = elpd_dict_all['lpd_pointwise_y']
    p_waic_pointwise_y_all_eta = elpd_dict_all['p_waic_pointwise_y']
    elpd_waic_pointwise_y_all_eta = elpd_dict_all['elpd_waic_pointwise_y']

    lpd_pointwise_z_all_eta = elpd_dict_all['lpd_pointwise_z']
    p_waic_pointwise_z_all_eta = elpd_dict_all['p_waic_pointwise_z']
    elpd_waic_pointwise_z_all_eta = elpd_dict_all['elpd_waic_pointwise_z']
  else:
    # Slower option: for loop
    # Takes longer to compute
    eta_grid_reshaped = eta_grid.reshape(num_groups, -1).T
    for hp_params_i in zip(eta_grid_reshaped):
      # eta_i = (eta_grid.reshape(num_groups, -1).T)[0]

      elpd_dict_i = elpd_estimate_pointwise(
          hp_params=hp_params_i,
          state_list=state_list,
          batch=batch,
          prng_key=prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          num_samples=config.num_samples_elpd,
          # eta=eta_i,
          # betahp=betahp_i,
      )
      lpd_pointwise_all_eta.append(elpd_dict_i['lpd_pointwise'])
      p_waic_pointwise_all_eta.append(elpd_dict_i['p_waic_pointwise'])
      elpd_waic_pointwise_all_eta.append(elpd_dict_i['elpd_waic_pointwise'])

      lpd_pointwise_y_all_eta.append(elpd_dict_i['lpd_pointwise_y'])
      p_waic_pointwise_y_all_eta.append(elpd_dict_i['p_waic_pointwise_y'])
      elpd_waic_pointwise_y_all_eta.append(elpd_dict_i['elpd_waic_pointwise_y'])

      lpd_pointwise_z_all_eta.append(elpd_dict_i['lpd_pointwise_z'])
      p_waic_pointwise_z_all_eta.append(elpd_dict_i['p_waic_pointwise_z'])
      elpd_waic_pointwise_z_all_eta.append(elpd_dict_i['elpd_waic_pointwise_z'])

    lpd_pointwise_all_eta = jnp.stack(lpd_pointwise_all_eta, axis=0)
    p_waic_pointwise_all_eta = jnp.stack(p_waic_pointwise_all_eta, axis=0)
    elpd_waic_pointwise_all_eta = jnp.stack(elpd_waic_pointwise_all_eta, axis=0)

    lpd_pointwise_y_all_eta = jnp.stack(lpd_pointwise_y_all_eta, axis=0)
    p_waic_pointwise_y_all_eta = jnp.stack(p_waic_pointwise_y_all_eta, axis=0)
    elpd_waic_pointwise_y_all_eta = jnp.stack(elpd_waic_pointwise_y_all_eta, axis=0)

    lpd_pointwise_z_all_eta = jnp.stack(lpd_pointwise_z_all_eta, axis=0)
    p_waic_pointwise_z_all_eta = jnp.stack(p_waic_pointwise_z_all_eta, axis=0)
    elpd_waic_pointwise_z_all_eta = jnp.stack(elpd_waic_pointwise_z_all_eta, axis=0)

  # Add pointwise elpd and lpd across observations
  lpd_all_eta = lpd_pointwise_all_eta.sum(axis=-1).reshape(grid_shape)
  p_waic_all_eta = p_waic_pointwise_all_eta.sum(axis=-1).reshape(grid_shape)
  elpd_waic_all_eta = elpd_waic_pointwise_all_eta.sum(
      axis=-1).reshape(grid_shape)

  lpd_y_all_eta = lpd_pointwise_y_all_eta.sum(axis=-1).reshape(grid_shape)
  p_waic_y_all_eta = p_waic_pointwise_y_all_eta.sum(axis=-1).reshape(grid_shape)
  elpd_waic_y_all_eta = elpd_waic_pointwise_y_all_eta.sum(
      axis=-1).reshape(grid_shape)

  lpd_z_all_eta = lpd_pointwise_z_all_eta.sum(axis=-1).reshape(grid_shape)
  p_waic_z_all_eta = p_waic_pointwise_z_all_eta.sum(axis=-1).reshape(grid_shape)
  elpd_waic_z_all_eta = elpd_waic_pointwise_z_all_eta.sum(
      axis=-1).reshape(grid_shape)

  elpd_surface_dict = {'lpd_all_eta':lpd_all_eta,'p_waic_all_eta':p_waic_all_eta,
  'elpd_waic_all_eta':elpd_waic_all_eta, 'lpd_y_all_eta':lpd_y_all_eta,
  'p_waic_y_all_eta':p_waic_y_all_eta, 'elpd_waic_y_all_eta':elpd_waic_y_all_eta,
  'lpd_z_all_eta':lpd_z_all_eta, 'p_waic_z_all_eta':p_waic_z_all_eta,
  'elpd_waic_z_all_eta':elpd_waic_z_all_eta}

  return elpd_surface_dict


def log_images(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    config: ConfigDict,
    show_elpd: bool,
    eta_grid_len: int,
    summary_writer: Optional[SummaryWriter],
    workdir_png: Optional[str],
    show_posterior_range_allhps:Optional[bool] = False,
    show_posterior_range_priorhpsVMP_etafixed:Optional[bool] = False,
    workdir_mcmc:Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  # Plot posterior samples
  if show_posterior_range_priorhpsVMP_etafixed:
    images = []
    priorhps = {'priorhp_low_bayes': [0.1,0.1],#[0.1,0.1],
                'priorhp_high_bayes':[11.82, 15],#[10.33, 15],
                'priorhp_converged_bayes':[1.71, 15],#[1.70, 15]
                'priorhp_mixed_bayes':[0.1, 3.],#[0.1, 3.]
                 'priorhp_ones_bayes':[1.,1.] }#[1.,1.]

    smi_etas = {'eta_bayes':1.} #[[1., 1.], [1., 0.0001]]
    posterior_sample_dict = {'eta_bayes':{}}
    posterior_sample_dfs_theta = {'eta_bayes':{}}
    posterior_sample_dfs_theta_aux = {'eta_bayes':{}}

    # produce necessary posterior samples
    for p_ix, (priorhp_k, priorhp_v) in enumerate(priorhps.items()):
        q_distr_out = sample_all_flows(
            params_tuple=[state.params for state in state_list],
            prng_key= next(prng_seq),
            flow_name=config.flow_name,
            flow_kwargs=config.flow_kwargs,
            hp=jnp.broadcast_to(jnp.array(priorhps[priorhp_k]),
                            (config.num_samples_plot,) + (len(priorhps[priorhp_k]),)),

        )
        posterior_sample_dict['eta_bayes'][priorhp_k] = q_distr_out['posterior_sample']
        theta = posterior_sample_dict['eta_bayes'][priorhp_k]['theta']
        theta_aux = posterior_sample_dict['eta_bayes'][priorhp_k]['theta_aux']

        _, theta_dim = theta.shape
        posterior_samples_df_theta = pd.DataFrame(
            theta, columns=[f"theta_{i}" for i in range(1, theta_dim + 1)])
        posterior_samples_df_theta['eta1'] = f'= 1'
        posterior_sample_dfs_theta['eta_bayes'][priorhp_k] = posterior_samples_df_theta

        posterior_samples_df_theta_aux = pd.DataFrame(
            theta_aux, columns=[f"theta_{i}" for i in range(1, theta_dim + 1)])
        posterior_samples_df_theta_aux['eta1'] = f'= 1'
        posterior_sample_dfs_theta_aux['eta_bayes'][priorhp_k] = posterior_samples_df_theta_aux
    
    # produce phi plots
    priorhp_main = {'main': {'eta_bayes': 'priorhp_converged_bayes'},
        'secondary': {'eta_bayes': 'priorhp_high_bayes'}}
    # fig_phi = plot_all.plot_posterior_phi_hprange(
    #     posterior_sample_dict=posterior_sample_dict,
    #     eta = ('eta_bayes',1.),
    #     priorhps = priorhps,
    #     priorhp_main = priorhp_main,
    #     plot_two=config.plot_two,
    # )
    # if workdir_png:
    #     fig_phi.savefig(pathlib.Path(workdir_png) / (f'epidemiology_phi_hprange_onlypriorhpVMP_etafixed1' + ".png"))
    # if summary_writer:
    #     images.append(plot_to_image(fig_phi))


    # # produce theta plots
    # fig_theta = plot_all.plot_posterior_theta_hprange_single_eta(
    #       posterior_sample_dfs=posterior_sample_dfs_theta,
    #       eta = ('eta_bayes',1.),
    #       priorhps = priorhps,
    #       priorhp_main = priorhp_main,
    # )
    # if workdir_png:
    #     fig_theta.savefig(pathlib.Path(workdir_png) / (f"epidemiology_theta_hprange_onlypriorhpVMP_etafixed1_optVShigh" + ".png"))
    # if summary_writer:
    #     images.append(plot_to_image(fig_theta))

    # produce theta plots at hp = 1,1
    priorhp_main = {'main': {'eta_bayes': 'priorhp_converged_bayes'},
    'secondary': {'eta_bayes': 'priorhp_low_bayes'}}
    fig_theta = plot_all.plot_posterior_theta_hprange_single_eta(
          posterior_sample_dfs=posterior_sample_dfs_theta,
          eta = ('eta_bayes',1.),
          priorhps = priorhps,
          priorhp_main = priorhp_main,
    )

    if workdir_png:
        fig_theta.savefig(pathlib.Path(workdir_png) / (f"epidemiology_theta_hprange_onlypriorhpVMP_etafixed1_optVSlow" + ".png"))
    if summary_writer:
        images.append(plot_to_image(fig_theta))

    if workdir_mcmc:
        mcmc_dirs = {'priorhp_low_bayes': '/mcmc_eta_1.00_c1_0.10_c2_0.10/mcmc_eta_1.00_c1_0.10_c2_0.10.sav', 
                'priorhp_high_bayes':'/mcmc_eta_1.00_c1_11.82_c2_15.00/mcmc_eta_1.00_c1_11.82_c2_15.00.sav', #mcmc_theta_eta1.0_c1_10.33_c2_15.sav',
                'priorhp_converged_bayes':f'/mcmc_eta_1.00_c1_1.71_c2_15.00/mcmc_eta_1.00_c1_1.71_c2_15.00.sav', #mcmc_theta_eta1.0_c1_1.7_c2_15.sav',
                 'priorhp_mixed_bayes':f'/mcmc_eta_1.00_c1_0.10_c2_3.00/mcmc_eta_1.00_c1_0.10_c2_3.00.sav', #mcmc_theta_eta1.0_c1_0.1_c2_3.sav',
                  'priorhp_ones_bayes':f'/mcmc_eta_1.00_c1_1.00_c2_1.00/mcmc_eta_1.00_c1_1.00_c2_1.00.sav'}
        for priorhp_k, priorhp_v in priorhps.items():
        # produce theta plots at hp opt VS MCMC
            with open(workdir_mcmc + mcmc_dirs[priorhp_k], 'rb') as fr:
                mcmc_theta_eta1 = pickle.load(fr)
            fig_theta_mcmc, wass = plot_all.plot_posterior_theta_vsmcmc_single_eta(
                posterior_sample_df_main=posterior_sample_dfs_theta['eta_bayes'][priorhp_k],
                mcmc_kde=mcmc_theta_eta1['theta'],
            )

            if workdir_png:
                fig_theta_mcmc.savefig(pathlib.Path(workdir_png) / (f"Epidemiology_theta_vsMCMC_eta1.0_{priorhp_k}_w_{wass:.3f}" + ".png"))
            if summary_writer:
                images.append(plot_to_image(fig_theta_mcmc))

        ############################################################################################################################
  if show_posterior_range_allhps:
    images = []
    priorhps = {'priorhp_converged_cut':[0.97,14.],#[0.41, 15.],#[1.15, 15.] 
                'priorhp_ones': [1.,1.],
                'priorhp_alternative_bayes':[0.1, 0.1],#[0.1, 0.1],
                'priorhp_alternative_cut':[5., 0.1],#[[0.1, 0.1], 
                'priorhp_converged_bayes':[13.04,15.],#[11.87,15.]}#[10.76,15.]}
                }
    priorhp_main = {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
        'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
    smi_etas = {'eta_bayes':0.87, 'eta_cut':0.02} #[[1., 1.], [1., 0.0001]]
    posterior_sample_dict = {'eta_bayes':{}, 'eta_cut':{}}
    posterior_sample_dfs_theta = {'eta_bayes':{}, 'eta_cut':{}}
    posterior_sample_dfs_theta_aux = {'eta_bayes':{}, 'eta_cut':{}}

    # produce necessary posterior samples
    for eta_ix, (eta_k, eta_v) in enumerate(smi_etas.items()):
        for p_ix, (priorhp_k, priorhp_v) in enumerate(priorhps.items()):
            q_distr_out = sample_all_flows(
                params_tuple=[state.params for state in state_list],
                prng_key= next(prng_seq),
                flow_name=config.flow_name,
                flow_kwargs=config.flow_kwargs,
                hp=jnp.broadcast_to(jnp.concatenate([jnp.array(smi_etas[eta_k])[None], jnp.array(priorhps[priorhp_k])], axis=-1),
                             (config.num_samples_plot,) + (1+len(jnp.array(priorhps[priorhp_k])),)),

            )
            posterior_sample_dict[eta_k][priorhp_k] = q_distr_out['posterior_sample']
            theta = posterior_sample_dict[eta_k][priorhp_k]['theta']
            theta_aux = posterior_sample_dict[eta_k][priorhp_k]['theta_aux']

            _, theta_dim = theta.shape
            posterior_samples_df_theta = pd.DataFrame(
                theta, columns=[f"theta_{i}" for i in range(1, theta_dim + 1)])
            posterior_samples_df_theta['eta1'] = f'= {0.87 if eta_v==0.87 else 0.02}'
            posterior_sample_dfs_theta[eta_k][priorhp_k] = posterior_samples_df_theta
    
            posterior_samples_df_theta_aux = pd.DataFrame(
                theta_aux, columns=[f"theta_{i}" for i in range(1, theta_dim + 1)])
            posterior_samples_df_theta_aux['eta1'] = f'= {0.87 if eta_v==0.87 else 0.02}'
            posterior_sample_dfs_theta_aux[eta_k][priorhp_k] = posterior_samples_df_theta_aux
    
    # produce phi vs loglambda plots
    # fig_theta_loglambda = plot_all.plot_phi_loglambda(
    #     posterior_sample_dict=posterior_sample_dict,
    #     priorhps=priorhps,
    #     smi_etas=smi_etas,
    #     priorhp_toplot={'eta_bayes':['priorhp_ones', 'priorhp_converged_bayes'],
    #                                         'eta_cut':['priorhp_ones', 'priorhp_converged_cut']},
    #     # xlim=[0, 0.3],
    #     # ylim=[-2.5, 2],
    #     )
    # if workdir_png:
    #     fig_theta_loglambda.savefig(pathlib.Path(workdir_png) / ( f"epidemiology_phi_vs_loglambda_eta{smi_etas[eta_k]}" + ".png"))
    # if summary_writer:
    #     images.append(plot_to_image(fig_theta_loglambda))

    # produce phi plots
    with open(workdir_mcmc + f'/mcmc_eta_0.87_c1_13.04_c2_15.00/mcmc_eta_0.87_c1_13.04_c2_15.00.sav', 'rb') as fr:
        mcmc_phis_eta1 = pickle.load(fr)['phi']
    with open(workdir_mcmc + f'/mcmc_eta_0.02_c1_0.97_c2_14.00/mcmc_eta_0.02_c1_0.97_c2_14.00.sav', 'rb') as fr:
        mcmc_phis_eta0 = pickle.load(fr)['phi']
    posterior_mcmc_dict = {'eta_bayes':mcmc_phis_eta1, 'eta_cut':mcmc_phis_eta0}
    for eta_ix, (eta_k, eta_v) in enumerate(smi_etas.items()):
        fig_phi = plot_all.plot_posterior_phi_hprange(
          plot_two=config.plot_two,
          posterior_sample_dict=posterior_sample_dict,
          mcmc_posterior_phis=posterior_mcmc_dict,
          eta = (eta_k,eta_v),
          priorhps = priorhps,
          priorhp_main = priorhp_main,
        )
        if workdir_png:
            fig_phi.savefig(pathlib.Path(workdir_png) / (f'epidemiology_phi_hprange_eta{smi_etas[eta_k]}_{"two" if config.plot_two else ""}' + ".png"))
        if summary_writer:
            images.append(plot_to_image(fig_phi))
    #     phi_no = posterior_sample_dict[eta_k][priorhp_k]['phi'].shape[1] 
    #     for phi_ix in np.arange(phi_no):
    #         fig_phi_single = plot_all.plot_posterior_phi_hprange_singlephi(
    #             posterior_sample_dict=posterior_sample_dict,
    #             phi_ix=phi_ix,
    #             eta = (eta_k,eta_v),
    #             priorhps = priorhps,
    #             priorhp_main = priorhp_main,
    #             )
    #         if workdir_png:
    #            fig_phi_single.savefig(pathlib.Path(workdir_png) / (f'epidemiology_phi_hprange_eta{smi_etas[eta_k]}_phi{phi_ix+1}' + ".png"))
    #         if summary_writer:
    #            images.append(plot_to_image(fig_phi_single))
           
           

    # # produce theta aux plots
    # fig_theta_aux = plot_all.plot_posterior_theta_hprange(
    #       posterior_sample_dfs=posterior_sample_dfs_theta_aux,
    #       smi_etas = smi_etas,
    #       priorhps = priorhps,
    #       priorhp_main = priorhp_main,
    # )
    # if workdir_png:
    #     fig_theta_aux.savefig(pathlib.Path(workdir_png) / (f"epidemiology_theta_aux_hprange_eta{smi_etas['eta_cut']}" + ".png"))
    # if summary_writer:
    #     images.append(plot_to_image(fig_theta_aux))

    # # produce theta plots
    fig_theta = plot_all.plot_posterior_theta_hprange(
          posterior_sample_dfs=posterior_sample_dfs_theta,
          smi_etas = smi_etas,
          priorhps = priorhps,
          priorhp_main = priorhp_main,
    )
    if workdir_png:
        fig_theta.savefig(pathlib.Path(workdir_png) / (f"epidemiology_theta_hprange_eta{smi_etas['eta_cut']}" + ".png"))
    if summary_writer:
        images.append(plot_to_image(fig_theta))

    # # # produce phi priors plot
    # # fig_betas = plot_all.plot_hprange_betapriors(
    # #   priorhps = priorhps,
    # #   priorhp_main = priorhp_main,)
    # # if workdir_png:
    # #     fig_betas.savefig(pathlib.Path(workdir_png) / (f"epidemiology_betaprior_eta{smi_etas['eta_cut']}" + ".png"))
    # # if summary_writer:
    # #     images.append(plot_to_image(fig_betas))

    # # produce theta plots at hp = 1,1
    # priorhp_main = {'main': {'eta_bayes': 'priorhp_ones',
    #         'eta_cut': 'priorhp_ones'},
    # 'secondary': {'eta_bayes': 'priorhp_converged_bayes',
    #         'eta_cut': 'priorhp_converged_cut'}}
    # fig_theta = plot_all.plot_posterior_theta_hprange(
    #       posterior_sample_dfs=posterior_sample_dfs_theta,
    #       smi_etas = smi_etas,
    #       priorhps = priorhps,
    #       priorhp_main = priorhp_main,
    # )

    # if workdir_png:
    #     fig_theta.savefig(pathlib.Path(workdir_png) / (f"epidemiology_theta_hprange_eta{smi_etas['eta_cut']}_hp1" + ".png"))
    # if summary_writer:
    #     images.append(plot_to_image(fig_theta))

    if workdir_mcmc:
        # produce theta plots at hp opt VS MCMC
        # with open(workdir_mcmc + f'/eta_1.000/mcmc_theta_eta1.0_c1_10.76_c2_15.sav', 'rb') as fr:
        # with open(workdir_mcmc + f'/eta_1.000/mcmc_theta_eta1.0_c1_11.87_c2_15.sav', 'rb') as fr:
        with open(workdir_mcmc + f'/mcmc_eta_0.87_c1_13.04_c2_15.00/mcmc_eta_0.87_c1_13.04_c2_15.00.sav', 'rb') as fr:
            mcmc_theta_eta1 = pickle.load(fr)
        mcmc_theta_eta1 = pd.DataFrame(mcmc_theta_eta1['theta'], columns=['theta_1', 'theta_2'])
        mcmc_theta_eta1['eta1'] = '= 0.87'#'= 1'
        # with open(workdir_mcmc + f'/eta_0.0001/mcmc_theta_eta0.0001_c1_1.15_c2_15.sav', 'rb') as fr:
        # with open(workdir_mcmc + f'/eta_0.007/mcmc_theta_eta0.007_c1_0.41_c2_15.sav', 'rb') as fr:
        with open(workdir_mcmc + f'/mcmc_eta_0.02_c1_0.97_c2_14.00/mcmc_eta_0.02_c1_0.97_c2_14.00.sav', 'rb') as fr:
            mcmc_theta_eta0 = pickle.load(fr)
        mcmc_theta_eta0 = pd.DataFrame(mcmc_theta_eta0['theta'], columns=['theta_1', 'theta_2'])
        mcmc_theta_eta0['eta1'] = '= 0.02'
        mcmc_main = pd.concat([mcmc_theta_eta1,
                             mcmc_theta_eta0])
        
        priorhp_main = {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
                'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}

        fig_theta_mcmc, wass = plot_all.plot_posterior_theta_hprange_vsmcmc_SMI(
            posterior_sample_dfs=posterior_sample_dfs_theta,
            mcmc_df=mcmc_main,
            smi_etas = smi_etas,
            priorhps = priorhps,
            priorhp_main = priorhp_main,
        )

        if workdir_png:
            fig_theta_mcmc.savefig(pathlib.Path(workdir_png) / (f"Theta_vsMCMC_hprange_eta{smi_etas['eta_cut']}_hpOPT_weta1_{wass['w1']:.3f}__weta0_{wass['w0']:.3f}" + ".png"))
        if summary_writer:
            images.append(plot_to_image(fig_theta_mcmc))

        ##############################################################

    # produce phi plots at hp = 1,1
    fig_phi = plot_all.plot_posterior_phi_etarange(
        posterior_sample_dict=posterior_sample_dict,
        smi_etas = smi_etas,
        priorhp = ('priorhp_ones', priorhps['priorhp_ones'])
    )
    if workdir_png:
        fig_phi.savefig(pathlib.Path(workdir_png) / (f"epidemiology_phi_etarange_c1_{priorhps['priorhp_ones'][0]}_c2_{priorhps['priorhp_ones'][1]}" + ".png"))
    if summary_writer:
        images.append(plot_to_image(fig_phi))

    # produce phi priors plot at hp = 1,1
    fig_betas = plot_all.plot_hprange_betapriors(
      priorhps = priorhps,
      priorhp_main = priorhp_main,)
    if workdir_png:
        fig_betas.savefig(pathlib.Path(workdir_png) / (f"epidemiology_betaprior_eta{smi_etas['eta_cut']}_hp1" + ".png"))
    if summary_writer:
        images.append(plot_to_image(fig_betas))

    
  ### ELPD ###

  # Define elements to grate grid of eta values

  eta_base = np.hstack([np.array([1., 0.]), np.array(config.priorhp_default_elpdplot)])
  eta_grid_base = np.tile(eta_base, [eta_grid_len + 1, eta_grid_len + 1, 1]).T


  if show_elpd:

    images = []
    prng_key_elpd = next(prng_seq)
    elpd_ranges = {'range_eta1': np.linspace(0., 1., eta_grid_len + 1),
                   'range_c1':np.linspace(0., 20., eta_grid_len + 1),
                   'range_c2':np.linspace(0., 20., eta_grid_len + 1),
                    'fixed_one':np.linspace(1., 1., eta_grid_len + 1),
                    'fixed_zero':np.linspace(0., 0., eta_grid_len + 1)}

    # ELPD one variable at a time     
    vary_one = {'names':[('eta1', f'Eta1, c1 hp fixed to {config.priorhp_default_elpdplot[0]} and c2 hp fixed to {config.priorhp_default_elpdplot[1]}'),
                         ('c1', f'Beta prior c1 hyperparameter, \n eta1 fixed to 0 and c2 hp fixed to {config.priorhp_default_elpdplot[1]}'),
                         ('c2',f'Beta prior c2 hyperparameter, \n eta1 fixed to 0 and c1 hp fixed to {config.priorhp_default_elpdplot[0]}'),
                         ('c1', f'Beta prior c1 hyperparameter, \n eta1 fixed to 1 and c2 hp fixed to {config.priorhp_default_elpdplot[1]}'),
                         ('c2',f'Beta prior c2 hyperparameter, \n eta1 fixed to 1 and c1 hp fixed to {config.priorhp_default_elpdplot[0]}')],
                'vals':[((0,1,2),('range_eta1', 'fixed_one')), 
                ((1,1,2),('fixed_zero', 'range_c1')),
                ((1,1,3),('fixed_zero', 'range_c2')),
                ((1,1,2),('fixed_one', 'range_c1')),
                ((1,1,3),('fixed_one', 'range_c2'))]}
    
    for v_ix, (indxs, range_keys) in enumerate(vary_one['vals']):
        eta_grid_mini = np.stack(
        np.meshgrid(
            elpd_ranges[range_keys[0]],
            elpd_ranges[range_keys[1]]),
        axis=0)
        eta_grid = eta_grid_base.copy()
        eta_grid_x_y_idx = [indxs[1], indxs[2]]
        eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

        elpd_surface_dict = elpd_surface_points(
            state_list=state_list,
            batch=batch, # train_ds
            prng_key=prng_key_elpd,
            config=config,
            eta_grid=eta_grid,
        )
        
        xlab = vary_one['names'][v_ix][0]
        fig, axs = plot_all.plot_elpd_one_variable(elpd_surface_dict=elpd_surface_dict,
            suptitle=vary_one['names'][v_ix][1],
            xlabel=xlab,
            x_values=eta_grid[eta_grid_x_y_idx[1]][:,0]if indxs[0]==1 else eta_grid[eta_grid_x_y_idx[0]][0],
            indx=indxs[0],
            is_long=False)

        if workdir_png:
            if not os.path.exists(pathlib.Path(workdir_png+'/elpd_plots')):
               os.makedirs(pathlib.Path(workdir_png +'/elpd_plots'))
            fig.savefig(pathlib.Path(workdir_png) / (f'elpd_plots/elpd_{xlab}' 
                                                     + (('_eta_at_1' if range_keys[0] == 'fixed_one' else '_eta_at_0') if range_keys[0]!='range_eta1' else "") 
                                                     + (f'_c1_at_{config.priorhp_default_elpdplot[0]}' if xlab!="c1" else "") 
                                                     + (f'_c2_at_{config.priorhp_default_elpdplot[1]}' if xlab!="c2" else "") 
                                                     + ".png"))
        if summary_writer:
            images.append(plot_to_image(fig))
       
    # ELPD surface keeping third variable fixed
    vary_two = {'names':[('eta1', 'c1', f'Vary eta1 and c1, c2 fixed at {config.priorhp_default_elpdplot[1]}'),
                        ('eta1', 'c2', f'Vary eta1 and c2, c1 fixed at {config.priorhp_default_elpdplot[0]}'),
                        ('c1', 'c2', f'Vary c1 and c2, eta1 fixed at 1'),
                        ('c1', 'c2', f'Vary c1 and c2, eta1 fixed at 0'),],
            'vals':[((1,2),('range_eta1', 'range_c1', 'fixed_one')), 
            ((1,3),('range_eta1', 'fixed_one', 'range_c2')), 
            ((2,3),('fixed_one', 'range_c1', 'range_c2')), 
            ((2,3),('fixed_zero', 'range_c1', 'range_c2'))]}

    for v_ix, (indxs, range_keys) in enumerate(vary_two['vals']):
        eta_grid = eta_grid_base.copy()
        if range_keys[0]=='fixed_one':
           eta_grid[1, :, :] = np.tile(1., [eta_grid_len + 1, eta_grid_len + 1, 1]).T
        
        eta_grid_mini = np.stack(
        np.meshgrid(
            elpd_ranges[range_keys[indxs[0]-1]],
            elpd_ranges[range_keys[indxs[1]-1]]),
        axis=0)
        
        eta_grid_x_y_idx = [indxs[0], indxs[1]]
        eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

        elpd_surface_dict = elpd_surface_points(
            state_list=state_list,
            batch=batch, # train_ds
            prng_key=prng_key_elpd,
            config=config,
            eta_grid=eta_grid,
        )

        xlab=vary_two['names'][v_ix][0]
        ylab=vary_two['names'][v_ix][1]

        # Plot the ELPD surface.
        fig_elpd_surface = plot_all.plot_elpd_surface(
            elpd_surface_dict=elpd_surface_dict,
            eta_grid=eta_grid,
            eta_grid_x_y_idx=[indxs[0], indxs[1]],
            xlab=xlab,
            ylab=ylab,
            suptitle=vary_two['names'][v_ix][2],
        )
        
        if workdir_png:
            if not os.path.exists(pathlib.Path(workdir_png+'/elpd_plots')):
               os.makedirs(pathlib.Path(workdir_png +'/elpd_plots'))
            fig_elpd_surface.savefig(pathlib.Path(workdir_png) / (f'elpd_plots/elpd_{xlab}_{ylab}' 
                                                                  + (('_eta_at_0' if 'fixed_zero' in range_keys else '_eta_at_1') if ((xlab!="eta1")&(ylab!="eta1")) else "")
                                                                  + (f'_c1_at_{config.priorhp_default_elpdplot[0]}' if ((xlab!="c1")&(ylab!="c1")) else "") 
                                                                  + (f'_c2_at_{config.priorhp_default_elpdplot[1]}' if ((xlab!="c2")&(ylab!="c2")) else "") 
                                                                  + ".png"))
        if summary_writer:
            images.append(plot_to_image(fig_elpd_surface))


########################################################################################################################

def train_and_evaluate(config: ConfigDict, workdir: str, workdir_mcmc: Optional[str]=False) -> TrainState:
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
  # Small data, no need to batch
  train_ds = load_dataset()
#   if config.drop_obs < 0:
#      config.drop_obs = None

#   if config.drop_obs:
#      train_ds = {key: jnp.delete(arr, config.drop_obs-1, axis=0) for key, arr in train_ds.items()}

  phi_dim = train_ds['Z'].shape[0]
  theta_dim = 2

  # phi_dim and theta_dim are also arguments of the flow,
  # as they define its dimension
  config.flow_kwargs.phi_dim = phi_dim
  config.flow_kwargs.theta_dim = theta_dim
  # Also is_smi modifies the dimension of the flow, due to the duplicated params
  config.flow_kwargs.is_smi = True

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))
  else:
    summary_writer = None

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state_list = []
  state_name_list = []

  state_name_list.append('phi')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(q_distr_phi),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'hp': jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))), # init vals right?
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # Get an initial sample of phi
  # (used below to initialize theta)
  phi_base_sample_init = hk.transform(q_distr_phi).apply(
      state_list[0].params,
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      hp=jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))),
  )['phi_base_sample']

  state_name_list.append('theta')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(q_distr_theta),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'phi_base_sample': phi_base_sample_init,
              'hp': jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))), # init vals right?
              'is_aux': False,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  if config.flow_kwargs.is_smi:
    state_name_list.append('theta_aux')
    state_list.append(
        initial_state_ckpt(
            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
            forward_fn=hk.transform(q_distr_theta),
            forward_fn_kwargs={
                'flow_name': config.flow_name,
                'flow_kwargs': config.flow_kwargs,
                'phi_base_sample': phi_base_sample_init,
                'hp': jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))), # init vals right?
                'is_aux': True,
            },
            prng_key=next(prng_seq),
            optimizer=make_optimizer(**config.optim_kwargs),
        ))

  # Print a useful summary of the execution of the flow architecture.
  logging.info('FLOW PHI:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: hk.transform(q_distr_phi).apply(
          params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          hp=jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))),
      ),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_list[0].params, next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)

  logging.info('FLOW THETA:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: hk.transform(q_distr_theta).apply(
          params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          phi_base_sample=phi_base_sample_init,
          hp=jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))),
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
  summary = tabulate_fn_(state_list[1].params, next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)

  # Jit function to update training states
  update_states_jit = lambda state_list, batch, prng_key: update_states(
      state_list=state_list,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer(**config.optim_kwargs),
      loss_fn=loss,
      loss_fn_kwargs={
          'mask_Z': jnp.array(config.mask_Z),
          'mask_Y': jnp.array(config.mask_Y),
          'num_samples': config.num_samples_elbo,
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'eta_sampling_a': config.eta_sampling_a,
          'eta_sampling_b': config.eta_sampling_b,
          'betahp_sampling_a': config.betahp_sampling_a_uniform,
          'betahp_sampling_b': config.betahp_sampling_b_uniform,
          'smi': config.estimate_smi,
      },
  )
  update_states_jit = jax.jit(update_states_jit)



  elbo_validation_jit = lambda state_list, batch, prng_key: elbo_estimate_along_eta(
      params_tuple=tuple(state.params for state in state_list),
      batch=batch,
      prng_key=prng_key,
      num_samples=config.num_samples_eval,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      eta_sampling_a= config.eta_sampling_a,
      eta_sampling_b= config.eta_sampling_b,
      betahp_sampling_a= config.betahp_sampling_a_uniform,
      betahp_sampling_b= config.betahp_sampling_b_uniform,
      smi=config.estimate_smi,
      mask_Z=jnp.array(config.mask_Z),
      mask_Y=jnp.array(config.mask_Y),
  )
  elbo_validation_jit = jax.jit(elbo_validation_jit)

  ############################################################################################################################
  # LOOCV as a loss function to optimize eta

  loss_neg_elpd_loocv = lambda hp_params, batch, prng_key, states_lists_vmp: -elpd_loocv_estimate(
      hp_params=hp_params,
      states_lists=states_lists_vmp,
      batch_predict=batch,
      prng_key=prng_key,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples=config.num_samples_elpd,
  )['lpd_pointwise'].sum()
  loss_neg_elpd_loocv = jax.jit(loss_neg_elpd_loocv)


  loss_neg_elpd_loocv_y = lambda hp_params, batch, prng_key, states_lists_vmp: -elpd_loocv_estimate(
      hp_params=hp_params,
      states_lists=states_lists_vmp,
      batch_predict=batch,
      prng_key=prng_key,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples=config.num_samples_elpd
  )['lpd_pointwise_y'].sum()
  loss_neg_elpd_loocv_y = jax.jit(loss_neg_elpd_loocv_y)


  loss_neg_elpd_loocv_z = lambda hp_params, batch, prng_key, states_lists_vmp: -elpd_loocv_estimate(
      hp_params=hp_params,
      states_lists=states_lists_vmp,
      batch_predict=batch,
      prng_key=prng_key,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples=config.num_samples_elpd,
  )['lpd_pointwise_z'].sum()
  loss_neg_elpd_loocv_z = jax.jit(loss_neg_elpd_loocv_z)

  ############################################################################################################################

  # elpd waic as a loss function to optimize eta
  loss_neg_elpd = lambda hp_params, batch, prng_key, state_list_vmp: -elpd_estimate_pointwise(
      hp_params=hp_params,
      state_list=state_list_vmp,
      batch=batch,
      prng_key=prng_key,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples=config.num_samples_elpd,
      # eta=eta,
      # betahp=betahp,
  )['elpd_waic_pointwise'].sum(axis=-1)
  loss_neg_elpd = jax.jit(loss_neg_elpd)

  # for Y module
  loss_neg_elpd_y = lambda hp_params, batch, prng_key, state_list_vmp: -elpd_estimate_pointwise(
      hp_params=hp_params,
      state_list=state_list_vmp,
      batch=batch,
      prng_key=prng_key,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples=config.num_samples_elpd,
      # eta=eta,
      # betahp=betahp,
  )['elpd_waic_pointwise_y'].sum(axis=-1)
  loss_neg_elpd_y = jax.jit(loss_neg_elpd_y)

  # for Z module
  loss_neg_elpd_z = lambda hp_params, batch, prng_key, state_list_vmp: -elpd_estimate_pointwise(
      hp_params=hp_params,
      state_list=state_list_vmp,
      batch=batch,
      prng_key=prng_key,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples=config.num_samples_elpd,
      # eta=eta,
      # betahp=betahp,
  )['elpd_waic_pointwise_z'].sum(axis=-1)
  loss_neg_elpd_z = jax.jit(loss_neg_elpd_z)


############################################################################################################################


  # # Jit optimization of eta
  # update_eta_star_state = lambda eta_star_state, batch, prng_key: update_state(
  #     state=eta_star_state,
  #     batch=batch,
  #     prng_key=prng_key,
  #     optimizer=make_optimizer_eta(**config.optim_kwargs_eta),
  #     loss_fn=loss_neg_elpd,
  #     loss_fn_kwargs={
  #         'state_list_vmp': state_list,
  #     },
  # )
  # update_eta_star_state = jax.jit(update_eta_star_state)

  ############################################################################################################################

  start_time = time.perf_counter()
  save_time_info = False
  save_after_training = False


  loss_stages_plot = False
  if loss_stages_plot:
    info_dict = {'lambda_training_loss':[], 
               'elbo_stage1':[], 'elbo_stage2':[]}

  if (state_list[0].step < config.training_steps):
    save_time_info = True

  if state_list[0].step < config.training_steps:
    save_after_training = True
    logging.info('Training Variational Meta-Posterior (VMP-flow)...')

    # Reset random key sequence
    prng_seq = hk.PRNGSequence(config.seed)

  while state_list[0].step < config.training_steps:
    

    # Plots to monitor training
    if config.activate_log_img:
        if ((state_list[0].step == 0) or
            (state_list[0].step % config.log_img_steps == 0)):
            # print("Logging images...\n")
            log_images(
                state_list=state_list,
                batch=train_ds,
                prng_key=next(prng_seq),
                config=config,
                show_elpd=False,
                show_posterior_range_allhps=False,
                eta_grid_len=20,
                summary_writer=summary_writer,
                workdir_png=workdir,
                #   workdir_mcmc=workdir_mcmc,
            )

    # Log learning rate
    summary_writer.scalar(
        tag='learning_rate',
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(state_list[0].step),
        step=state_list[0].step,
    )

    # SGD step
    state_list, metrics = update_states_jit(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
    )


    if loss_stages_plot:
        elbo_validation_dict = elbo_validation_jit(
            state_list=state_list,
            batch=train_ds,
            prng_key=next(prng_seq),
        )
        info_dict['elbo_stage1'].append(elbo_validation_dict['stage_1'])
        info_dict['elbo_stage2'].append(elbo_validation_dict['stage_2'])

    # The computed training loss corresponds to the model before update
    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state_list[0].step - 1,
    )

    if loss_stages_plot:
       info_dict['lambda_training_loss'].append(metrics["train_loss"])

    if state_list[0].step == 1:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    # Metrics for evaluation
    if state_list[0].step % 100 == 0:
       logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    if state_list[0].step % config.eval_steps == 0:

      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

      elbo_validation_dict = elbo_validation_jit(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
      )
      for k, v in elbo_validation_dict.items():
        summary_writer.scalar(
            tag=f'elbo_{k}',
            value=v.mean(),
            step=state_list[0].step,
        )

    if state_list[0].step % config.checkpoint_steps == 0:
      for state_i, state_name_i in zip(state_list, state_name_list):
        save_checkpoint(
            state=state_i,
            checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
            keep=config.checkpoints_keep,
        )


  logging.info('Final training step: %i', state_list[0].step)
  
   # End the timer
  end_time = time.perf_counter()
  elapsed_time = end_time - start_time

  hours, rem = divmod(elapsed_time, 3600)  # Divide by 3600 to get hours and remainder
  minutes, seconds = divmod(rem, 60)  # Divide remainder by 60 to get minutes and seconds

  # Prepare the output strings
  elapsed_time_str = f"Total elapsed time: {elapsed_time:.4f} seconds\n"
  formatted_time_str = f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds\n"

  # Print to console
  print(elapsed_time_str)
  print(formatted_time_str)

  # Save to file
  if save_time_info: 
    print("Saving timing info to file...")
    with open(workdir + "/timing_info.txt", "w") as file:
        file.write(elapsed_time_str)
        file.write(formatted_time_str)

  if loss_stages_plot:
     fig_two_stages = plot_all.lambda_loss_two_stages(
        info_dict=info_dict,
        step_now=state_list[0].step,
     )
     fig_two_stages.savefig(pathlib.Path(workdir) / ('lambda_training_stages' + ".png"))
     
     fig_lambda_loss = plot_all.lambda_loss(
        info_dict=info_dict,
        loss_low_treshold=300,
        loss_high_treshold=500000,
        step_now=state_list[0].step,
     )
     fig_lambda_loss.savefig(pathlib.Path(workdir) / ('lambda_training_loss' + ".png"))

#########################################################################
  # Saving checkpoint at the end of the training process
  # (in case training_steps is not multiple of checkpoint_steps)
  if save_after_training:
    for state_i, state_name_i in zip(state_list, state_name_list):
      save_checkpoint(
          state=state_i,
          checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
          keep=config.checkpoints_keep,
      )

#   # Plot of sample variance vs eta range
#   etas = jnp.linspace(0., 1., 200)
#   sample_dict_all = jax.vmap(lambda eta_i: sample_all_flows(
#         params_tuple=[state.params for state in state_list],
#         prng_key=next(prng_seq),
#         flow_name=config.flow_name,
#         flow_kwargs=config.flow_kwargs,
#         hp=jnp.broadcast_to(jnp.array([1., eta_i, 1., 1.]), (config.num_samples_elpd, 4)),
#     ))(etas) # dict where each key is (n_etas, n_samples, n_dim) e.g. phi is (200, 1000, 13)
  
#   fig_sampl, fig_poisson = plot_all.posterior_sample_variance(
#     etas=etas,
#     batch=train_ds,
#     sample_dict_all=sample_dict_all,
#     config=config,
#     )
#   fig_sampl.savefig(pathlib.Path(workdir) / ('var_postsamples_with_eta' + ".png"))
#   fig_poisson.savefig(pathlib.Path(workdir) / ('var_Poissonmu_with_eta' + ".png"))
  
  # Last plot of posteriors
  if config.activate_log_img:
    log_images(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
        config=config,
        show_elpd=False,
        show_posterior_range_allhps=config.estimate_smi,
        show_posterior_range_priorhpsVMP_etafixed=bool(1 - bool(config.estimate_smi)),
        eta_grid_len=20,
        summary_writer=summary_writer,
        workdir_png=workdir,
        workdir_mcmc=workdir_mcmc,
    )
# #########################################################################
  ## Find best eta ###

  logging.info('Finding best hyperparameters...')

  if config.train_hyperparameters:
    # Reset random key sequence
    prng_seq = hk.PRNGSequence(config.seed)
    train_ds = load_dataset()

    if config.tune_hparams == 'elpd_loocv':
        
        phi_dim = train_ds['Z'].shape[0]
        theta_dim = 2
        config.flow_kwargs.phi_dim = phi_dim
        config.flow_kwargs.theta_dim = theta_dim
        # Also is_smi modifies the dimension of the flow, due to the duplicated params
        config.flow_kwargs.is_smi = True

        states_lists_dict = {'z':[], 'y':[]}

        elpd_loocv_pointwise_jit = lambda hp_params, batch, prng_key, states_lists_vmp: elpd_loocv_estimate(
                hp_params=hp_params,
                states_lists=states_lists_vmp,
                batch_predict=batch,
                prng_key=prng_key,
                flow_name=config.flow_name,
                flow_kwargs=config.flow_kwargs,
                num_samples=config.num_samples_elpd,
                )
        elpd_loocv_pointwise_jit = jax.jit(elpd_loocv_pointwise_jit)

        #    states_lists = []
        for lik_type in ['z', 'y']:
            for obs_idx in jnp.arange(phi_dim):
                    checkpoint_dir = str(pathlib.Path(workdir) / f'loocv_{lik_type}/dropped_{obs_idx}/checkpoints')
                    try:
                        assert os.path.isdir(checkpoint_dir), f"Directory needed for ELPD-LOOCV optim does not exist: {checkpoint_dir}"
                    except AssertionError as error:
                        sys.exit(1)  # Exit with a non-zero status code to indicate error
                    # assert os.path.isdir(checkpoint_dir), f"Directory needed for ELPD-LOOCV optim does not exist: {checkpoint_dir}"
                    state_list = []
                    state_name_list = []

                    state_name_list.append('phi')
                    state_list.append(
                        initial_state_ckpt(
                            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
                            forward_fn=hk.transform(q_distr_phi),
                            forward_fn_kwargs={
                                'flow_name': config.flow_name,
                                'flow_kwargs': config.flow_kwargs,
                                'hp': jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))), # init vals right?
                            },
                            prng_key=next(prng_seq),
                            optimizer=make_optimizer(**config.optim_kwargs),
                        ))

                    # Get an initial sample of phi
                    # (used below to initialize theta)
                    phi_base_sample_init = hk.transform(q_distr_phi).apply(
                        state_list[0].params,
                        next(prng_seq),
                        flow_name=config.flow_name,
                        flow_kwargs=config.flow_kwargs,
                        hp=jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))),
                    )['phi_base_sample']

                    state_name_list.append('theta')
                    state_list.append(
                        initial_state_ckpt(
                            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
                            forward_fn=hk.transform(q_distr_theta),
                            forward_fn_kwargs={
                                'flow_name': config.flow_name,
                                'flow_kwargs': config.flow_kwargs,
                                'phi_base_sample': phi_base_sample_init,
                                'hp': jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))), # init vals right?
                                'is_aux': False,
                            },
                            prng_key=next(prng_seq),
                            optimizer=make_optimizer(**config.optim_kwargs),
                        ))
                    if config.flow_kwargs.is_smi:
                        state_name_list.append('theta_aux')
                        state_list.append(
                            initial_state_ckpt(
                                checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
                                forward_fn=hk.transform(q_distr_theta),
                                forward_fn_kwargs={
                                    'flow_name': config.flow_name,
                                    'flow_kwargs': config.flow_kwargs,
                                    'phi_base_sample': phi_base_sample_init,
                                    'hp': jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))), # init vals right?
                                    'is_aux': True,
                                },
                                prng_key=next(prng_seq),
                                optimizer=make_optimizer(**config.optim_kwargs),
                            ))
                    states_lists_dict[lik_type].append(state_list)

    elif config.tune_hparams == 'elpd_waic':
            phi_dim = train_ds['Z'].shape[0]
            theta_dim = 2
            config.flow_kwargs.phi_dim = phi_dim
            config.flow_kwargs.theta_dim = theta_dim
            # Also is_smi modifies the dimension of the flow, due to the duplicated params
            config.flow_kwargs.is_smi = True

            checkpoint_dir = str(pathlib.Path(workdir) / f'checkpoints')

            state_list = []
            state_list.append(
                initial_state_ckpt(
                    checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
                    forward_fn=hk.transform(q_distr_phi),
                    forward_fn_kwargs={
                        'flow_name': config.flow_name,
                        'flow_kwargs': config.flow_kwargs,
                        'hp': jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))), # init vals right?
                    },
                    prng_key=next(prng_seq),
                    optimizer=make_optimizer(**config.optim_kwargs),
                ))

            # Get an initial sample of phi
            # (used below to initialize theta)
            phi_base_sample_init = hk.transform(q_distr_phi).apply(
                state_list[0].params,
                next(prng_seq),
                flow_name=config.flow_name,
                flow_kwargs=config.flow_kwargs,
                hp=jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))),
            )['phi_base_sample']

            state_list.append(
                initial_state_ckpt(
                    checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
                    forward_fn=hk.transform(q_distr_theta),
                    forward_fn_kwargs={
                        'flow_name': config.flow_name,
                        'flow_kwargs': config.flow_kwargs,
                        'phi_base_sample': phi_base_sample_init,
                        'hp': jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))), # init vals right?
                        'is_aux': False,
                    },
                    prng_key=next(prng_seq),
                    optimizer=make_optimizer(**config.optim_kwargs),
                ))
            if config.flow_kwargs.is_smi:
                state_list.append(
                    initial_state_ckpt(
                        checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
                        forward_fn=hk.transform(q_distr_theta),
                        forward_fn_kwargs={
                            'flow_name': config.flow_name,
                            'flow_kwargs': config.flow_kwargs,
                            'phi_base_sample': phi_base_sample_init,
                            'hp': jnp.ones((config.num_samples_elbo, len(config.cond_hparams_names))), # init vals right?
                            'is_aux': True,
                        },
                        prng_key=next(prng_seq),
                        optimizer=make_optimizer(**config.optim_kwargs),
                    ))
    else:
        raise ValueError("Please define valid config.tune_hparams")

    if not os.path.exists(workdir + f'/hparam_tuning_{config.tune_hparams}'):
        os.makedirs(workdir + f'/hparam_tuning_{config.tune_hparams}', exist_ok=True)

    # Initialize search with Bayes
    all_optimisers = {'elbo_opt':make_optimizer(**config.optim_kwargs),
                        'plain_lr1':make_optimizer_eta(**config.optim_kwargs_hp),  
                        'plain_lr2':make_optimizer_eta(config.optim_kwargs_hp_learning_rate_alternative)}


    all_inits = {'default':jnp.array([1., 1., 1.]), 
                'low':jnp.array([0.00001, 0.5, 0.5]), 
                'high':jnp.array([0.5, 5., 5.]),
                'medium':jnp.array([1, 3., 3.]),
                'small':jnp.array([0.5, 0.5, 0.5]),
                'smallmedium':jnp.array([0.5, 2., 2.]),
                'mediumhigh':jnp.array([0.99, 3., 3.]),
                'mediummedium':jnp.array([0.9, 2., 2.]),}
    
    all_names = ['eta', 'c1', 'c2']
    all_loglik = ['z','y']
    all_elpd_waic_optim = {'y': loss_neg_elpd_y, 'z': loss_neg_elpd_z}
    all_elpd_loocv_optim = {'y': loss_neg_elpd_loocv_y, 'z': loss_neg_elpd_loocv_z} 

    optim_mask = jnp.array([1 if i in config.cond_hparams_names else 0 for i in all_names])
    print('optim mask:', optim_mask)
    optim_mask_indices = (tuple(i for i, x in enumerate(optim_mask) if x == 0),tuple(i for i, x in enumerate(optim_mask) if x == 1))

    for optimiser_name, my_optimiser in all_optimisers.items():  

        for (all_init_name, all_init) in all_inits.items():  
            
            hp_star_init = all_init[optim_mask==1]
            hp_fixed = all_init[optim_mask==0]

            for lik_type in all_loglik: 
                print(f"optimiser: {optimiser_name}, init type: {all_init_name}, loglik: {lik_type}")
                info_dict = {'init_vals':hp_star_init, 'likelihood':lik_type,
                            'init_type':all_init_name,
                'loss':[], 'params':[], 'step':[], 'hp_names':config.cond_hparams_names}
    
                if config.tune_hparams == 'waic':

                    # if jax.process_index() == 0:
                    #     summary_writer_hp = tensorboard.SummaryWriter(workdir + f'/hparam_tuning_{config.tune_hparams}/{lik_type}_{all_init_name}_{optimiser_name}')
                    #     summary_writer_hp.hparams({all_init_name:hp_star_init, optimiser_name:my_optimiser, lik_type:all_elpd_waic_optim[lik_type]})

                    # Jit optimization of eta
                    update_hp_star_state_waic = lambda hp_star_state, batch, prng_key: update_state(
                        state=hp_star_state,
                        batch=batch,
                        prng_key=prng_key,
                        optimizer=my_optimiser, #make_optimizer_eta(**config.optim_kwargs_hp),
                        loss_fn=all_elpd_waic_optim[lik_type],
                        loss_fn_kwargs={
                            'state_list_vmp': state_list,
                        },
                    )
                    update_hp_star_state = jax.jit(update_hp_star_state_waic)

                elif config.tune_hparams == 'elpd_loocv':
                    if jax.process_index() == 0:
                        summary_writer_hp = tensorboard.SummaryWriter(workdir + f'/hparam_tuning_{config.tune_hparams}/{lik_type}_{all_init_name}_{optimiser_name}')
                        summary_writer_hp.hparams({all_init_name:hp_star_init, optimiser_name:my_optimiser, lik_type:all_elpd_loocv_optim[lik_type]})

                    # Jit optimization of eta
                    update_hp_star_state_loocv = lambda hp_star_state, batch, prng_key: update_state(
                            state=hp_star_state,
                            batch=batch,
                            prng_key=prng_key,
                            optimizer=my_optimiser, #make_optimizer_eta(**config.optim_kwargs_hp),
                            loss_fn=all_elpd_loocv_optim[lik_type],
                            loss_fn_kwargs={
                                'states_lists_vmp': states_lists_dict[lik_type]
                            },
                        )
                    update_hp_star_state = jax.jit(update_hp_star_state_loocv)


                # SGD over elpd #
                hp_star_state = TrainState(
                    params=hp_star_init,
                    opt_state=my_optimiser.init(hp_star_init),
                    step=0,
                )
                for _ in range(config.hp_star_steps):
                    hp_star_state, neg_elpd = update_hp_star_state(
                        hp_star_state,
                        batch=train_ds,
                        prng_key=next(prng_seq),
                    )

                    # # why is off-diag not exactly zero?? like -0.0004??
                    # if ((config.hp_star_steps%1000==0)&(config.tune_hparams=='elpd_loocv')):
                    #     elpd_loocv_dict = elpd_loocv_pointwise_jit(
                    #         hp_params = hp_star_state.params,
                    #         states_lists_vmp=states_lists_dict[lik_type],
                    #         batch=train_ds,
                    #         prng_key=next(prng_seq),
                    #     )
                            
                    #     lpds = elpd_loocv_dict[f'lpd_pointwise_{lik_type}']
                    #     jax.debug.print('lpds shape:', lpds.shape)
                    #     n_est_nonzeros = jnp.count_nonzero(lpds - np.diag(np.diagonal(lpds)))
                    #     assert n_est_nonzeros == 0

                    # if state_list[0].step % config.hp_star_steps == 0:
                    #   logging.info("STEP: %5d; training loss: %.3f", state_list[0].step,
                    #                neg_elpd["train_loss"])


                    if hp_star_state.step % 100 == 0:
                        labs = "STEP: %5d; training loss: %.6f " + ' '.join([hp + ':%.3f' for hp in config.cond_hparams_names])
                        logging.info(labs,
                            float(hp_star_state.step),
                        float(neg_elpd["train_loss"]), *[float(hp_star_state.params[i]) for i in range(len(config.cond_hparams_names))])
                        print(hp_star_state.params)

                    hp_star_state = TrainState(
                        params=jnp.hstack([jnp.clip(hp_star_state.params[config.cond_hparams_names.index('eta')],0., 1.) if 'eta' in config.cond_hparams_names else [],
                                        jnp.clip(hp_star_state.params[config.cond_hparams_names.index('c1')],0.000001, 15.) if 'c1' in config.cond_hparams_names else [],
                                        jnp.clip(hp_star_state.params[config.cond_hparams_names.index('c2')],0.000001, 15) if 'c2' in config.cond_hparams_names else [],
                                            ]),
                        opt_state=hp_star_state.opt_state,
                        step=hp_star_state.step,
                    )
                    if hp_star_state.step % 100 == 0:
                        print(hp_star_state.params)

                    info_dict['loss'].append(neg_elpd["train_loss"])
                    info_dict['params'].append(hp_star_state.params)
                    info_dict['step'].append(hp_star_state.step)

                    if config.tune_hparams_tensorboard:
                        summary_writer_hp.scalar(
                            tag=f'hp_loss_neg_{config.tune_hparams}',
                            value=neg_elpd['train_loss'],
                            step=hp_star_state.step - 1,
                        )
                        for hp_star_name, hp_star_i in zip(config.cond_hparams_names, hp_star_state.params):
                            summary_writer_hp.scalar(
                                tag=f'hp_opt_{config.tune_hparams}_{hp_star_name}',
                                value=hp_star_i,
                                step=hp_star_state.step - 1,
                            )
                with open(workdir + f"/hp_info_{'eta' if config.estimate_smi else 'only'}c1c2_{config.tune_hparams}_{all_init_name}_{info_dict['likelihood']}_{optimiser_name}.sav", 'wb') as f:
                        pickle.dump(info_dict, f)
        bools = np.array(optim_mask==1)
        plot_all.plot_optim_hparams_vs_true(path=workdir,
                                    init_names=list(all_inits.keys()),
                                    optimiser_name=optimiser_name,
                                    loglik_types=all_loglik,
                                    loss_type=config.tune_hparams,
                                    hp_names=np.array(all_names)[bools],
                                    )


    return state_list
