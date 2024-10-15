"""A simple example of a flow model trained on Epidemiology data."""

import pathlib
import os
import warnings
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256) 

from absl import logging

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import scipy

from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax
import distrax

import variationalmetaposterior.examples.epidemiology_new.old_py.flows_all_additive as flows_all_additive
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
  q_distr = getattr(flows_all_additive, flow_name + '_phi')(**flow_kwargs)

  num_samples = hp.shape[0]

  # Sample from flows
  (phi_sample, phi_log_prob_posterior,
   phi_base_sample) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=[hp[:,:2],hp[:,2:], None],
             )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows_all_additive.split_flow_phi(
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
  q_distr = getattr(flows_all_additive, flow_name + '_theta')(**flow_kwargs)

  # Sample from flow
  (theta_sample, theta_log_prob_posterior) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
    #   context=[hp[:,2:], phi_base_sample],
      context=[hp[:,:2],hp[:,2:], phi_base_sample],
  )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows_all_additive.split_flow_theta(
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
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta_sampling_a: float,
    eta_sampling_b: float,
    betahp_sampling_a: float,
    betahp_sampling_b: float,
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample eta values (only for Y module)
  etas_elbo = jax.random.beta(
      key=next(prng_seq),
      a=eta_sampling_a,
      b=eta_sampling_b,
      shape=(num_samples, 1),
  )

  # Set eta_z=1
  etas_elbo = jnp.concatenate([jnp.ones_like(etas_elbo), etas_elbo], axis=-1)

  # Sample beta concentration values
#   betahp_elbo = jax.random.uniform(
#       key=next(prng_seq),
#       minval=betahp_sampling_a, 
#       maxval=betahp_sampling_b,
#       shape=(num_samples, 2),
#   )
  
#   # Sample beta concentration values
  betahp_elbo = jax.random.gamma(
      key=next(prng_seq),
      a=betahp_sampling_a, 
      shape=(num_samples, 2),
  )/betahp_sampling_b

#   quantile_points = jnp.linspace(0, 1, num_samples)
#   betahp_elbo = jnp.tile(jnp.array([scipy.stats.gamma.ppf(p, a=betahp_sampling_a, scale=1/betahp_sampling_b) for p in quantile_points][:-1]), reps=(2,1))

  # concat all
  hp_elbo = jnp.concatenate([etas_elbo, betahp_elbo], axis=-1)
#   hp_elbo = jnp.ones(shape=(num_samples, 4))

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      prng_key=next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      hp=hp_elbo,
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
def compute_elpd(
    posterior_sample_dict: Dict[str, Any],
    batch: Batch,
) -> Mapping[str, Array]:
  """Compute ELPD.

  Estimates the ELPD based on two Monte Carlo approximations:
    1) Using WAIC.

  Args:
    posterior_sample_dict: Dictionary of posterior samples.
    batch: Batch of data (the one that was for training).

  Returns:
    Dictionary of ELPD estimates, with keys:
      - 'elpd_waic_pointwise': WAIC-based estimate.
  """

  # Initialize dictionary for output
  elpd_out = {}

  num_samples, _ = posterior_sample_dict['phi'].shape # TOCHECK: is it correct?

  ### WAIC ###
#   # Compute LPD

  loglik_pointwise_insample = log_prob_fun_integrated.log_lik_vectorised(
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

  assert len(hp_params) == 4
  q_distr_out_i = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=prng_key,  # same key to reduce variance of posterior along eta
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      hp=jnp.broadcast_to(hp_params, (num_samples, 4)) # num modules
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
    show_posterior_range:bool,
    eta_grid_len: int,
    summary_writer: Optional[SummaryWriter],
    workdir_png: Optional[str],
    workdir_mcmc:Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  # Plot posterior samples
  if show_posterior_range:
    images = []
    priorhps = {'priorhp_converged_cut':[0.778, 15.000], 
                'priorhp_ones': [1.,1.],
                'priorhp_alternative_bayes':[0.2, 0.3],
                'priorhp_alternative_cut':[4, 0.04], 
                'priorhp_converged_bayes':[5.741, 15.000] }
    priorhp_main = {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
        'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
    smi_etas = {'eta_bayes':1., 'eta_cut':0.0001} #[[1., 1.], [1., 0.0001]]
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
                hp=jnp.broadcast_to(jnp.concatenate([jnp.stack([1., jnp.array(smi_etas[eta_k])]), jnp.array(priorhps[priorhp_k])], axis=-1),
                             (config.num_samples_plot,) + (4,)),

            )
            posterior_sample_dict[eta_k][priorhp_k] = q_distr_out['posterior_sample']
            theta = posterior_sample_dict[eta_k][priorhp_k]['theta']
            theta_aux = posterior_sample_dict[eta_k][priorhp_k]['theta_aux']

            _, theta_dim = theta.shape
            posterior_samples_df_theta = pd.DataFrame(
                theta, columns=[f"theta_{i}" for i in range(1, theta_dim + 1)])
            posterior_samples_df_theta['eta1'] = f'= {1 if eta_v==1. else 0}'
            posterior_sample_dfs_theta[eta_k][priorhp_k] = posterior_samples_df_theta
    
            posterior_samples_df_theta_aux = pd.DataFrame(
                theta_aux, columns=[f"theta_{i}" for i in range(1, theta_dim + 1)])
            posterior_samples_df_theta_aux['eta1'] = f'= {1 if eta_v==1. else 0}'
            posterior_sample_dfs_theta_aux[eta_k][priorhp_k] = posterior_samples_df_theta_aux
    
    # produce phi vs loglambda plots
    fig_theta_loglambda = plot_all.plot_phi_loglambda(
        posterior_sample_dict=posterior_sample_dict,
        priorhps=priorhps,
        smi_etas=smi_etas,
        priorhp_toplot={'eta_bayes':['priorhp_ones', 'priorhp_converged_bayes'],
                                            'eta_cut':['priorhp_ones', 'priorhp_converged_cut']},
        # xlim=[0, 0.3],
        # ylim=[-2.5, 2],
        )
    if workdir_png:
        fig_theta_loglambda.savefig(pathlib.Path(workdir_png) / ( "epidemiology_phi_vs_loglambda" + ".png"))
    if summary_writer:
        images.append(plot_to_image(fig_theta_loglambda))

    # produce phi plots
    for eta_ix, (eta_k, eta_v) in enumerate(smi_etas.items()):
        fig_phi = plot_all.plot_posterior_phi_hprange(
          posterior_sample_dict=posterior_sample_dict,
          eta = (eta_k,eta_v),
          priorhps = priorhps,
          priorhp_main = priorhp_main,
        )
        if workdir_png:
            fig_phi.savefig(pathlib.Path(workdir_png) / (f'epidemiology_phi_hprange_eta{smi_etas[eta_k]}' + ".png"))
        if summary_writer:
            images.append(plot_to_image(fig_phi))

    # produce theta aux plots
    fig_theta_aux = plot_all.plot_posterior_theta_hprange(
          posterior_sample_dfs=posterior_sample_dfs_theta_aux,
          smi_etas = smi_etas,
          priorhps = priorhps,
          priorhp_main = priorhp_main,
    )
    if workdir_png:
        fig_theta_aux.savefig(pathlib.Path(workdir_png) / (f"epidemiology_theta_aux_hprange_eta{smi_etas['eta_cut']}" + ".png"))
    if summary_writer:
        images.append(plot_to_image(fig_theta_aux))

    # produce theta plots
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

    # produce phi priors plot
    fig_betas = plot_all.plot_hprange_betapriors(
      priorhps = priorhps,
      priorhp_main = priorhp_main,)
    if workdir_png:
        fig_betas.savefig(pathlib.Path(workdir_png) / (f"epidemiology_betaprior_eta{smi_etas['eta_cut']}" + ".png"))
    if summary_writer:
        images.append(plot_to_image(fig_betas))

    # produce theta plots at hp = 1,1
    priorhp_main = {'main': {'eta_bayes': 'priorhp_ones',
            'eta_cut': 'priorhp_ones'},
    'secondary': {'eta_bayes': 'priorhp_converged_bayes',
            'eta_cut': 'priorhp_converged_cut'}}
    fig_theta = plot_all.plot_posterior_theta_hprange(
          posterior_sample_dfs=posterior_sample_dfs_theta,
          smi_etas = smi_etas,
          priorhps = priorhps,
          priorhp_main = priorhp_main,
    )

    if workdir_png:
        fig_theta.savefig(pathlib.Path(workdir_png) / (f"epidemiology_theta_hprange_eta{smi_etas['eta_cut']}_hp1" + ".png"))
    if summary_writer:
        images.append(plot_to_image(fig_theta))

    if workdir_mcmc:
        # produce theta plots at hp opt VS MCMC
        with open(workdir_mcmc + f'/eta_1.000/mcmc_theta_eta1.0_c1_5.741_c2_15.sav', 'rb') as fr:
            mcmc_theta_eta1 = pickle.load(fr)
        with open(workdir_mcmc + f'/eta_0.0001/mcmc_theta_eta0.0001_c1_0778_c2_15.sav', 'rb') as fr:
            mcmc_theta_eta0001 = pickle.load(fr)
            priorhp_main = {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
                'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
        fig_theta_mcmc = plot_all.plot_posterior_theta_hprange_vsmcmc(
            posterior_sample_dfs=posterior_sample_dfs_theta,
            mcmc_dfs={'eta1':mcmc_theta_eta1['theta'], 'eta0001':mcmc_theta_eta0001['theta']},
            smi_etas = smi_etas,
            priorhps = priorhps,
            priorhp_main = priorhp_main,
        )

        if workdir_png:
            fig_theta_mcmc.savefig(pathlib.Path(workdir_png) / (f"epidemiology_theta_vsMCMC_hprange_eta{smi_etas['eta_cut']}_hpOPT" + ".png"))
        if summary_writer:
            images.append(plot_to_image(fig_theta_mcmc))

        # produce theta plots at hp = 1,1 VS MCMC
        with open(workdir_mcmc + f'/eta_1.000/mcmc_theta_eta1.0.sav', 'rb') as fr:
            mcmc_theta_eta1 = pickle.load(fr)
        with open(workdir_mcmc + f'/eta_0.0001/mcmc_theta_eta0.0001.sav', 'rb') as fr:
            mcmc_theta_eta0001 = pickle.load(fr)
        priorhp_main = {'main': {'eta_bayes': 'priorhp_ones',
                'eta_cut': 'priorhp_ones'},
        'secondary': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'}}
        fig_theta_mcmc = plot_all.plot_posterior_theta_hprange_vsmcmc(
            posterior_sample_dfs=posterior_sample_dfs_theta,
            mcmc_dfs={'eta1':mcmc_theta_eta1['theta'], 'eta0001':mcmc_theta_eta0001['theta']},
            smi_etas = smi_etas,
            priorhps = priorhps,
            priorhp_main = priorhp_main,
        )

        if workdir_png:
            fig_theta_mcmc.savefig(pathlib.Path(workdir_png) / (f"epidemiology_theta_vsMCMC_hprange_eta{smi_etas['eta_cut']}_hp1" + ".png"))
        if summary_writer:
            images.append(plot_to_image(fig_theta_mcmc))

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
              'hp': jnp.ones((config.num_samples_elbo, 4)), # init vals right?
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
      hp=jnp.ones((config.num_samples_elbo, 4)),
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
              'hp': jnp.ones((config.num_samples_elbo, 4)), # init vals right?
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
                'hp': jnp.ones((config.num_samples_elbo, 4)), # init vals right?
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
          hp=jnp.ones((config.num_samples_elbo, 4)),
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
          hp=jnp.ones((config.num_samples_elbo, 4)),
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
          'num_samples': config.num_samples_elbo,
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'eta_sampling_a': config.eta_sampling_a,
          'eta_sampling_b': config.eta_sampling_b,
          'betahp_sampling_a': config.betahp_sampling_a,
          'betahp_sampling_b': config.betahp_sampling_b,
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
      betahp_sampling_a= config.betahp_sampling_a,
      betahp_sampling_b= config.betahp_sampling_b,
    #   eta_sampling_a=1.,
    #   eta_sampling_b=1.,
    #   betahp_sampling_a=1.,
    #   betahp_sampling_b=1.,
  )
  elbo_validation_jit = jax.jit(elbo_validation_jit)

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


  # Jit optimization of eta
  update_hp_star_state = lambda hp_star_state, batch, prng_key: update_state(
      state=hp_star_state,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer_eta(**config.optim_kwargs_hp),
      loss_fn=loss_neg_elpd_z,
      loss_fn_kwargs={
          'state_list_vmp': state_list,
      },
  )
  update_hp_star_state = jax.jit(update_hp_star_state)

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

  save_after_training = False
  loss_stages_plot = False
  if loss_stages_plot:
    info_dict = {'lambda_training_loss':[], 
               'elbo_stage1':[], 'elbo_stage2':[]}


  if state_list[0].step < config.training_steps:
    save_after_training = True
    logging.info('Training Variational Meta-Posterior (VMP-flow)...')

    # Reset random key sequence
    prng_seq = hk.PRNGSequence(config.seed)

  while state_list[0].step < config.training_steps:
    

    # Plots to monitor training
    if ((state_list[0].step == 0) or
        (state_list[0].step % config.log_img_steps == 0)):
      # print("Logging images...\n")
      log_images(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
          config=config,
          show_elpd=False,
          show_posterior_range=True,
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

# #########################################################################
#   ## Find best eta ###

#   logging.info('Finding best hyperparameters...')

  

#   # Reset random key sequence
#   prng_seq = hk.PRNGSequence(config.seed)

#   # Initialize search with Bayes
#   hp_star = jnp.array([1., 1., 1., 1.])
#   info_dict = {'init':hp_star, 'likelihood':'z',
#   'loss':[], 'params':[], 'step':[]}

#   # key_search = next(prng_seq)

#   # SGD over elpd #
#   hp_star_state = TrainState(
#       params=hp_star,
#       opt_state=make_optimizer_eta(**config.optim_kwargs_hp).init(hp_star),
#       step=0,
#   )
#   for _ in range(config.hp_star_steps):
#     hp_star_state, neg_elpd = update_hp_star_state(
#         hp_star_state,
#         batch=train_ds,
#         prng_key=next(prng_seq),
#     )

#     # if state_list[0].step % config.hp_star_steps == 0:
#     #   logging.info("STEP: %5d; training loss: %.3f", state_list[0].step,
#     #                neg_elpd["train_loss"])
#     # info_dict['loss'].append(neg_elpd["train_loss"])
#     # info_dict['params'].append(hp_star_state.params)
#     # info_dict['step'].append(hp_star_state.step)

#     if hp_star_state.step % 100 == 0:
#       logging.info("STEP: %5d; training loss: %.3f; eta0:%.3f; eta1: %.3f; conc1: %.3f, conc2:%.3f", hp_star_state.step,
#                    neg_elpd["train_loss"], hp_star_state.params[0], hp_star_state.params[1],
#                    hp_star_state.params[2], hp_star_state.params[3])

#     # Clip eta_star to [0,1] hypercube and hp_star to [0.000001,..]
#     hp_star_state = TrainState(
#         params=jnp.hstack([jnp.clip(hp_star_state.params[:2],0, 1),
#                            jnp.clip(hp_star_state.params[2:],0.000001, 15)]),
#         opt_state=hp_star_state.opt_state,
#         step=hp_star_state.step,
#     )

#     summary_writer.scalar(
#         tag='rnd_eff_hp_star_neg_elpd',
#         value=neg_elpd['train_loss'],
#         step=hp_star_state.step - 1,
#     )
#     for i, hp_star_i in enumerate(hp_star_state.params):
#       summary_writer.scalar(
#           tag=f'rnd_eff_eta_star_{i}',
#           value=hp_star_i,
#           step=hp_star_state.step - 1,
#       )
# # #   with open(workdir + f"/hp_info_eta{hp_star[1]:.6f}_{info_dict['likelihood']}.sav", 'wb') as f:
# # #     pickle.dump(info_dict, f)

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
  log_images(
      state_list=state_list,
      batch=train_ds,
      prng_key=next(prng_seq),
      config=config,
      show_elpd=True,
      show_posterior_range=True,
      eta_grid_len=20,
      summary_writer=summary_writer,
      workdir_png=workdir,
    #   workdir_mcmc=workdir_mcmc,
  )

  return state_list
