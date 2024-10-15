"""A simple example of a flow model trained on Epidemiology data."""

import pathlib

from absl import logging

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax
import distrax

import flows_all
import log_prob_fun_all
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
    eta: Array,
    betahp: Array,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  # Define normalizing flows
  q_distr = getattr(flows_all, flow_name + '_phi')(**flow_kwargs)

  num_samples = eta.shape[0]

  # Sample from flows
  (phi_sample, phi_log_prob_posterior,
   phi_base_sample) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=[eta, betahp, None],
   )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows_all.split_flow_phi(
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
    eta: Array,
    betahp: Array,
    is_aux: bool,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  num_samples = phi_base_sample.shape[0]

  # Define normalizing flows
  q_distr = getattr(flows_all, flow_name + '_theta')(**flow_kwargs)

  # Sample from flow
  (theta_sample, theta_log_prob_posterior) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
      context=[eta, betahp, phi_base_sample],
  )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows_all.split_flow_theta(
          samples=theta_sample,
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
    smi_eta: SmiEta,
    betahp: Array,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  prng_seq = hk.PRNGSequence(prng_key)

  # phi
  q_distr_out = hk.transform(q_distr_phi).apply(
      params_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      eta=smi_eta['modules'],
      betahp = betahp,
  )

  # theta
  q_distr_out_theta = hk.transform(q_distr_theta).apply(
      params_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      phi_base_sample=q_distr_out['phi_base_sample'],
      eta=smi_eta['modules'],
      betahp = betahp,
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
      eta=smi_eta['modules'],
      betahp = betahp,
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

#   etas_elbo =  jax.random.normal(
#   key=next(prng_seq),
#   shape=(num_samples, 1))*0.001+0.003

  # Set eta_z=1
  etas_elbo = jnp.concatenate([jnp.ones_like(etas_elbo), etas_elbo], axis=-1)
  smi_eta_elbo = {'modules': etas_elbo}

  # Sample beta concentration values
  betahp_elbo = jax.random.uniform(
      key=next(prng_seq),
      minval=betahp_sampling_a, 
      maxval=betahp_sampling_b,
      shape=(num_samples, 2),
  )

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      prng_key=next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      smi_eta=smi_eta_elbo,
      betahp = betahp_elbo,
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
      lambda posterior_sample_i, smi_eta_i, betahp_i: log_prob_fun_all.log_prob_joint(
          batch=batch,
          posterior_sample_dict=posterior_sample_i,
          smi_eta=smi_eta_i,
          betahp=betahp_i,
      ))(
          jax.tree_map(lambda x: jnp.expand_dims(x, 1),
                       posterior_sample_dict_stg1),
          smi_eta_elbo,
          betahp_elbo,
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
      lambda posterior_sample_i, betahp_i: log_prob_fun_all.log_prob_joint(
          batch=batch,
          posterior_sample_dict=posterior_sample_i,
          smi_eta=None,
          betahp=betahp_i,
      ))(
          jax.tree_map(lambda x: jnp.expand_dims(x, 1),
                       posterior_sample_dict_stg2),
        betahp_elbo,                   
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

  loglik_pointwise_insample = log_prob_fun_all.log_lik_vectorised(
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


  loglik_pointwise_y_insample = log_prob_fun_all.log_lik_y_vectorised(
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


  loglik_pointwise_z_insample = log_prob_fun_all.log_lik_z_vectorised(
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
      smi_eta={
          'modules': jnp.broadcast_to(hp_params[:2], (num_samples, 2)) # num modules
      },
      betahp=jnp.broadcast_to(hp_params[2:], (num_samples, 2))
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

def posterior_sample_variance(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    config: ConfigDict,
    workdir_png: Optional[str],
) -> None:

    prng_seq = hk.PRNGSequence(prng_key)
    etas = jnp.linspace(0., 1., 200)

    sample_dict_all = jax.vmap(lambda eta_i: sample_all_flows(
        params_tuple=[state.params for state in state_list],
        prng_key=prng_key,
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        smi_eta={'modules': jnp.broadcast_to(jnp.array([1., eta_i]), (config.num_samples_elpd, 2))},
        betahp=jnp.broadcast_to(jnp.array([1., 1.]), (config.num_samples_elpd, 2)),
    ))(etas) # dict where each key is (n_etas, n_samples, n_dim) e.g. phi is (200, 1000, 13)

    var_phi = jnp.var(sample_dict_all['posterior_sample']['phi'], axis=1) # (n_etas, 13)
    var_theta = jnp.var(sample_dict_all['posterior_sample']['theta'], axis=1) # (n_etas, 2)


    fig, ax = plt.subplots(1,3, figsize=(15,5))
    fig.suptitle(f'Variance of {config.num_samples_elpd} posterior samples for a range of eta1, betahps fixed to 1', fontsize=18)

    for dim in range(var_phi.shape[1]):
        ax[0].plot(etas, var_phi[:,dim], label=f'Phi{dim}')
    ax[0].legend()
    ax[0].set_xlabel('eta range', fontsize=15)

    ax[1].plot(etas, var_theta[:,0], label=f'Theta0')
    ax[1].legend()
    ax[1].set_xlabel('eta range', fontsize=15)

    ax[2].plot(etas, var_theta[:,1], label=f'Theta1')
    ax[2].legend()
    ax[2].set_xlabel('eta range', fontsize=15)
    plt.tight_layout()
    plt.close()

    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('var_postsamples_with_eta' + ".png"))

    fig, ax = plt.subplots(figsize=(10,10))
    fig.suptitle(f'Variance of Poisson mean of Y module for {config.num_samples_elpd} posterior samples', fontsize=18)
    for dim in range(var_phi.shape[1]):
        log_incidence = sample_dict_all['posterior_sample']['theta'][:,:,0] + sample_dict_all['posterior_sample']['theta'][:,:,1] * sample_dict_all['posterior_sample']['phi'][:,:,dim]
        mu = batch['T'][dim] * (1. / 1000) * jnp.exp(log_incidence)
        varmu = jnp.var(mu, axis=1)
        ax.plot(etas, varmu, label=f'Phi{dim}')
    ax.legend()
    ax.set_xlabel('eta range', fontsize=15)
    ax.set_ylabel('variance of mu', fontsize=15)
    fig.tight_layout()
    plt.close()
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('var_Poissonmu_with_eta' + ".png"))





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
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  eta_plot = jnp.array(config.eta_plot)
  betahp_plot = jnp.array(config.betahp_plot)

  assert eta_plot.ndim == 2
  assert betahp_plot.ndim == 2

  # Plot posterior samples

  if show_posterior_range:
    images = []
    betahps = [ [0.22, 0.84], [1.,1.],[1.93, 1.47],[3.0, 3.0], [5., 5.] ]
    smi_etas = [[1., 1.], [1., 0.0001]]
    colors = ['green', 'black', 'red', 'orange', 'pink']
    dfs = {smi_etas[0][1]:{}, smi_etas[1][1]:{}}
    # dfs = {}
    for eta_ix, smi_eta in enumerate(smi_etas):
        fig_phi, ax_phi = plt.subplots(2,2, figsize=(10, 10), sharex=True)
        ax_phi_flattened = ax_phi.flatten()
        for b_ix, betahp in enumerate(betahps):
            q_distr_out = sample_all_flows(
                params_tuple=[state.params for state in state_list],
                prng_key= next(prng_seq),
                flow_name=config.flow_name,
                flow_kwargs=config.flow_kwargs,
                smi_eta={
                    'modules':
                        jnp.broadcast_to(jnp.array(smi_eta), (config.num_samples_plot,) +
                                        eta_plot.shape[1:])
                },
                betahp=jnp.broadcast_to(jnp.array(betahp),
                                        (config.num_samples_plot,) + betahp_plot.shape[1:]),
            )

            phi = q_distr_out['posterior_sample']['phi']
            theta = q_distr_out['posterior_sample']['theta']
            _, phi_dim = phi.shape

            n_samples, theta_dim = theta.shape
            posterior_samples_df = pd.DataFrame(
                theta, columns=[f"theta_{i}" for i in range(1, theta_dim + 1)])
            posterior_samples_df['eta1'] = f'= {1 if smi_eta[1]==1. else 0}'
            dfs[smi_eta[1]][tuple(betahp)] = posterior_samples_df

            
            # phi plot
            for phi_ix, phi_no in enumerate([7,8,9,12]):
                if betahp == [0.22, 0.84]:
                    sns.kdeplot(phi[:,phi_no], ax=ax_phi_flattened[phi_ix], color='black')
                else:
                    sns.kdeplot(phi[:,phi_no], ax=ax_phi_flattened[phi_ix], color=colors[b_ix], alpha=0.3)
                ax_phi_flattened[phi_ix].set_title(fr'$\phi_{{{phi_no+1}}}$', fontsize=15)

                ax_phi_flattened[phi_ix].xaxis.set_tick_params(which='both', labelbottom=True)

        if eta_ix == 1:
            fig_phi.suptitle(r'$\phi$ posterior distributions for $\eta$ ≈ 0', fontsize=20)
        else:
            fig_phi.suptitle(f'Phi distributions for eta {smi_eta[1]:.3f}', fontsize=20)
        fig_phi.legend([fr'OPT VALUES $c_1$: {betahps[0][0]}, $c_2$: {betahps[0][1]}', 
        fr'$c_1$: {betahps[1][0]},  $c_2$: {betahps[1][1]}', 
        fr'$c_1$: {betahps[2][0]},  $c_2$: {betahps[2][1]}',
        fr'$c_1$: {betahps[3][0]},  $c_2$: {betahps[3][1]}',
        fr'$c_1$: {betahps[4][0]},  $c_2$: {betahps[4][1]}'],
        loc='lower center', ncol=3, fontsize='medium')

        fig_phi.tight_layout()
        fig_phi.subplots_adjust(left=None, bottom=0.15, right=None, top=0.9, wspace=0.2, hspace=0.3)
        
        if workdir_png:
            # fig_theta.savefig(pathlib.Path(workdir_png) / (f'epidemiology_theta_hprange_eta{smi_eta[1]}' + ".png"))
            fig_phi.savefig(pathlib.Path(workdir_png) / (f'epidemiology_phi_hprange_eta{smi_eta[1]}' + ".png"))
        if summary_writer:
            # images.append(plot_to_image(fig_theta))
            images.append(plot_to_image(fig_phi))

    pars = {'alpha': np.clip(100 / n_samples, 0., 1.),
            # 'colour': colour,
        }
    df_main = pd.concat([dfs[smi_etas[0][1]][tuple(betahps[1])],
                             dfs[smi_etas[1][1]][tuple(betahps[1])]])
    grid = sns.JointGrid(
                        x='theta_1',
                        y='theta_2',
                        data=df_main,
                        hue='eta1',
                        xlim=[-3, -1],
                        ylim=[5, 55],
                        height=5)
    g = grid.plot_joint(sns.scatterplot, alpha=pars['alpha'],)
    g.ax_joint.get_legend().set_title(r'$\eta$ values')
    g.ax_joint.set_xlabel(r"$\theta_1$")
    g.ax_joint.set_ylabel(r"$\theta_2$")
    sns.kdeplot(
        dfs[smi_etas[0][1]][tuple(betahps[2])]['theta_1'],
        ax=g.ax_marg_x,
        #legend=False,
        )
    sns.kdeplot(
        dfs[smi_etas[0][1]][tuple(betahps[2])]['theta_2'],
        ax=g.ax_marg_y,
        label=fr'$\eta_1$: 1, $c_1$: {betahps[2][0]}, $c_2$: {betahps[2][1]} ',
        vertical=True,
        )
    
    
    sns.kdeplot(
        dfs[smi_etas[1][1]][tuple(betahps[0])]['theta_1'],
        ax=g.ax_marg_x,
        #legend=False,
        )
    sns.kdeplot(
        dfs[smi_etas[1][1]][tuple(betahps[0])]['theta_2'],
        ax=g.ax_marg_y,
        label=fr'$\eta_1$: 0, $c_1$: {betahps[0][0]}, $c_2$: {betahps[0][1]} ',
        vertical=True,
        )
    
    sns.kdeplot(
        dfs[smi_etas[0][1]][tuple(betahps[3])]['theta_1'],
        ax=g.ax_marg_x,
        alpha=0.3,
        color='#1f77b4',
        #legend=False,
        )
    sns.kdeplot(
        dfs[smi_etas[0][1]][tuple(betahps[3])]['theta_2'],
        ax=g.ax_marg_y,
        label=fr'$\eta_1$: 1, $c_1$: {betahps[3][0]}, $c_2$: {betahps[3][1]} ',
        vertical=True,
        alpha=0.3,
        color='#1f77b4',
        )
        
    sns.kdeplot(
        dfs[smi_etas[1][1]][tuple(betahps[4])]['theta_1'],
        ax=g.ax_marg_x,
        alpha=0.3,
        color='orange',
        #legend=False,
        )
    sns.kdeplot(
        dfs[smi_etas[1][1]][tuple(betahps[4])]['theta_2'],
        ax=g.ax_marg_y,
        label=fr'$\eta_1$: 0, $c_1$: {betahps[4][0]}, $c_2$: {betahps[4][1]} ',
        vertical=True,
        alpha=0.3,
        color='orange',
        )

    # Add title
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(r'Joint SMI posterior for $\theta_1$ and $\theta_2$', fontsize=13)

    plt.legend(loc='upper right')
    # g.ax_marg_y.legend(loc='lower left')
    sns.move_legend(g.ax_joint, "upper left")
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.5, hspace=0.5)

    fig_theta = g.fig
    if workdir_png:
        fig_theta.savefig(pathlib.Path(workdir_png) / (f'epidemiology_theta_hprange_eta{smi_eta[1]}' + ".png"))
    if summary_writer:
        images.append(plot_to_image(fig_theta))

  key_flow = next(prng_seq)
  for i in range(eta_plot.shape[0]):
    # Sample from flow
    q_distr_out = sample_all_flows(
        params_tuple=[state.params for state in state_list],
        prng_key=key_flow,
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        smi_eta={
            'modules':
                jnp.broadcast_to(eta_plot[[i], :], (config.num_samples_plot,) +
                                 eta_plot.shape[1:])
        },
        betahp=jnp.broadcast_to(betahp_plot[[i], :],
                                 (config.num_samples_plot,) + betahp_plot.shape[1:]),
    )

    plot_all.posterior_samples(
        posterior_sample_dict=q_distr_out['posterior_sample'],
        step=state_list[0].step,
        summary_writer=summary_writer,
        eta=eta_plot[i][1],
        betahp=betahp_plot[i],
        workdir_png=workdir_png,
    )


  ### ELPD ###

  # Define elements to grate grid of eta values

  eta_base = np.array([1., 0., 1., 1.])
  eta_grid_base = np.tile(eta_base, [eta_grid_len + 1, eta_grid_len + 1, 1]).T


  if show_elpd:

    images = []
    prng_key_elpd = next(prng_seq)

    ########### vary eta_1 only
    eta_grid_mini = np.stack(
    np.meshgrid(
        np.linspace(0., 1., eta_grid_len + 1),
        np.linspace(1., 1., eta_grid_len + 1)),
    axis=0)
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [1, 2]
    eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

    elpd_surface_dict = elpd_surface_points(
        state_list=state_list,
        batch=batch, # train_ds
        prng_key=prng_key_elpd,
        config=config,
        eta_grid=eta_grid,
    )

    fig, axs = plot_all.plot_elpd_one_variable(elpd_surface_dict=elpd_surface_dict,
        suptitle='Eta1, Beta prior hyperparameters fixed to 1',
        xlabel='eta_1',
        x_values=eta_grid[eta_grid_x_y_idx[0]][0],
        indx=0,)

    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('elpd_eta1_fixedhp' + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    ########### vary conc1 only, eta fixed to 0, conc2 to 1
    eta_grid_mini = np.stack(
    np.meshgrid(
        np.linspace(0., 0., eta_grid_len + 1),
        np.linspace(0.00001, 2., eta_grid_len + 1)),
    axis=0)
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [1, 2]
    eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

    elpd_surface_dict = elpd_surface_points(
        state_list=state_list,
        batch=batch, # train_ds
        prng_key=prng_key_elpd,
        config=config,
        eta_grid=eta_grid,
    )

    fig, axs = plot_all.plot_elpd_one_variable(elpd_surface_dict=elpd_surface_dict,
            suptitle='Beta prior alpha hyperparameter, eta1 fixed to 0 and beta hp fixed to 1',
            xlabel='conc1',
            x_values=eta_grid[eta_grid_x_y_idx[1]][:,0],
            indx=1,)

    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('elpd_conc1_fixedeta0conc2' + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    ########### vary conc2 only, eta fixed to 0, conc1 to 1
    eta_grid_mini = np.stack(
    np.meshgrid(
        np.linspace(0., 0., eta_grid_len + 1),
        np.linspace(0.00001, 2., eta_grid_len + 1)),
    axis=0)
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [1, 3]
    eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

    elpd_surface_dict = elpd_surface_points(
        state_list=state_list,
        batch=batch, # train_ds
        prng_key=prng_key_elpd,
        config=config,
        eta_grid=eta_grid,
    )

    fig, axs = plot_all.plot_elpd_one_variable(elpd_surface_dict=elpd_surface_dict,
            suptitle='Beta prior beta hyperparameter, eta1 fixed to 0 and alpha hp fixed to 1',
            xlabel='conc1',
            x_values=eta_grid[eta_grid_x_y_idx[1]][:,0],
            indx=1,)

    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('elpd_conc2_fixedeta0conc1' + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    ########### vary conc1 only, eta fixed to 1, conc2 to 1
    eta_grid_mini = np.stack(
    np.meshgrid(
        np.linspace(1., 1., eta_grid_len + 1),
        np.linspace(0.00001, 2., eta_grid_len + 1)),
    axis=0)
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [1, 2]
    eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

    elpd_surface_dict = elpd_surface_points(
        state_list=state_list,
        batch=batch, # train_ds
        prng_key=prng_key_elpd,
        config=config,
        eta_grid=eta_grid,
    )

    fig, axs = plot_all.plot_elpd_one_variable(elpd_surface_dict=elpd_surface_dict,
            suptitle='Beta prior alpha hyperparameter, eta1 fixed to 1 and beta hp fixed to 1',
            xlabel='conc1',
            x_values=eta_grid[eta_grid_x_y_idx[1]][:,0],
            indx=1,)

    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('elpd_conc1_fixedeta1conc2' + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    ########### vary conc2 only, eta fixed to 1, conc1 to 1
    eta_grid_mini = np.stack(
    np.meshgrid(
        np.linspace(1., 1., eta_grid_len + 1),
        np.linspace(0.00001, 2., eta_grid_len + 1)),
    axis=0)
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [1, 3]
    eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

    elpd_surface_dict = elpd_surface_points(
        state_list=state_list,
        batch=batch, # train_ds
        prng_key=prng_key_elpd,
        config=config,
        eta_grid=eta_grid,
    )

    fig, axs = plot_all.plot_elpd_one_variable(elpd_surface_dict=elpd_surface_dict,
            suptitle='Beta prior beta hyperparameter, eta1 fixed to 1 and alpha hp fixed to 1',
            xlabel='conc1',
            x_values=eta_grid[eta_grid_x_y_idx[1]][:,0],
            indx=1,)

    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('elpd_conc2_fixedeta1conc1' + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))


    ############ Vary eta_1 and conc1
    eta_grid_mini = np.stack(
    np.meshgrid(
        np.linspace(0., 1., eta_grid_len + 1),
        np.linspace(0.00001, 2., eta_grid_len + 1)),
    axis=0)
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [1, 2]
    eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

    elpd_surface_dict = elpd_surface_points(
        state_list=state_list,
        batch=batch, # train_ds
        prng_key=prng_key_elpd,
        config=config,
        eta_grid=eta_grid,
    )
    
    # Plot the ELPD surface.
    fig, axs = plt.subplots(
        nrows=3, ncols=2, figsize=(2 * 3, 3*3), subplot_kw={"projection": "3d"})

    for mod_ix, mod_name in enumerate(['', '_y', '_z']):
        for i, metric in enumerate([elpd_surface_dict[f'lpd{mod_name}_all_eta'],
                                    elpd_surface_dict[f'elpd_waic{mod_name}_all_eta'],]):
            axs[mod_ix,i].set_title([f"Full likelihood \n {['- LPD', '- ELPD WAIC'][i]}", 
                                     f"Y module\n {['- LPD', '- ELPD WAIC'][i]}", 
                                     f"Z module\n {['- LPD', '- ELPD WAIC'][i]}"][mod_ix])
            axs[mod_ix,i].plot_surface(
                eta_grid[eta_grid_x_y_idx[0]],
                eta_grid[eta_grid_x_y_idx[1]],
                -metric,
                cmap=matplotlib.cm.inferno,
                # linewidth=0,
                # antialiased=False,
            )
            axs[mod_ix,i].view_init(30, 225)
            axs[mod_ix,i].set_xlabel(f'eta_{eta_grid_x_y_idx[0]}')
            axs[mod_ix,i].set_ylabel(f'conc_{eta_grid_x_y_idx[1]-1}')

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0, hspace=0.6)


    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('elpd_surface_eta_conc1' + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    ############ Vary eta_1 and conc2
    eta_grid_mini = np.stack(
        np.meshgrid(
            np.linspace(0., 1., eta_grid_len + 1),
            np.linspace(0.00001, 2., eta_grid_len + 1)),
        axis=0)
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [1, 3]
    eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

    # Plot the ELPD surface.
    fig, axs = plt.subplots(
        nrows=3, ncols=2, figsize=(2 * 3, 3*3), subplot_kw={"projection": "3d"})

    for mod_ix, mod_name in enumerate(['', '_y', '_z']):
        for i, metric in enumerate([elpd_surface_dict[f'lpd{mod_name}_all_eta'],
                                    elpd_surface_dict[f'elpd_waic{mod_name}_all_eta'],]):
            axs[mod_ix,i].set_title([f"Full likelihood \n {['- LPD', '- ELPD WAIC'][i]}", 
                                     f"Y module\n {['- LPD', '- ELPD WAIC'][i]}", 
                                     f"Z module\n {['- LPD', '- ELPD WAIC'][i]}"][mod_ix])
            axs[mod_ix,i].plot_surface(
                eta_grid[eta_grid_x_y_idx[0]],
                eta_grid[eta_grid_x_y_idx[1]],
                -metric,
                cmap=matplotlib.cm.inferno,
                # linewidth=0,
                # antialiased=False,
            )
            axs[mod_ix,i].view_init(30, 225)
            axs[mod_ix,i].set_xlabel(f'eta_{eta_grid_x_y_idx[0]}')
            axs[mod_ix,i].set_ylabel(f'conc_{eta_grid_x_y_idx[1]-1}')
            # axs[mod_ix,i].set_zlabel(['- LPD', '- ELPD WAIC'][i])

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0, hspace=0.6)

    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('elpd_surface_eta_conc2' + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))


    ############ Vary conc_1 and conc2
    # eta_base = np.array([1., 0.3, 1., 1.])
    # eta_grid_base = np.tile(eta_base, [eta_grid_len + 1, eta_grid_len + 1, 1]).T

    eta_grid_mini = np.stack(
    np.meshgrid(
        np.linspace(0.00001, 2., eta_grid_len + 1),
        np.linspace(0.00001, 2., eta_grid_len + 1)),
    axis=0)
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [2, 3]
    eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

    elpd_surface_dict = elpd_surface_points(
        state_list=state_list,
        batch=batch, # train_ds
        prng_key=prng_key_elpd,
        config=config,
        eta_grid=eta_grid,
    )
    
    # Plot the ELPD surface.
    fig, axs = plt.subplots(
        nrows=3, ncols=2, figsize=(2 * 3, 3*3), subplot_kw={"projection": "3d"})

    for mod_ix, mod_name in enumerate(['', '_y', '_z']):
        for i, metric in enumerate([elpd_surface_dict[f'lpd{mod_name}_all_eta'],
                                    elpd_surface_dict[f'elpd_waic{mod_name}_all_eta'],]):
            axs[mod_ix,i].set_title([f"Full likelihood \n {['- LPD', '- ELPD WAIC'][i]}", 
                                     f"Y module\n {['- LPD', '- ELPD WAIC'][i]}", 
                                     f"Z module\n {['- LPD', '- ELPD WAIC'][i]}"][mod_ix])
            axs[mod_ix,i].plot_surface(
                eta_grid[eta_grid_x_y_idx[0]],
                eta_grid[eta_grid_x_y_idx[1]],
                -metric,
                cmap=matplotlib.cm.inferno,
                # linewidth=0,
                # antialiased=False,
            )
            axs[mod_ix,i].view_init(30, 225)
            axs[mod_ix,i].set_xlabel(f'conc_{eta_grid_x_y_idx[0]-1}')
            axs[mod_ix,i].set_ylabel(f'conc_{eta_grid_x_y_idx[1]-1}')

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0, hspace=0.6)


    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('elpd_surface_conc1_conc2' + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    if summary_writer:
      plot_name = 'rnd_eff_elpd_surface'
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=state_list[0].step,
      )

########################################################################################################################

def train_and_evaluate(config: ConfigDict, workdir: str) -> TrainState:
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
              'eta': jnp.ones((config.num_samples_elbo, 2)),
              'betahp': jnp.ones((config.num_samples_elbo, 2)), # init vals right?
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
      eta=jnp.ones((config.num_samples_elbo, 2)),
      betahp=jnp.ones((config.num_samples_elbo, 2)),
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
              'eta': jnp.ones((config.num_samples_elbo, 2)),
              'betahp': jnp.ones((config.num_samples_elbo, 2)), # init vals right?
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
                'eta': jnp.ones((config.num_samples_elbo, 2)),
                'betahp': jnp.ones((config.num_samples_elbo, 2)), # init vals right?
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
          eta=jnp.ones((config.num_samples_elbo, 2)),
          betahp=jnp.ones((config.num_samples_elbo, 2)),
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
          eta=jnp.ones((config.num_samples_elbo, 2)),
          betahp=jnp.ones((config.num_samples_elbo, 2)),
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
      eta_sampling_a=1.,
      eta_sampling_b=1.,
      betahp_sampling_a=1.,
      betahp_sampling_b=1.,
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
      loss_fn=loss_neg_elpd_y,
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
          show_posterior_range=False,
          eta_grid_len=20,
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

    # SGD step
    state_list, metrics = update_states_jit(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
    )
    # The computed training loss corresponds to the model before update
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

    # Wait until computations are done before the next step
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info('Final training step: %i', state_list[0].step)

#########################################################################
  ## Find best eta ###

#   logging.info('Finding best hyperparameters...')

  

#   # Reset random key sequence
#   prng_seq = hk.PRNGSequence(config.seed)

#   # Initialize search with Bayes
#   hp_star = jnp.array([1., 1., 1., 1.])
#   info_dict = {'init':hp_star, 'likelihood':'y',
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
#     info_dict['loss'].append(neg_elpd["train_loss"])
#     info_dict['params'].append(hp_star_state.params)
#     info_dict['step'].append(hp_star_state.step)

#     if hp_star_state.step % 100 == 0:
#       logging.info("STEP: %5d; training loss: %.3f; eta0:%.3f; eta1: %.3f; conc1: %.3f, conc2:%.3f", hp_star_state.step,
#                    neg_elpd["train_loss"], hp_star_state.params[0], hp_star_state.params[1],
#                    hp_star_state.params[2], hp_star_state.params[3])

#     # Clip eta_star to [0,1] hypercube and hp_star to [0.000001,..]
#     hp_star_state = TrainState(
#         params=jnp.hstack([jnp.clip(hp_star_state.params[:2],0, 1),
#                            jnp.clip(hp_star_state.params[2:],0.000001)]),
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
#   with open(workdir + f"/hp_info_eta{hp_star[1]:.6f}_{info_dict['likelihood']}.sav", 'wb') as f:
#     pickle.dump(info_dict, f)

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
#   posterior_sample_variance(
#     state_list=state_list,
#     batch=train_ds,
#     prng_key=next(prng_seq),
#     config=config,
#     workdir_png=workdir,
#     )

  # Last plot of posteriors
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
  )

  return state_list
