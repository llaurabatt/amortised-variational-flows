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
from log_prob_fun_integrated import PriorHparams, sample_priorhparams_values

import plot_all
from train_flow import load_dataset, make_optimizer

from modularbayes._src.utils.training import TrainState
from modularbayes import (plot_to_image, normalize_images, flatten_dict, initial_state_ckpt, 
update_state, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, Callable, ConfigDict, Dict, List,
                                      NamedTuple, Optional, PRNGKey, SmiEta, SummaryWriter,
                                      Tuple, Mapping)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)

def make_optimizer_eta(learning_rate: float) -> optax.GradientTransformation:
  optimizer = optax.adabelief(learning_rate=learning_rate)
  return optimizer


def q_distr_mu_sigma(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    cond_values: Array,
    num_samples: int,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  # Define normalizing flows
  q_distr = getattr(flows_all_integrated, flow_name + '_mu_sigma')(**flow_kwargs)


  # Sample from flows
  (mu_sigma_sample, mu_sigma_log_prob_posterior) = q_distr.sample_and_log_prob(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=[cond_values, None],
   )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows_all_integrated.split_flow_mu_sigma(
          samples=mu_sigma_sample,
          **flow_kwargs,
      ))

  # log P(mu, sigma)
  q_distr_out['mu_sigma_log_prob'] = mu_sigma_log_prob_posterior

  return q_distr_out



def sample_all_flows(
    params_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    cond_values: Array,
    num_samples: int,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  prng_seq = hk.PRNGSequence(prng_key)

  # mu
  q_distr_out = hk.transform(q_distr_mu_sigma).apply(
      params_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      cond_values=cond_values,
      num_samples=num_samples,
  )

  return q_distr_out

def get_cond_values(
    cond_hparams_names: List,
    num_samples: float,
    prior_hparams_init: Optional[NamedTuple],
    ):

  cond_prior_hparams_names = cond_hparams_names.copy()

  cond_prior_hparams_init = []
  for k in prior_hparams_init._fields:
        if k in cond_prior_hparams_names:
            val = getattr(prior_hparams_init, k)
            sample_val = jnp.ones((num_samples,))*val
            cond_prior_hparams_init.append(sample_val[:,None])
  cond_prior_hparams_init = jnp.hstack(cond_prior_hparams_init)

  return cond_prior_hparams_init

def elbo_estimate_along_eta(
    params_tuple: Tuple[hk.Params],
    batch: Batch,
    prng_key: PRNGKey,
    mask_Y: Array,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    cond_hparams: List,
    sample_priorhparams_fn:Callable,
    sample_priorhparams_kwargs: Dict[str, Any],
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  prior_hparams_sample, cond_prior_hparams_values = sample_priorhparams_fn(
      prng_seq=prng_seq,
      num_samples=num_samples, 
      cond_hparams=cond_hparams,
      **sample_priorhparams_kwargs,
  )


  cond_values = cond_prior_hparams_values

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      prng_key=next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      cond_values=cond_values,
      num_samples=num_samples,
  )

  shared_params_names = [
      'mu',
      'sigma',
  ]

  # ELBO :  posterior
  posterior_sample_dict = {}
  for key in shared_params_names:
    posterior_sample_dict[key] = q_distr_out['posterior_sample'][key]



  log_prob_joint = jax.vmap(
      lambda posterior_sample_i, prior_hparams_i: log_prob_fun_integrated.log_prob_joint(
          batch=batch,
          posterior_sample_dict=posterior_sample_i,
          mask_Y=mask_Y.reshape(batch.shape),
          prior_hparams=prior_hparams_i
      ))(
          jax.tree_map(lambda x: jnp.expand_dims(x, 1),
                       posterior_sample_dict),
          prior_hparams_sample,                   
    )

  log_q = (q_distr_out['mu_sigma_log_prob'])

  # TODO: check reshape
  elbo = log_prob_joint['log_prob'].reshape(-1) - log_q

  elbo_dict = {'elbo': elbo, 
               'log_prob_joint': log_prob_joint['log_prob'].reshape(-1),
               'log_lik':log_prob_joint['log_lik'], 
               'log_mu':log_prob_joint['log_mu'],
               'log_sigma':log_prob_joint['log_sigma'],
               'neg_log_q': -log_q,
               'cond_values':cond_prior_hparams_values}

  return elbo_dict


def loss(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_estimate_along_eta(
      params_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['elbo']))

  return loss_avg


########################################################################################################################
def compute_lpd(
    posterior_sample_dict: Dict[str, Any],
    batch: Batch,
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

  num_samples, mu_dim = posterior_sample_dict['mu'].shape 
  num_obs = batch.shape[0]

  ### WAIC ###
#   # Compute LPD

  loglik_pointwise_insample = log_prob_fun_integrated.log_lik_vectorised(
      mask_neg_Y.reshape(num_obs, mu_dim),
      batch,
      posterior_sample_dict['mu'],
      posterior_sample_dict['sigma'],
  ).sum(2)

  lpd_pointwise = jax.scipy.special.logsumexp(
      loglik_pointwise_insample, axis=0) - jnp.log(num_samples)
  lpd_out['lpd_pointwise'] = lpd_pointwise

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
    mask_neg_Y:Array,
    # eta: Array,
    # betahp: Array,
):
  cond_values = hp_params 
#   assert len(hp_params) == 4
  q_distr_out_i = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=prng_key,  # same key to reduce variance of posterior along eta
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      cond_values=jnp.broadcast_to(cond_values, (num_samples, len(cond_values))),
      num_samples=num_samples,
  )

  lpd_dict = compute_lpd_jit(
      posterior_sample_dict=q_distr_out_i['posterior_sample'],
      batch=batch,
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
    
    n_obs, mu_dim = batch_predict.shape
    all_mask_neg_Y = jnp.eye((n_obs*mu_dim))
    elpd_loocv_fixed_hp_params = pytrees_vmap(lambda state_list, key, mask_neg_Y:  lpd_estimate_pointwise(
        hp_params=hp_params,
        state_list=state_list,
        batch=batch_predict,
        prng_key=key,
        flow_name=flow_name,
        flow_kwargs=flow_kwargs,
        num_samples=num_samples,
        mask_neg_Y=mask_neg_Y,
        ))
    
    elpd_loocv_fixed_hp_params_jit = jax.jit(elpd_loocv_fixed_hp_params)

    n_obs = batch_predict['Y_g'].shape[0]
    keys = jax.random.split(
      prng_key, n_obs)

    return elpd_loocv_fixed_hp_params_jit(states_lists, keys, all_mask_neg_Y)
########################################################################################################################

def elbo_estimate(
    hp_params: Array,
    hp_optim_mask_indices:Tuple,
    hp_fixed_values:Array,
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    mask_Y: Array,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  hp_fixed_values = jnp.array(hp_fixed_values)
  hp_params_all = jnp.zeros(len(hp_optim_mask_indices[0])+ len(hp_optim_mask_indices[1]))#jnp.zeros(optim_mask.shape)
  hp_params_all = hp_params_all.at[(hp_optim_mask_indices[1],)].set(hp_params)
  hp_params_all = hp_params_all.at[(hp_optim_mask_indices[0],)].set(hp_fixed_values)

        
  cond_values = hp_params

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      cond_values=jnp.broadcast_to(cond_values, (num_samples, len(cond_values))),
      num_samples=num_samples,
  )

  shared_params_names = [
      'mu',
      'sigma',
  ]

  # ELBO :  posterior
  posterior_sample_dict = {}
  for key in shared_params_names:
    posterior_sample_dict[key] = q_distr_out['posterior_sample'][key]


  hp_params_all_samples = jnp.broadcast_to(hp_params_all, (num_samples, len(hp_params_all)))
  log_prob_joint = jax.vmap(
      lambda posterior_sample_i, prior_hparams_i: log_prob_fun_integrated.log_prob_joint(
          batch=batch,
          posterior_sample_dict=posterior_sample_i,
          mask_Y=mask_Y.reshape(batch.shape),
          prior_hparams=prior_hparams_i
      ))(
          jax.tree_map(lambda x: jnp.expand_dims(x, 1),
                       posterior_sample_dict),
          PriorHparams(*jnp.split(hp_params_all_samples, hp_params_all_samples.shape[1], axis=1)),                   
    )

  log_q = (q_distr_out['mu_sigma_log_prob'])

  # TODO: check reshape
  elbo = log_prob_joint['log_prob'].reshape(-1) - log_q

  elbo_dict = {'elbo': elbo, 
               'log_prob_joint': log_prob_joint['log_prob'].reshape(-1),
               'log_lik':log_prob_joint['log_lik'], 
               'log_mu':log_prob_joint['log_mu'],
               'log_sigma':log_prob_joint['log_sigma'],
               'neg_log_q': -log_q}

  return elbo_dict


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

  num_samples, mu_dim = posterior_sample_dict['mu'].shape 
  n_obs = batch.shape[0]

  ### WAIC ###
#   # Compute LPD

  loglik_pointwise_insample = log_prob_fun_integrated.log_lik_vectorised(
      jnp.ones((n_obs, mu_dim)),
      batch,
      posterior_sample_dict['mu'],
      posterior_sample_dict['sigma'],
  ).sum(2)

  lpd_pointwise = jax.scipy.special.logsumexp(
      loglik_pointwise_insample, axis=0) - jnp.log(num_samples)
  elpd_out['lpd_pointwise'] = lpd_pointwise

  p_waic_pointwise = jnp.var(loglik_pointwise_insample, axis=0)
  elpd_out['p_waic_pointwise'] = p_waic_pointwise

  elpd_waic_pointwise = lpd_pointwise - p_waic_pointwise
  elpd_out['elpd_waic_pointwise'] = elpd_waic_pointwise

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

  cond_values = hp_params 
#   assert len(hp_params) == 4
  q_distr_out_i = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=prng_key,  # same key to reduce variance of posterior along eta
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      cond_values=jnp.broadcast_to(cond_values, (num_samples, len(cond_values))),
      num_samples=num_samples,
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

    lpd_pointwise_all_eta = jnp.stack(lpd_pointwise_all_eta, axis=0)
    p_waic_pointwise_all_eta = jnp.stack(p_waic_pointwise_all_eta, axis=0)
    elpd_waic_pointwise_all_eta = jnp.stack(elpd_waic_pointwise_all_eta, axis=0)

  # Add pointwise elpd and lpd across observations
  lpd_all_eta = lpd_pointwise_all_eta.sum(axis=-1).reshape(grid_shape)
  p_waic_all_eta = p_waic_pointwise_all_eta.sum(axis=-1).reshape(grid_shape)
  elpd_waic_all_eta = elpd_waic_pointwise_all_eta.sum(
      axis=-1).reshape(grid_shape)


  elpd_surface_dict = {'lpd_all_eta':lpd_all_eta,'p_waic_all_eta':p_waic_all_eta,
  'elpd_waic_all_eta':elpd_waic_all_eta}

  return elpd_surface_dict


def log_images(
    state_lists: Dict,
    true_params: Dict[str, Any],
    prng_key: PRNGKey,
    config: ConfigDict,
    summary_writer: Optional[SummaryWriter],
    workdir_png: Optional[str],
    mcmc_samples: Optional[Dict[str, Any]] = {'mu': None, 'sigma': None},
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  if config.checkpoint_dir_comparison.true:              

    q_distr_out_truehparams = sample_all_flows(
        params_tuple=tuple(state.params for state in state_lists['true']),
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        cond_values=None, #cond_values_true,
        num_samples=config.num_samples_eval,
    )

    mu_samples_truehparams = q_distr_out_truehparams['posterior_sample']['mu']
    sigma_samples_truehparams = q_distr_out_truehparams['posterior_sample']['sigma']
    mu_dim = mu_samples_truehparams.shape[1]

    assert mu_samples_truehparams.shape == sigma_samples_truehparams.shape == (config.num_samples_eval, mu_dim)

  if config.opt_cond_hparams_values:
    assert len(config.cond_hparams_names) == len(config.opt_cond_hparams_values)
    cond_values_opt = get_cond_values(cond_hparams_names=config.cond_hparams_names,
                        num_samples=config.num_samples_eval,
                        prior_hparams_init=PriorHparams(*config.opt_cond_hparams_values),
                        )
    
    q_distr_out_opthparams = sample_all_flows(
        params_tuple=tuple(state.params for state in state_lists['opt']),
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        cond_values=cond_values_opt,
        num_samples=config.num_samples_eval,
    )
    mu_samples_opthparams = q_distr_out_opthparams['posterior_sample']['mu']
    sigma_samples_opthparams = q_distr_out_opthparams['posterior_sample']['sigma']
    assert mu_samples_opthparams.shape == sigma_samples_opthparams.shape == (config.num_samples_eval, mu_dim)
    
  else:
    mu_samples_opthparams = sigma_samples_opthparams = None

  if config.alternative_cond_hparams_values:
    assert len(config.cond_hparams_names) == len(config.alternative_cond_hparams_values)
    chkpt = config.checkpoint_dir_comparison.to_dict()['alternative']
    cond_values_alternative = get_cond_values(cond_hparams_names=config.cond_hparams_names,
                        num_samples=config.num_samples_eval,
                        prior_hparams_init=PriorHparams(*config.alternative_cond_hparams_values),
                        )
    cond_values = None if 'VP' in chkpt else cond_values_alternative
    
    q_distr_out_alternativehparams = sample_all_flows(
        params_tuple=tuple(state.params for state in state_lists['alternative']),
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        cond_values=cond_values,
        num_samples=config.num_samples_eval,
    )
    mu_samples_alternativehparams = q_distr_out_alternativehparams['posterior_sample']['mu']
    sigma_samples_alternativehparams = q_distr_out_alternativehparams['posterior_sample']['sigma']
    assert mu_samples_alternativehparams.shape == sigma_samples_alternativehparams.shape == (config.num_samples_eval, mu_dim)
    
  else:
    mu_samples_alternativehparams = sigma_samples_alternativehparams = None

  for par_name, par_samples in {'\mu':{'true_hparams':mu_samples_truehparams, 'opt_hparams':mu_samples_opthparams, 'alternative_hparams':mu_samples_alternativehparams,
                                       'mcmc_true_hparams':mcmc_samples['mu']}, 
                                    '\sigma':{'true_hparams':sigma_samples_truehparams, 'opt_hparams':sigma_samples_opthparams, 'alternative_hparams':sigma_samples_alternativehparams,
                                              'mcmc_true_hparams':mcmc_samples['sigma']},
                                    }.items():
    fig = plot_all.plot_final_posterior_vs_true(par_samples=par_samples['true_hparams'], 
                                                compare_samples=par_samples['opt_hparams'],
                                                alternative_samples=par_samples['alternative_hparams'],
                                                mcmc_samples=par_samples['mcmc_true_hparams'],
                                                par_name_plot=par_name, 
                                                true_params=true_params)
    fig.savefig(pathlib.Path(workdir_png) / (f"posterior_vs_true_{par_name[1:]}{'_hparamcompare' if par_samples['opt_hparams'] is not None else ''}{'_addalternative' if par_samples['alternative_hparams'] is not None else ''}" + ".png"))
    images = [plot_to_image(fig)]
    plot_name = fr'posterior_vs_true_\{par_name}'
    summary_writer.image(
        tag=plot_name,
        image=normalize_images(images),
        step=state_lists['save'][0].step,
        )




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
  train_ds, true_params, sim_data_fig = load_dataset(n_groups=config.synth_n_groups, 
                                       n_obs=config.synth_n_obs,
                                       seed=config.seed_synth)
  true_params_filename = f'true_params_SEED{config.seed_synth}_{config.synth_n_obs}obs_{config.synth_n_groups}groups.pickle'
  true_params_file_path = os.path.join(workdir, true_params_filename)
  with open(true_params_file_path, 'wb') as f:
     pickle.dump(true_params, f)

  sim_data_filename = f'sim_data_summary_SEED{config.seed_synth}_{config.synth_n_obs}obs_{config.synth_n_groups}groups.png'
  sim_data_file_path = os.path.join(workdir, sim_data_filename)
  sim_data_fig.savefig(sim_data_file_path)


  mu_dim = sigma_dim = train_ds.shape[1] #len(train_ds.keys())

  # mu_dim and theta_dim are also arguments of the flow,
  # as they define its dimension
  config.flow_kwargs.mu_dim = mu_dim
  config.flow_kwargs.sigma_dim = sigma_dim
  config.flow_kwargs.num_groups = config.synth_n_groups

  new_defaults = {'mu_prior_mean_m': config.prior_hparams.mu_prior_mean_m, 
            'mu_prior_scale_s': config.prior_hparams.mu_prior_scale_s, 
            'sigma_prior_concentration': config.prior_hparams.sigma_prior_concentration, 
            'sigma_prior_scale': config.prior_hparams.sigma_prior_scale, 
            }

  PriorHparams.set_defaults(**new_defaults)

  prior_hparams_init = PriorHparams()

  if config.cond_hparams_names:
    cond_values_init = get_cond_values(cond_hparams_names=config.cond_hparams_names,
                        num_samples=config.num_samples_elbo,
                        prior_hparams_init=prior_hparams_init
                        )
  else:
    cond_values_init = None

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)


  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state_list = []
  state_name_list = []

  state_name_list.append('mu_sigma')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(q_distr_mu_sigma),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'cond_values': cond_values_init, 
              'num_samples':config.num_samples_elbo,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # Print a useful summary of the execution of the flow architecture.
  logging.info('FLOW MU and SIGMA:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: hk.transform(q_distr_mu_sigma).apply(
          params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          cond_values=cond_values_init,
          num_samples=config.num_samples_elbo,
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

  # Jit function to update training states
  update_states_jit = lambda state_list, batch, prng_key: update_states(
      state_list=state_list,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer(**config.optim_kwargs),
      loss_fn=loss,
      loss_fn_kwargs={
          'mask_Y': jnp.array(config.mask_Y),
          'num_samples': config.num_samples_elbo,
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'cond_hparams':config.cond_hparams_names,
          'sample_priorhparams_fn': sample_priorhparams_values,
          'sample_priorhparams_kwargs': config.prior_hparams_hparams,
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
      cond_hparams=config.cond_hparams_names,
      sample_priorhparams_fn=sample_priorhparams_values,
      sample_priorhparams_kwargs=config.prior_hparams_hparams,
      mask_Y=jnp.array(config.mask_Y),
  )
  elbo_validation_jit = jax.jit(elbo_validation_jit)

  ############################################################################################################################
  # LOOCV as a loss function to optimize prior hparams

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
  )['elpd_waic_pointwise'].sum(axis=-1) # sum over n_obs
  loss_neg_elpd = jax.jit(loss_neg_elpd)

  # elbo as a loss function to optimize eta
  loss_neg_elbo = lambda hp_params, hp_optim_mask_indices, hp_fixed_values, batch, prng_key, state_list_vmp: -elbo_estimate(
    hp_params=hp_params,
    hp_optim_mask_indices=hp_optim_mask_indices,
    hp_fixed_values=hp_fixed_values,
    state_list=state_list_vmp,
    batch=batch,
    prng_key=prng_key,
    mask_Y=config.mask_Y,
    flow_name=config.flow_name,
    flow_kwargs=config.flow_kwargs,
    num_samples=config.num_samples_elbo_optim,
  )['elbo'].mean() # mean over n_obs
  loss_neg_elbo = jax.jit(loss_neg_elbo)

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

  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))
  else:
    summary_writer = None

  if state_list[0].step < config.training_steps:
    save_time_info = True
    save_after_training = True
    logging.info('Training Variational Meta-Posterior (VMP-flow)...')

    # Reset random key sequence
    prng_seq = hk.PRNGSequence(config.seed)



  while state_list[0].step < config.training_steps:
    

    # Plots to monitor training
    # if ((state_list[0].step == 0) or
    #     (state_list[0].step % config.log_img_steps == 0)):
    #   # print("Logging images...\n")
    #   log_images(
    #       state_list=state_list,
    #       batch=train_ds,
    #       prng_key=next(prng_seq),
    #       config=config,
    #       show_elpd=False,
    #       show_posterior_range_allhps=False,
    #       eta_grid_len=20,
    #       summary_writer=summary_writer,
    #       workdir_png=workdir,
    #     #   workdir_mcmc=workdir_mcmc,
    #   )

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


      if metrics["train_loss"]>100000:
          log_mus = elbo_validation_dict['log_mu']
          idx = jnp.where(log_mus<-10000)[0]

          jax.debug.print(str(len(idx)))
          jax.debug.print(str(log_mus[idx]))
          jax.debug.print(str(elbo_validation_dict['cond_values'][idx,:]))
      for k, v in elbo_validation_dict.items():
        if v is None:
            continue
        else:
            summary_writer.scalar(
                tag=f'elbo_{k}',
                value=v.mean(),
                step=state_list[0].step,
            )

      # track posteriors of parameters mu and sigma
      if config.cond_hparams_names:
            cond_values_init = get_cond_values(cond_hparams_names=config.cond_hparams_names,
                                num_samples=config.num_samples_eval,
                                prior_hparams_init=prior_hparams_init
                                )
      else:
            cond_values_init = None        
      q_distr_out = sample_all_flows(
            params_tuple=tuple(state.params for state in state_list),
            prng_key=next(prng_seq),
            flow_name=config.flow_name,
            flow_kwargs=config.flow_kwargs,
            cond_values=cond_values_init,
            num_samples=config.num_samples_eval,
        )
      mu_samples = q_distr_out['posterior_sample']['mu']
      sigma_samples = q_distr_out['posterior_sample']['sigma']
      assert mu_samples.shape == sigma_samples.shape == (config.num_samples_eval, mu_dim)
      
      for dim in range(mu_dim):
            summary_writer.scalar(
                tag=f'true_mu_{dim}',
                value=true_params['mu'][dim],
                step=state_list[0].step,
            )

            summary_writer.scalar(
                tag=f'true_sigma_{dim}',
                value=true_params['sigma'][dim],
                step=state_list[0].step,
            )

            summary_writer.histogram(f"mu_coordinate_{dim}_distribution", mu_samples[:, dim], step=state_list[0].step)
            summary_writer.histogram(f"sigma_coordinate_{dim}_distribution", sigma_samples[:, dim], step=state_list[0].step)



        

    if state_list[0].step % config.checkpoint_steps == 0:
      for state_i, state_name_i in zip(state_list, state_name_list):
        save_checkpoint(
            state=state_i,
            checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
            keep=config.checkpoints_keep,
        )
      
      for par_name, par_samples in {'\mu':mu_samples, '\sigma': sigma_samples}.items():
        fig = plot_all.plot_final_posterior_vs_true(par_samples=par_samples, par_name_plot=par_name, true_params=true_params)
        fig.savefig(pathlib.Path(workdir) / (f"posterior_vs_true_{par_name[1:]}" + ".png"))
        images = [plot_to_image(fig)]
        plot_name = fr'posterior_vs_true_{par_name[1:]}'
        summary_writer.image(
            tag=plot_name,
            image=normalize_images(images),
            step=state_list[0].step,
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

  # Last plot of posteriors
  
  if config.log_img_steps != np.inf: 
    state_lists = {'true':[], 'opt':[], 'save':[]} #, 'alternative':[]}
    for state_list_name, v in state_lists.items():
        if state_list_name != 'save':
            chkpt = config.checkpoint_dir_comparison.to_dict()[state_list_name]
            if chkpt:
                if chkpt == checkpoint_dir:
                    state_lists[state_list_name] = state_list
                    state_lists['save'] = state_list
                else:
                    if 'VP' in chkpt:
                        cond_values_init = None 
                    else:
                        cond_values_init = get_cond_values(cond_hparams_names=['mu_prior_mean_m', 'mu_prior_scale_s', 'sigma_prior_concentration', 'sigma_prior_scale'],
                                            num_samples=config.num_samples_eval,
                                            prior_hparams_init=prior_hparams_init
                                            )    
                    state_lists[state_list_name].append(
                        initial_state_ckpt(
                            checkpoint_dir=chkpt + '/mu_sigma',
                            forward_fn=hk.transform(q_distr_mu_sigma),
                            forward_fn_kwargs={
                                'flow_name': config.flow_name,
                                'flow_kwargs': config.flow_kwargs,
                                'cond_values': cond_values_init, 
                                'num_samples':config.num_samples_eval,
                            },
                            prng_key=next(prng_seq),
                            optimizer=make_optimizer(**config.optim_kwargs),
                        ))

    if config.mcmc_samples_true_hparams_path:
        with open(config.mcmc_samples_true_hparams_path, 'rb') as f:
            mcmc_samples = pickle.load(f)
    else:
        mcmc_samples = {'mu': None, 'sigma': None}
        
    log_images(
        state_lists=state_lists,
        true_params=true_params,
        prng_key=next(prng_seq),
        config=config,
        summary_writer=summary_writer,
        workdir_png=workdir,
        mcmc_samples=mcmc_samples,
    )

# #########################################################################
  ## Find best eta ###

  
     
  if config.train_hyperparameters:
    logging.info('Finding best hyperparameters...')

    if not os.path.exists(workdir + f'/hparam_tuning_{config.tune_hparams}'):
        os.makedirs(workdir + f'/hparam_tuning_{config.tune_hparams}', exist_ok=True)


    # Reset random key sequence
    prng_seq = hk.PRNGSequence(config.seed)
    train_ds, true_params, _ = load_dataset(n_groups=config.synth_n_groups, n_obs=config.synth_n_obs)

    
    if config.cond_hparams_names:
        cond_values_init = get_cond_values(cond_hparams_names=config.cond_hparams_names,
                            num_samples=config.num_samples_elpd,
                            prior_hparams_init=prior_hparams_init
                            )
    else:
        cond_values_init = None
    # mu_dim and theta_dim are also arguments of the flow,
    # as they define its dimension

    if config.tune_hparams == 'elpd_loocv':
        n_obs = train_ds.shape[0]
        mu_dim = sigma_dim = train_ds.shape[1]
        config.flow_kwargs.mu_dim = mu_dim
        config.flow_kwargs.sigma_dim = sigma_dim
        # Also is_smi modifies the dimension of the flow, due to the duplicated params
        config.flow_kwargs.num_groups = config.synth_n_groups

        states_lists= []
        #    states_lists = []

        for obs_idx in jnp.arange(n_obs*mu_dim):
                    checkpoint_dir = str(pathlib.Path(workdir) / f'loocv/dropped_{obs_idx}/checkpoints')
                    try:
                        assert os.path.isdir(checkpoint_dir), f"Directory needed for ELPD-LOOCV optim does not exist: {checkpoint_dir}"
                    except AssertionError as error:
                        sys.exit(1)  # Exit with a non-zero status code to indicate error
                    # assert os.path.isdir(checkpoint_dir), f"Directory needed for ELPD-LOOCV optim does not exist: {checkpoint_dir}"
                    state_list = []
                    state_name_list = []

                    state_name_list.append('MU and SIGMA')
                    state_list.append(
                        initial_state_ckpt(
                            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
                            forward_fn=hk.transform(q_distr_mu_sigma),
                            forward_fn_kwargs={
                                'flow_name': config.flow_name,
                                'flow_kwargs': config.flow_kwargs,
                                'cond_values': cond_values_init, 
                                'num_samples':config.num_samples_elpd,
                            },
                            prng_key=next(prng_seq),
                            optimizer=make_optimizer(**config.optim_kwargs),
                        ))

                    states_lists.append(state_list)

    elif ((config.tune_hparams == 'elpd_waic') or (config.tune_hparams == 'elbo')):
            n_obs = train_ds.shape[0]
            mu_dim = sigma_dim = train_ds.shape[1]
            config.flow_kwargs.mu_dim = mu_dim
            config.flow_kwargs.sigma_dim = sigma_dim
            # Also is_smi modifies the dimension of the flow, due to the duplicated params
            config.flow_kwargs.num_groups = config.synth_n_groups

            checkpoint_dir = str(pathlib.Path(workdir) / f'checkpoints')

            state_list = []
            state_list.append(
                initial_state_ckpt(
                    checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
                    forward_fn=hk.transform(q_distr_mu_sigma),
                    forward_fn_kwargs={
                        'flow_name': config.flow_name,
                        'flow_kwargs': config.flow_kwargs,
                        'cond_values': cond_values_init, 
                        'num_samples':config.num_samples_elpd,                },
                    prng_key=next(prng_seq),
                    optimizer=make_optimizer(**config.optim_kwargs),
                ))


    else:
        raise ValueError("Please define valid config.tune_hparams")


    # Initialize search with Bayes
    all_optimisers = {'elbo_opt':make_optimizer(**config.optim_kwargs),
                        'plain_lr1':make_optimizer_eta(**config.optim_kwargs_hp),  
                        'plain_lr2':make_optimizer_eta(config.optim_kwargs_hp_learning_rate_alternative)}

    all_inits = {'true':jnp.array([0., 1., 1., 1.]), 
                #    'low':jnp.array([0.00001, 0.00001, 0.00001]), 
                'high':jnp.array([5., 2., 5., 5.]),
                'medium':jnp.array([1, 1., 1., 1.]),
                'small':jnp.array([0.5, 0.5, 0.5, 0.5]),
                    }
    
    all_names = ['mu_prior_mean_m', 'mu_prior_scale_s', 'sigma_prior_concentration', 'sigma_prior_scale']
    optim_mask = jnp.array([1 if i in config.cond_hparams_names else 0 for i in all_names])
    print('optim mask:', optim_mask)
    optim_mask_indices = (tuple(i for i, x in enumerate(optim_mask) if x == 0),tuple(i for i, x in enumerate(optim_mask) if x == 1))

    for optimiser_name, my_optimiser in all_optimisers.items():  
        for (all_init_name, all_init) in all_inits.items():  
            
            hp_star_init = all_init[optim_mask==1]
            hp_fixed = all_init[optim_mask==0]

            print(f"init type: {all_init_name}, optimiser {optimiser_name}")
            info_dict = {'init_vals':hp_star_init, 
                            'init_type':all_init_name, 'optimiser_name':optimiser_name,
            'loss':[], 'params':[], 'step':[], 'hp_names':config.cond_hparams_names}

            if jax.process_index() == 0:
                summary_writer_hp = tensorboard.SummaryWriter(workdir + f'/hparam_tuning_{config.tune_hparams}/{all_init_name}_{optimiser_name}')
                summary_writer_hp.hparams({all_init_name:hp_star_init, optimiser_name:my_optimiser})

            if config.tune_hparams == 'elbo':
                # Jit optimization of eta
                update_hp_star_state_elbo = lambda hp_star_state, batch, prng_key: update_state(
                    state=hp_star_state,
                    batch=batch,
                    prng_key=prng_key,
                    optimizer=my_optimiser,#make_optimizer(**config.optim_kwargs),#make_optimizer_eta(**config.optim_kwargs_hp),
                    loss_fn=loss_neg_elbo,
                    loss_fn_kwargs={
                        'state_list_vmp': state_list,
                        'hp_fixed_values':hp_fixed,
                        'hp_optim_mask_indices':optim_mask_indices,
                    },
                )
                update_hp_star_state = jax.jit(update_hp_star_state_elbo)

            elif config.tune_hparams == 'elpd_waic':
                # Jit optimization of eta
                update_hp_star_state_waic = lambda hp_star_state, batch, prng_key: update_state(
                    state=hp_star_state,
                    batch=batch,
                    prng_key=prng_key,
                    optimizer=my_optimiser,#make_optimizer_eta(**config.optim_kwargs_hp),
                    loss_fn=loss_neg_elpd,
                    loss_fn_kwargs={
                        'state_list_vmp': state_list,
                    },
                )
                update_hp_star_state = jax.jit(update_hp_star_state_waic)
            elif config.tune_hparams == 'elpd_loocv':
                # Jit optimization of eta
                update_hp_star_state_loocv = lambda hp_star_state, batch, prng_key: update_state(
                        state=hp_star_state,
                        batch=batch,
                        prng_key=prng_key,
                        optimizer=my_optimiser, #make_optimizer_eta(**config.optim_kwargs_hp),
                        loss_fn=loss_neg_elpd_loocv,
                        loss_fn_kwargs={
                            'states_lists_vmp': states_lists,
                        },
                    )
                update_hp_star_state = jax.jit(update_hp_star_state_loocv)


            # SGD over elpd #
            hp_star_state = TrainState(
                params=hp_star_init,
                opt_state=my_optimiser.init(hp_star_init),#make_optimizer(**config.optim_kwargs).init(hp_star_init),#make_optimizer_eta(**config.optim_kwargs_hp).init(hp_star_init),
                step=0,
            )
            for _ in range(config.hp_star_steps):
                hp_star_state, neg_elpd = update_hp_star_state(
                    hp_star_state,
                    batch=train_ds,
                    prng_key=next(prng_seq),
                )

                # if state_list[0].step % config.hp_star_steps == 0:
                #   logging.info("STEP: %5d; training loss: %.3f", state_list[0].step,
                #                neg_elpd["train_loss"])
                info_dict['loss'].append(neg_elpd["train_loss"])
                info_dict['params'].append(hp_star_state.params)
                info_dict['step'].append(hp_star_state.step)

                if hp_star_state.step % 100 == 0:
                    # logging.info("STEP: %5d; training loss: %.3f; eta0:%.3f; eta1: %.3f; conc1: %.3f, conc2:%.3f", hp_star_state.step,
                    #             neg_elpd["train_loss"], hp_star_state.params[0], hp_star_state.params[1],
                    #             hp_star_state.params[2], hp_star_state.params[3])
                    
                    labs = "STEP: %5d; training loss: %.6f " + ' '.join([hp + ':%.3f' for hp in config.cond_hparams_names])
                    logging.info(labs,
                        float(hp_star_state.step),
                    float(neg_elpd["train_loss"]), *[float(hp_star_state.params[i]) for i in range(len(config.cond_hparams_names))])


                # Clip eta_star to [0,1] hypercube and hp_star to [0.000001,..]
                # hp_star_state = TrainState(
                #     params=jnp.hstack([jnp.clip(hp_star_state.params[:2],0, 1),
                #                     jnp.clip(hp_star_state.params[2:],0.000001, 15)]),
                #     opt_state=hp_star_state.opt_state,
                #     step=hp_star_state.step,
                # )

                hp_star_state = TrainState(
                    params=jnp.hstack([hp_star_state.params[config.cond_hparams_names.index('mu_prior_mean_m')],
                                    jnp.clip(hp_star_state.params[config.cond_hparams_names.index('mu_prior_scale_s')],0.000001, 15.) if 'mu_prior_scale_s' in config.cond_hparams_names else [],
                                    jnp.clip(hp_star_state.params[config.cond_hparams_names.index('sigma_prior_concentration')],0.000001, 15.) if 'sigma_prior_concentration' in config.cond_hparams_names else [],
                                    jnp.clip(hp_star_state.params[config.cond_hparams_names.index('sigma_prior_scale')],0.000001, 15) if 'sigma_prior_scale' in config.cond_hparams_names else [],
                                        ]),
                    opt_state=hp_star_state.opt_state,
                    step=hp_star_state.step,
                )

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
            with open(workdir + f"/hp_info_allhps_{all_init_name}_{optimiser_name}_{config.tune_hparams}.sav", 'wb') as f:
                        pickle.dump(info_dict, f)

        bools = np.array(optim_mask==1)
        plot_all.plot_optim_hparams_vs_true(path=workdir,
                                    init_names=list(all_inits.keys()),
                                    optimiser_name=optimiser_name,
                                    true_vals=true_params,
                                    hp_names=np.array(all_names)[bools],
                                    loss_type=config.tune_hparams,
                                    )



  return state_list
