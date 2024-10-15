"""MCMC sampling for the Epidemiology model."""

import time
from absl import logging
import pickle
import ml_collections

import numpy as np

import jax
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp



from flax.metrics import tensorboard

import haiku as hk

import log_prob_fun_integrated
from log_prob_fun_integrated import PriorHparams
import plot_all
from train_flow import load_dataset

from modularbayes import flatten_dict
from modularbayes._src.typing import Any, Mapping, Optional

tfb = tfp.bijectors
tfd = tfp.distributions
tfm = tfp.mcmc

np.set_printoptions(suppress=True, precision=4)

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
ConfigDict = ml_collections.ConfigDict
SmiEta = Mapping[str, np.ndarray]


def get_posterior_sample_init(mu_dim: int, 
                              sigma_dim: int,
                              key: PRNGKey,
                              prior_hparams: PriorHparams,):
  
    # hparams = {'mu_prior_mean_m':0., 'mu_prior_scale_s':1, 'sigma_prior_concentration':3, 'sigma_prior_scale':1.5}
    keys = jax.random.split(key, 2)
    #%%

    mu_g = tfd.Normal(loc=prior_hparams.mu_prior_mean_m, 
                        scale=prior_hparams.mu_prior_scale_s).sample(
        sample_shape=(mu_dim,), seed=keys[0])
    sigma_g = tfd.InverseGamma(concentration=prior_hparams.sigma_prior_concentration, 
                        scale=prior_hparams.sigma_prior_scale).sample(
        sample_shape=(sigma_dim,), seed=keys[1])
    posterior_sample = {}
    posterior_sample['mu'] = jnp.expand_dims(mu_g, axis=0)
    posterior_sample['sigma'] = jnp.expand_dims(sigma_g, axis=0)

    return posterior_sample


@jax.jit
def log_prob_fn(
    batch: Batch,
    model_params: Array,
    model_params_init: Mapping[str, Any],
    prior_hparams: PriorHparams,
    mask_Y: Array,
):
  """Log probability function for the Epidemiology model."""

  leaves_init, treedef = jax.tree_util.tree_flatten(model_params_init)

  leaves = []
  for i in range(len(leaves_init) - 1):
    param_i, model_params = jnp.split(
        model_params, leaves_init[i].flatten().shape, axis=-1)
    leaves.append(param_i.reshape(leaves_init[i].shape))
  leaves.append(model_params.reshape(leaves_init[-1].shape))

  posterior_sample_dict = jax.tree_util.tree_unflatten(
      treedef=treedef, leaves=leaves)


  log_prob = log_prob_fun_integrated.log_prob_joint(
      batch=batch,
      posterior_sample_dict=posterior_sample_dict,
      prior_hparams=prior_hparams,
      mask_Y=mask_Y,
  )['log_prob'].squeeze()

  return log_prob


def sample_and_evaluate(config: ConfigDict, workdir: str) -> Mapping[str, Any]:
  """Sample and evaluate the epidemiology model."""

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)

  train_ds, true_params, sim_data_fig = load_dataset(n_groups=config.synth_n_groups, 
                                       n_obs=config.synth_n_obs,
                                       seed=config.seed_synth)


  mu_dim = sigma_dim = train_ds.shape[1] #len(train_ds.keys())
  # Small data, no need to batch

  true_prior_hparams_dict = {'mu_prior_mean_m': config.true_prior_hparams.mu_prior_mean_m, 
            'mu_prior_scale_s': config.true_prior_hparams.mu_prior_scale_s, 
            'sigma_prior_concentration': config.true_prior_hparams.sigma_prior_concentration, 
            'sigma_prior_scale': config.true_prior_hparams.sigma_prior_scale, 
            }

  PriorHparams.set_defaults(**true_prior_hparams_dict)

  true_prior_hparams = PriorHparams()


  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))


  # Initilize the model parameters
  posterior_sample_dict_init = get_posterior_sample_init(
      mu_dim=mu_dim,
      sigma_dim=sigma_dim,
      key=next(prng_seq),
      prior_hparams=true_prior_hparams,
  )
  # better init for theta
#   posterior_sample_dict_init['theta'] = jnp.array([[-1.5, 20]])

  ### Sample First Stage ###

  logging.info("\t sampling... ")

  times_data = {}
  times_data['start_sampling'] = time.perf_counter()

  target_log_prob_fn = lambda state: log_prob_fn(
      batch=train_ds,
      model_params=state,
      model_params_init=posterior_sample_dict_init,
      prior_hparams=true_prior_hparams,
      mask_Y=config.mask_Y,
  )

  posterior_sample_init = jnp.concatenate([
      posterior_sample_dict_init['mu'],
      posterior_sample_dict_init['sigma'],
  ],
                                          axis=-1)[0, :]
  target_log_prob_fn(posterior_sample_init)

  inner_kernel = tfm.NoUTurnSampler(
      target_log_prob_fn=target_log_prob_fn,
      step_size=config.mcmc_step_size,
  )

  # Define bijectors for mapping values to parameter domain
  # phi goes to (0,1)
  # theta1 goes to [-Inf,Inf]
  # theta2 goes to [0,Inf]
  block_bijectors = [tfb.Identity(), tfb.Softplus()]
  block_sizes = [mu_dim, sigma_dim]
  kernel_bijector = tfb.Blockwise(
      bijectors=block_bijectors, block_sizes=block_sizes)

  kernel = tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=inner_kernel, bijector=kernel_bijector)

  times_data['start_mcmc'] = time.perf_counter()
  posterior_sample = tfm.sample_chain(
      num_results=config.num_samples,
      num_burnin_steps=config.num_burnin_steps,
      kernel=kernel,
      current_state=posterior_sample_init,
      trace_fn=None,
      seed=next(prng_seq),
  )

  posterior_sample_dict = {}
  posterior_sample_dict['mu'], posterior_sample_dict['sigma'] = jnp.split(
      posterior_sample, [mu_dim], axis=-1)

  logging.info("posterior means mu %s",
               str(posterior_sample_dict['mu'].mean(axis=0)))

  times_data['end_mcmc'] = time.perf_counter()

  logging.info("posterior means sigma %s",
               str(posterior_sample_dict['sigma'].mean(axis=0)))

  times_data['end_sampling'] = time.perf_counter()

  logging.info("Sampling times:")
  logging.info("\t Total: %s",
               str(times_data['end_sampling'] - times_data['start_sampling']))
  logging.info(
      "\t Stg 1: %s",
      str(times_data['end_mcmc'] - times_data['start_mcmc']))
  
  logging.info("Saving samples...")
  with open(workdir + f"/mcmc_samples_true_hparams.sav", 'wb') as f:
       pickle.dump(posterior_sample_dict, f)

  logging.info("Plotting results...")

  for par_name, par_samples in {'\mu':posterior_sample_dict['mu'], '\sigma': posterior_sample_dict['sigma']}.items():
    fig = plot_all.plot_final_posterior_vs_true(mcmc_samples=par_samples, par_name_plot=par_name, true_params=true_params)
    fig.savefig(workdir + f"/posterior_vs_true_{par_name[1:]}" + ".png")



  # j = 2
  # fig, axs = plt.subplots(2, 1)
  # axs[0].hist(posterior_sample_dict['phi'][:, j], 30)
  # axs[0].axvline(
  #     x=(train_ds['Z'] / train_ds['N'])[j], color='red', linestyle='dashed')
  # axs[0].set_xlim(-0.01, 1.01)
  # axs[1].plot(posterior_sample_dict['phi'][:, j])

  # j = 13
  # fig, axs = plt.subplots(2, 1)
  # axs[0].hist(states[:, j], 30)
  # axs[1].plot(states[:, j])
  with open(workdir + f"/mcmc_samples_{config.synth_n_groups}groups_{config.synth_n_obs}obs_true_hparams.sav", 'wb') as f:
    pickle.dump(posterior_sample_dict, f)

  return posterior_sample_dict
