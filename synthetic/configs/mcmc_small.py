"""Hyperparameter configuration."""

import ml_collections
import numpy as np


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'mcmc'

  config.num_samples = 10_000
  config.num_samples_subchain = 100
  config.num_burnin_steps = 1_000
  config.mcmc_step_size = 0.01

  config.synth_n_groups = 10
  config.synth_n_obs = 8
  config.mask_Y = np.ones((config.synth_n_obs*config.synth_n_groups,))

  config.seed = 4
  config.seed_synth = 3

  config.true_prior_hparams = ml_collections.ConfigDict()
  config.true_prior_hparams.mu_prior_mean_m = 0.
  config.true_prior_hparams.mu_prior_scale_s = 1.
  config.true_prior_hparams.sigma_prior_concentration = 1.5
  config.true_prior_hparams.sigma_prior_scale = 0.5

  return config
