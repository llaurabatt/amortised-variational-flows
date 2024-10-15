"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'mcmc'

  config.num_samples = 10_000
  config.num_samples_subchain = 100
  config.num_burnin_steps = 1_000
  config.mcmc_step_size = 0.01

  config.smi_eta = 0.87
  config.c1 = 1.0
  config.c2 = 1.0

  config.seed = 0

  return config
