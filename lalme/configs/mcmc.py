"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'mcmc'

  # Dataset to use
  config.dataset_id = 'coarsen_8_items'

  # Data specification
  config.num_profiles_anchor_keep = None
  config.num_profiles_floating_keep = None
  config.num_items_keep = None
  config.remove_empty_forms = True

  # Model hyperparameters.
  config.model_hparams = ml_collections.ConfigDict()
  # Number of basis GPs
  config.model_hparams.num_basis_gps = 5
  # Grid of inducing points
  config.model_hparams.inducing_grid_shape = (10, 10)
  # Ranges of posterior locations
  # (NOTE: these will be modified in the training script)
  config.model_hparams.loc_x_range = (0., 1.)
  config.model_hparams.loc_y_range = (0., 0.8939394)

  # Define priors
  config.prior_hparams = ml_collections.ConfigDict()
  config.prior_hparams.mu_prior_concentration = 1.
  config.prior_hparams.mu_prior_rate = 0.5
  config.prior_hparams.zeta_prior_a = 1.
  config.prior_hparams.zeta_prior_b = 1.
  config.prior_hparams.w_prior_scale = 1.
  config.prior_hparams.a_prior_scale = 10.
  config.kernel_name = 'ExponentiatedQuadratic'
  config.kernel_kwargs = ml_collections.ConfigDict()
  config.kernel_kwargs.amplitude = 0.1
  config.kernel_kwargs.length_scale = 0.1
  config.gp_jitter = 1e-3

  config.num_samples_gamma_profiles = 10

  config.num_samples = 2_000
  config.num_burnin_steps_stg1 = 500
  config.num_samples_subchain_stg2 = 50
  config.num_chunks_stg2 = 50

  config.mcmc_step_size = 0.01

  config.iterate_smi_eta = ()
  config.smi_eta = {'items': None, 'profiles': None}

  config.seed = 0

  return config
