"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'mcmc' #'mcmc'

  # Dataset to use
  config.dataset_id = 'coarsen_8_items'

  # Data specification
  config.num_lp_anchor_train = 120 # 120
  config.num_lp_floating_train = 10
  config.num_items_keep = 5#8
  config.num_lp_anchor_val = 0
  config.num_lp_anchor_test = 0
  config.remove_empty_forms = True

  config.lp_floating_train = (136, 234, 1002, 501, 236, 237, 319, 515, 699, 755)

  # Model hyperparameters.
  config.model_hparams = ml_collections.ConfigDict()
  # Number of basis GPs
  config.model_hparams.num_basis_gps = 10
  # Grid of inducing points
  config.model_hparams.inducing_grid_shape = (11, 11)
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
  config.prior_hparams.w_prior_scale = 5.
  config.prior_hparams.a_prior_scale = 10.
  config.kernel_name = 'ExponentiatedQuadratic'
  config.kernel_kwargs = ml_collections.ConfigDict()
  config.kernel_kwargs.amplitude = 0.2
  config.kernel_kwargs.length_scale = 0.3
  config.gp_jitter = 1e-3

  config.num_samples_gamma_profiles = 100

  config.num_samples = 15_000 # 10_000, 5_000, 5_000, 100 CHRIS 5_000
  config.thinning = 10 # Laura
  # config.num_samples_perchunk_stg1 = 100 # 100, 20 
  config.num_samples_subchain_stg2 = 100 # 300, 100, 100, 10 CHRIS 100
  config.num_samples_perchunk_stg2 = 10 # 100, 100, 20 CHRIS 100
  # config.num_burnin_steps_stg1 = 1_000 # CHRIS 1_000
  config.num_steps_call_warmup = 500 # CHRIS 200

  config.mcmc_step_size = 0.01
  config.num_chains = 1
  config.max_wass_samples = 1_000

  # SMI degree of influence of floating profiles
  config.eta_profiles_floating = 1.0

  config.seed = 1

  # Samples from variational posterior to compare locations
  config.path_variational_samples = ''

  # Plotting
  config.lp_floating_grid10 = [
      136, 234, 1002, 501, 236, 237, 319, 515, 699, 755
  ]
  config.plot_floating_aux = True
  # config.lp_floating_grid10 = None

  return config
