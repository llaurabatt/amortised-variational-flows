"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'flow'

  # Dataset to use
  config.dataset_id = 'coarsen_all_items'

  # Defined in `flows.py`.
  config.flow_name = 'mean_field'

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()
  config.flow_kwargs.num_basis_gps = 10
  config.flow_kwargs.inducing_grid_shape = (11, 11)
  # Ranges of posterior locations
  # (NOTE: these will be modified in the training script)
  config.flow_kwargs.loc_x_range = (0., 1.)
  config.flow_kwargs.loc_y_range = (0., 0.8939394)

  # SMI degree of influence of floating profiles
  config.eta_profiles_floating = 1.0

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
  config.kernel_kwargs.amplitude = 0.65
  config.kernel_kwargs.length_scale = 0.2
  config.gp_jitter = 1e-3

  # Number of training steps to run.
  config.training_steps = 30_000

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 3e-2,
      'warmup_steps': 2_000,
      'transition_steps': 10_000,
      'decay_rate': 0.5,
      'transition_begin': 0,
      'staircase': False,
      'end_value': None,
  }

  config.num_lp_anchor_train = 120
  config.num_lp_floating_train = 247
  config.num_items_keep = 71
  config.num_lp_anchor_val = 0
  config.num_lp_anchor_test = 0
  config.remove_empty_forms = True

  # Number of posteriors samples to approximate the variational loss (ELBO).
  config.num_samples_elbo = 20
  config.num_samples_gamma_profiles = 10

  # How often to evaluate the model.
  config.eval_steps = config.training_steps // 10
  config.show_basis_fields_during_training = False
  config.show_linguistic_fields_during_training = False

  config.num_samples_eval = 100

  # How often to generate posterior plots.
  config.log_img_steps = config.training_steps // 5
  config.log_img_at_end = True

  # How often to save model checkpoints.
  config.checkpoint_steps = config.training_steps // 2
  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Number of samples used in the plots.
  config.num_samples_plot = 10_000
  config.num_samples_chunk_plot = 1_000

  # Floating profiles to plot in grid
  config.lp_floating_grid10 = [5, 29, 30, 16, 45, 52, 46, 38, 51, 49]
  config.lp_random_anchor_10 = [85, 133, 363, 544, 1135, 91, 90, 1287, 612, 731]

  # Use random location for anchor profiles for evaluation
  config.include_random_anchor = True
  # Metric for Hyperparameter Optimization
  config.synetune_metric = "mean_dist_anchor_val_min"

  # Random seed
  config.seed = 0

  return config
