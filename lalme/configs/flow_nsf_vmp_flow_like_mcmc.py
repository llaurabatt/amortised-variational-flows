"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'vmp_flow'

  # Dataset to use
  config.dataset_id = 'coarsen_8_items'

  # Defined in `flows.py`.
  config.flow_name = 'meta_nsf'

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()
  config.flow_kwargs.num_basis_gps = 5
  config.flow_kwargs.inducing_grid_shape = (10, 10)
  # Number of layers to use in the flow.
  config.flow_kwargs.num_layers = 6
  # Hidden sizes of the MLP conditioner.
  config.flow_kwargs.hidden_sizes_conditioner = [30] * 5
  # Hidden sizes of the MLP conditioner.
  config.flow_kwargs.hidden_sizes_conditioner_eta = [30] * 5
  # Number of bins to use in the rational-quadratic spline.
  config.flow_kwargs.num_bins = 10
  # the bounds of the quadratic spline transformer
  config.flow_kwargs.spline_range = (-10., 10)
  # Ranges of posterior locations
  # (NOTE: these will be modified in the training script)
  config.flow_kwargs.loc_x_range = (0., 1.)
  config.flow_kwargs.loc_y_range = (0., 0.8939394)

  # Define priors
  config.prior_params = ml_collections.ConfigDict()
  config.prior_params.mu_prior_concentration = 0.1
  config.prior_params.mu_prior_rate = 0.1
  config.prior_params.zeta_prior_a = 1.
  config.prior_params.zeta_prior_b = 1.
  config.prior_params.w_prior_scale = 0.1
  config.prior_params.a_prior_scale = 1.
  config.kernel_name = 'ExponentiatedQuadratic'
  config.kernel_kwargs = ml_collections.ConfigDict()
  config.kernel_kwargs.amplitude = 0.7
  config.kernel_kwargs.length_scale = 0.5
  config.gp_jitter = 1e-3

  # Number of training steps to run.
  config.training_steps = 20_000

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 3e-4,
      'warmup_steps': 3_000,
      'transition_steps': config.training_steps / 4,
      'decay_rate': 0.5,
      'transition_begin': 0,
      'staircase': False,
      'end_value': None,
  }

  config.num_profiles_anchor_keep = None
  config.num_profiles_floating_keep = 20
  config.num_items_keep = None
  config.remove_empty_forms = True

  # Number of posteriors samples to approximate the variational loss (ELBO).
  config.num_samples_elbo = 5
  config.num_samples_gamma_profiles = 5

  # How often to evaluate the model.
  config.eval_steps = int(config.training_steps / 10)
  config.num_samples_eval = 100

  config.include_random_anchor = True

  # How often to log images to monitor convergence.
  config.log_img_steps = int(config.training_steps / 10)
  config.show_basis_fields_during_training = True
  config.show_linguistic_fields_during_training = True

  # Number of posteriors samples used in the plots.
  config.num_samples_plot = 2_000

  config.eta_plot = [
      [0.001],
      [0.5],
      [1.0],
  ]

  # How often to save model checkpoints.
  config.checkpoint_steps = int(config.training_steps / 2)
  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Number of samples of eta for Meta-Posterior training
  config.eta_sampling_a = 0.2
  config.eta_sampling_b = 1.0

  # Initial seed for random numbers.
  config.seed = 123

  return config
