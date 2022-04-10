"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'flow'

  # Dataset to use
  config.dataset_id = 'coarsen_all_items'

  # Defined in `flows.py`.
  config.flow_name = 'nsf'

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()
  config.flow_kwargs.num_base_gps = 7
  config.flow_kwargs.inducing_grid_shape = (10, 10)
  # Number of layers to use in the flow.
  config.flow_kwargs.num_layers = 6
  # Hidden sizes of the MLP conditioner.
  config.flow_kwargs.hidden_sizes = [5, 25, 5]
  # Number of bins to use in the rational-quadratic spline.
  config.flow_kwargs.num_bins = 10
  # the bounds of the quadratic spline transformer
  config.flow_kwargs.spline_range = (-10., 10)
  # Ranges of posterior locations
  # (NOTE: these will be modified in the training script)
  config.flow_kwargs.loc_x_range = (0., 1.)
  config.flow_kwargs.loc_y_range = (0., 0.8939394)

  # SMI degree of influence of floating profiles
  config.iterate_smi_eta = ()
  config.flow_kwargs.smi_eta = {'items': None, 'profiles': None}

  # Number of training steps to run.
  config.training_steps = 30_000

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 5e-3,
      'warmup_steps': 3_000,
      'transition_steps': 10_000,
      'decay_rate': 0.5,
      'transition_begin': 0,
      'staircase': False,
      'end_value': None,
  }

  config.num_profiles_anchor_keep = None
  config.num_profiles_floating_keep = None
  config.num_items_keep = None
  config.remove_empty_forms = True

  # config.loc_bounds = 5.
  # config.loc_bounds_penalty = 10.

  config.kernel_name = 'ExponentiatedQuadratic'
  config.kernel_kwargs = ml_collections.ConfigDict()
  config.kernel_kwargs.amplitude = 0.1
  config.kernel_kwargs.length_scale = 0.1
  config.gp_jitter = 1e-3

  # Number of posteriors samples to approximate the variational loss (ELBO).
  config.num_samples_elbo = 20
  config.num_samples_gamma_profiles = 10

  # Number of steps using a random influence eta
  config.random_eta_steps = 0

  # How often to evaluate the model.
  config.eval_steps = int(config.training_steps / 20)

  config.num_samples_eval = 100

  config.include_random_anchor = True

  # How often to generate posterior plots.
  config.log_img_steps = int(config.training_steps / 10)

  # How often to save model checkpoints.
  config.checkpoint_steps = int(config.training_steps / 4)
  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Number of posteriors samples used in the plots.
  config.num_samples_plot = 1_000

  # Number of profiles locations to plot
  config.num_profiles_plot = 20

  # Random seed
  config.seed = 0

  return config
