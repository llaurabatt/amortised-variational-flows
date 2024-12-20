"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'vmp_flow_elpd'

  # Defined in `epidemiology.models.flows`.
  config.flow_name = 'meta_nsf'

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()
  # Number of modules
  config.flow_kwargs.num_modules = 2  # TOCHECK: where to call this arg, whether it is only a flow arg
  # Number of layers to use in the flow.
  config.flow_kwargs.num_layers = 4
  # Hidden sizes
  # Hidden sizes of the MLP conditioner.
  config.flow_kwargs.hidden_sizes_conditioner = [5] * 3 
  # Hidden sizes of the MLP conditioner for eta.
  config.flow_kwargs.hidden_sizes_conditioner_eta = [5] * 3 
  # Hidden sizes of the MLP conditioner for prior.
  config.flow_kwargs.hidden_sizes_conditioner_prior = [5] * 3 
  # Number of bins to use in the rational-quadratic spline.
  config.flow_kwargs.num_bins = 10
  # the lower bound of the spline's range
  config.flow_kwargs.range_min = -10.
  # the upper bound of the spline's range
  config.flow_kwargs.range_max = 40.

  # Number of samples to approximate ELBO's gradient
  config.num_samples_elbo = 100

  # Number of training steps to run.
  config.training_steps = 20_000

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 3e-3,
      'warmup_steps': 3_000,
      'transition_steps': config.training_steps / 4,
      'decay_rate': 0.5,
      'transition_begin': 0,
      'staircase': False,
      'end_value': None,
  }

  # How often to evaluate the model.
  config.eval_steps = config.training_steps / 10
  config.num_samples_eval = 5_000
  config.num_modules = 2


  # Initial seed for random numbers.
  config.seed = 0

  # How often to log images to monitor convergence.
  config.log_img_steps = config.training_steps / 10

  # Number of posteriors samples used in the plots.
  config.num_samples_plot = 10_000

  config.num_samples_elpd = 1_000

  config.eta_plot = [[1., 0.001], 
                     [1., 0.1], 
                     [1., 1.]]
  
  # How often to save model checkpoints.
  config.checkpoint_steps = config.training_steps / 4

  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Number of samples of eta for Meta-Posterior training
  config.eta_sampling_a = 0.2
  config.eta_sampling_b = 1.0
  config.betahp_sampling_a = 0.5
  config.betahp_sampling_b = 1.5

  return config
