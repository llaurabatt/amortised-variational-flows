"""Hyperparameter configuration."""

import ml_collections
import jax.numpy as jnp


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'vmp_flow_elpd_integrated'

  config.estimate_smi = True #CHECK ALWAYS!!
  config.cond_hparams_names = ['eta', 'c1', 'c2']
  config.train_hyperparameters = False

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
  config.num_samples_elbo = 1000 #100

  # Number of training steps to run.
  config.training_steps = 50_000

  # Number of SGD steps to find the best hyperparameters
  config.hp_star_steps = 15_000#10_000

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

  config.optim_kwargs_hp_learning_rate_alternative = 1e-2
  # Optimizer for searching hp
  config.optim_kwargs_hp = ml_collections.ConfigDict()
  config.optim_kwargs_hp.learning_rate = 1e-4

  # How often to evaluate the model.
  config.eval_steps = jnp.inf #config.training_steps / 10
  config.num_samples_eval = 5_000
  config.num_modules = 2


  # Initial seed for random numbers.
  config.seed = 0

  # How often to log images to monitor convergence.
  config.activate_log_img = False
  config.log_img_steps = config.training_steps / 10

  # Number of posteriors samples used in the plots.
  config.num_samples_plot = 10_000

  config.num_samples_elpd = 1_000

  # config.drop_obs = -100
  config.mask_Y = (1,)*13
  config.mask_Z = (1,)*13
  config.tune_hparams = 'elpd_loocv' #'elpd_waic'
  config.tune_hparams_tensorboard = False

  config.priorhp_default_elpdplot = [1., 1.]
  config.plot_two = True
# [0.073, 3.72][1., 1.][5.108, 4.375]
# [0.778, 15.000][1., 1.][5.741, 15.000]
  # How often to save model checkpoints.
  config.checkpoint_steps = jnp.inf #config.training_steps / 4

  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Number of samples of eta for Meta-Posterior training
  config.eta_sampling_a = 0.4
  config.eta_sampling_b = 1.0
  # config.betahp_sampling_a = 2 #0.6 
  # config.betahp_sampling_b_rate = 1/3 #1/4
  config.betahp_sampling_a_uniform = 0
  config.betahp_sampling_b_uniform = 15


  return config
