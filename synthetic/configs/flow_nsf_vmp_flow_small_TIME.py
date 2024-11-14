"""Hyperparameter configuration."""

import ml_collections
import numpy as np


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'vmp_flow_elpd_integrated'

  config.cond_hparams_names = ['mu_prior_mean_m', 'mu_prior_scale_s', 'sigma_prior_concentration', 'sigma_prior_scale']
  config.opt_cond_hparams_values = [0.4, 0.85, 1.05, 0.43] # 10 groups 8 obs
  # config.alternative_cond_hparams_values = [0., 10000, 0.01, 0.01]
  config.alternative_cond_hparams_values = []#[2., 2., 0.5, 1.5] 
  config.checkpoint_dir_comparison = ml_collections.ConfigDict()
  config.checkpoint_dir_comparison.alternative = ''
  config.checkpoint_dir_comparison.true = '/home/llaurabat/mount/vmp-output/synthetic/vp-true-ng10-nobs8/checkpoints'
  config.checkpoint_dir_comparison.opt = '/home/llaurabat/mount/vmp-output/synthetic/vmp-ng10-nobs8/checkpoints'
  config.mcmc_samples_true_hparams_path = '/home/llaurabat/mount/vmp-output/synthetic/mcmc-ng10-nobs8/mcmc_samples_10groups_8obs_true_hparams.sav'
  config.true_hparams = [0., 1., 1.5, 0.5]
  config.train_hyperparameters = False

  # synthetic data settings
  
  
  config.synth_n_groups = 10
  config.synth_n_obs = 8
  # config.mask_Y = np.ones((config.synth_n_obs, config.synth_n_groups))
  config.mask_Y = np.ones((config.synth_n_obs*config.synth_n_groups,))

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
  config.training_steps = 40_000 #50_000

  # Number of SGD steps to find the best hyperparameters
  config.hp_star_steps = 3_000#10_000
  
  config.tuning_vals_optim_kwargs = {
      'grad_clip_value': [1.0, 5.0],  # Experiment with values like 1.0, 5.0, and 10.0
      'peak_value': [3e-3, 1e-4, 1e-3, 1e-2],  # Experiment within this range
      'warmup_steps': [1000, 3000, 5000],  # Adjust based on total steps
      'transition_steps': [0.25 * config.hp_star_steps, 0.5 * config.hp_star_steps],  # Use a fraction of your total training steps
      'decay_rate': [0.5, 0.9, 0.95],  # Experiment with how fast the LR decays
      }

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 3e-3,
      'warmup_steps': 3_000,
      'transition_steps': config.hp_star_steps/ 4,#config.training_steps / 4,
      'decay_rate': 0.5,
      'transition_begin': 0,
      'staircase': False,
      'end_value': None,
  }
  # Optimizer for searching hp
  config.optim_kwargs_hp = ml_collections.ConfigDict()
  config.optim_kwargs_hp.learning_rate = 1e-4
  config.optim_kwargs_hp_learning_rate_alternative = 1e-2



  # How often to evaluate the model.
  config.eval_steps = np.inf #config.training_steps / 20#10
  config.num_samples_eval = 5_000
  config.num_modules = 2


  # Initial seed for random numbers.
  config.seed = 0
  config.seed_synth = 3

  # How often to log images to monitor convergence.
  config.log_img_steps = np.inf #config.training_steps / 10

  # Number of posteriors samples used in the plots.
  config.num_samples_plot = 10_000

  config.num_samples_elpd = 1_000
  config.num_samples_elbo_optim = 1_000

  config.priorhp_default_elpdplot = [1., 1.]
  config.plot_two = True
# [0.073, 3.72][1., 1.][5.108, 4.375]
# [0.778, 15.000][1., 1.][5.741, 15.000]
  # How often to save model checkpoints.
  config.checkpoint_steps = np.inf #config.training_steps / 4

  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Number of samples of eta for Meta-Posterior training
  config.prior_hparams = ml_collections.ConfigDict()
  config.prior_hparams.mu_prior_mean_m = 0.
  config.prior_hparams.mu_prior_scale_s = 10000.
  config.prior_hparams.sigma_prior_concentration = 0.01
  config.prior_hparams.sigma_prior_scale = 0.01
  
  # Number of samples of eta for Meta-Posterior training
  config.prior_hparams_hparams = ml_collections.ConfigDict()
  config.prior_hparams_hparams.mu_m_gaussian_mean = 0.
  config.prior_hparams_hparams.mu_m_gaussian_scale = 3.
  config.prior_hparams_hparams.mu_s_gamma_a_shape = 2. #1.1 # 0.5 breaks everything, 1 almost breaks
  config.prior_hparams_hparams.mu_s_gamma_b_rate = 1. #1/1.1#1/2
  config.prior_hparams_hparams.sigma_hps_gamma_a_shape = 0.5 #0.5
  config.prior_hparams_hparams.sigma_hps_gamma_b_rate = 1/2 #1/2

  config.tune_hparams = 'elbo'


  return config
