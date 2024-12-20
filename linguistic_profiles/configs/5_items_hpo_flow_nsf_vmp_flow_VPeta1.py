"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'hp_opt_vmp_flow_allhp'#'vmp_flow_allhp' # vmp_flow

  # Dataset to use
  config.dataset_id = 'coarsen_8_items'

  # Defined in `flows.py`.
  config.flow_name = 'meta_nsf'
  config.cond_hparams_names = []
  config.optim_prior_hparams_dir_fixed_eta = ''
  config.eta_fixed = 1.0 # should still optimise eta
  config.tune_vmp_hparams_fix_eta = False
  config.tune_vmp_hparams = False

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()
  config.flow_kwargs.is_additive_flow = False
  config.flow_kwargs.num_basis_gps = 10
  config.flow_kwargs.inducing_grid_shape = (11, 11)
  # Number of layers to use in the flow.
  config.flow_kwargs.num_layers = 8
  # Hidden sizes of the MLP conditioner.
  config.flow_kwargs.hidden_sizes_conditioner = [30] * 5
  # Hidden sizes of the MLP conditioner.
  config.flow_kwargs.hidden_sizes_conditioner_eta = [30] * 5
  # Number of bins to use in the rational-quadratic spline.
  config.flow_kwargs.num_bins = 10
  # the bounds of the quadratic spline transformer
  config.flow_kwargs.spline_range = (-10., 10.)
  # Ranges of posterior locations
  # (NOTE: these will be modified in the training script)
  config.flow_kwargs.loc_x_range = (0., 1.)
  config.flow_kwargs.loc_y_range = (0., 0.8939394)

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

  config.prior_hparams_hparams = ml_collections.ConfigDict()
  config.prior_hparams_hparams.w_sampling_scale_alpha = 5.
  config.prior_hparams_hparams.w_sampling_scale_beta = 1.
  config.prior_hparams_hparams.a_sampling_scale_alpha = 10.
  config.prior_hparams_hparams.a_sampling_scale_beta = 1.
  config.prior_hparams_hparams.kernel_sampling_amplitude_alpha = 0.1
  config.prior_hparams_hparams.kernel_sampling_amplitude_beta = 0.4
  config.prior_hparams_hparams.kernel_sampling_lengthscale_alpha = 0.2
  config.prior_hparams_hparams.kernel_sampling_lengthscale_beta = 0.5

  # Number of training steps to run.
  config.training_steps = 200_000 #300_000

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs.init_value = 0.
#   config.optim_kwargs.lr_schedule_kwargs.peak_value = 3e-4
  config.optim_kwargs.lr_schedule_kwargs.warmup_steps = 3_000
  config.optim_kwargs.lr_schedule_kwargs.transition_steps = 10_000
#   config.optim_kwargs.lr_schedule_kwargs.decay_rate = 0.5
  config.optim_kwargs.lr_schedule_kwargs.transition_begin = 0
  config.optim_kwargs.lr_schedule_kwargs.staircase = False
  config.optim_kwargs.lr_schedule_kwargs.end_value = None

#   config.optim_kwargs.lr_schedule_kwargs = {
#       'init_value': 0.,
#       'peak_value': 3e-4,
#       'warmup_steps': 3_000,
#       'transition_steps': 10_000,
#       'decay_rate': 0.5,
#       'transition_begin': 0,
#       'staircase': False,
#       'end_value': None,
#   }
  

  config.num_lp_anchor_train = 120
  config.num_lp_floating_train = 10
  config.num_items_keep = 5 #8
  config.num_lp_anchor_val = 0
  config.num_lp_anchor_test = 0
  config.remove_empty_forms = True
  config.ad_hoc_val_profiles = False

  config.lp_floating_train = (136, 234, 1002, 501, 236, 237, 319, 515, 699, 755)

  # Number of posteriors samples to approximate the variational loss (ELBO).
  config.num_samples_elbo = 5
  config.num_samples_gamma_profiles = 5

  # How often to evaluate the model.
  config.eval_steps = config.training_steps // 10 #5
  config.num_samples_eval = 100
  config.eval_last = False

  config.max_steps_nan = 1_000
  config.max_steps_inf = 2_000

  # How often to log images to monitor convergence.
  config.log_img_steps = 0#config.training_steps // 5
  config.log_img_at_end = False
  config.save_samples = False

  # Number of samples used in the plots.
  config.num_samples_plot = 1_000
  config.num_samples_save = 1_000
  config.eta_save = [1.0]
  config.num_samples_chunk_plot = 500

  # Floating profiles to plot in grid
  config.floating_anchor_copies = False # CHECK ALWAYS!!!
  config.lp_floating_grid10 = [
      136, 234, 1002, 501, 236, 237, 319, 515, 699, 755
  ]
  # config.lp_floating_grid10 = None
  config.lp_random_anchor_10 = None

  # eta shown in figures
  config.eta_plot = [0.001, 0.250, 0.500, 0.750, 1.000]

  # How often to save model checkpoints.
  config.checkpoint_steps = 0 #config.training_steps // 5
  config.save_last_checkpoint = True
  # How many checkpoints to keep.
  config.checkpoints_keep = 2
  config.save_best_checkpoint = True

  # Number of samples of eta for Meta-Posterior training
  config.eta_sampling_a = 0.5
  config.eta_sampling_b = 0.5

  # Use random location for anchor profiles for evaluation
  config.include_random_anchor = False
  # Metric for Hyperparameter Optimization
  config.synetune_metric = "mean_dist_anchor_val_min"
  config.path_mcmc_img = ''
  config.path_MCMC_samples = ''
  config.max_wass_samples = 1_000
  # Random seed
  config.seed = 1

  config.use_wandb = True
  config.fixed_configs_wandb = {
            # "kernel_amplitude": 0.2,
            # "kernel_length_scale": 0.3,
            "peak_value":3e-4,
            "decay_rate":0.5,
        }

  


  config.wandb_project_name = ''

  config.sweep_configuration = ml_collections.ConfigDict()
  config.sweep_configuration.method = "bayes"

  config.sweep = True
  config.sweep_id = ''
  config.n_sweep_runs = 10
  config.wandb_evaleta  = 1.0
  config.sweep_configuration.metric = {"goal": "minimize", "name": "min_WD_joint_vs_MCMC"} # f"mean_dist_anchor_val_eta{config.wandb_evaleta}"}

  config.sweep_configuration.parameters = {
            # "kernel_amplitude": { "min": 0.03, "max": 1.0},
            # "kernel_length_scale": { "min": 0.03, "max": 1.0},
            "peak_value":{ "min": 1e-5, "max": 1e-2},
            "decay_rate":{ "min": 0.1, "max": 1.0},
        }
  


  return config