"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.n_amortisation_points = 10
  # Dataset to use
  config.dataset_id = 'coarsen_all_items'
  config.workdir_VMP = '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow'
  config.workdirs_VP = ['/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow/VP_eta_0.05',
                        '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow/VP_eta_0.250',
                        '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow/VP_eta_0.420',
                        '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow/VP_eta_0.500',
                        # '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow/VP_eta_0.610',
                        '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow/VP_eta_0.750',
                        '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow/VP_eta_1.000']
  # config.workdir_VMP = '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow'
  # # config.workdirs_VP = ['/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow/VP_eta_0.001',
  # config.workdirs_VP = ['/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow/VP_eta_0.05',
  #                       '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow/VP_eta_0.250',
  #                       '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow/VP_eta_0.420',
  #                       '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow/VP_eta_0.500',
  #                       '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow/VP_eta_0.610',
  #                       '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow/VP_eta_0.750',
  #                       '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow/VP_eta_1.000']
  config.workdir_AdditiveVMP = ''
  config.optim_prior_hparams_dir = '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow'
  # config.optim_prior_hparams_dir = '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES/all_items/nsf/vmp_flow'
  config.cond_hparams_names = ['w_prior_scale', 'a_prior_scale', 'kernel_amplitude', 'kernel_length_scale', 'eta']
  config.etas = [0.05, 0.25, 0.42, 0.5, 0.75, 1.]
  config.prior_hparams_fixed = [5., 10., 1., 0.5, 1., 1., 0.2, 0.3]
  config.num_samples_amortisation_plot = 50
  config.loss_type = 'ELBO'
  # Defined in `flows.py`.
  config.flow_name = 'meta_nsf'

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()
  config.flow_kwargs.num_basis_gps = 10
  config.flow_kwargs.inducing_grid_shape = (11, 11)
  # Number of layers to use in the flow.
  config.flow_kwargs.num_layers = 6
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

  config.prior_hparams_hparams = ml_collections.ConfigDict()
  config.prior_hparams_hparams.w_sampling_scale_alpha = 5.
  config.prior_hparams_hparams.w_sampling_scale_beta = 1.
  config.prior_hparams_hparams.a_sampling_scale_alpha = 10.
  config.prior_hparams_hparams.a_sampling_scale_beta = 1.
  config.prior_hparams_hparams.kernel_sampling_amplitude_alpha = 0.1
  config.prior_hparams_hparams.kernel_sampling_amplitude_beta = 0.4
  config.prior_hparams_hparams.kernel_sampling_lengthscale_alpha = 0.2
  config.prior_hparams_hparams.kernel_sampling_lengthscale_beta = 0.5
  config.kernel_name = 'ExponentiatedQuadratic'
  config.kernel_kwargs = ml_collections.ConfigDict()
  config.kernel_kwargs.amplitude = 0.2
  config.kernel_kwargs.length_scale = 0.3
  config.num_samples_gamma_profiles = 3
  config.num_samples_elbo = 3
  config.gp_jitter = 1e-3

  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs.init_value = 0.
  config.optim_kwargs.lr_schedule_kwargs.peak_value = 3e-4
  config.optim_kwargs.lr_schedule_kwargs.warmup_steps = 3_000
  config.optim_kwargs.lr_schedule_kwargs.transition_steps = 10_000
  config.optim_kwargs.lr_schedule_kwargs.decay_rate = 0.6
  config.optim_kwargs.lr_schedule_kwargs.transition_begin = 0
  config.optim_kwargs.lr_schedule_kwargs.staircase = False
  config.optim_kwargs.lr_schedule_kwargs.end_value = None
  
  config.floating_anchor_copies = False # CHECK ALWAYS!!!
  config.num_lp_anchor_train = 80
  config.num_lp_floating_train = 247
  config.num_items_keep = 71
  config.num_lp_anchor_val = 40
  config.num_lp_anchor_test = 0
  config.remove_empty_forms = True
  config.ad_hoc_val_profiles = False # CHECK ALWAYS!!!
  config.ad_hoc_val_list = [83, 104, 138, 94, 301, 348, 377, 441, 732, 1132, 1198, 1199, 1204, 1301,
  1327, 1329, 1330, 1332, 1345, 1348]

  config.eta_sampling_a = 0.5
  config.eta_sampling_b = 0.5
  # Use random location for anchor profiles for evaluation
  config.include_random_anchor = False


  # Random seed
  config.seed = 1

  return config
