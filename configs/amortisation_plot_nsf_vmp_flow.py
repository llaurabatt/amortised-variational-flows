"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Dataset to use
  config.dataset_id = 'coarsen_all_items'
  config.workdir_VMP = ''
  config.workdir_AdditiveVMP = ''
  config.workdirs_VP = []

  config.cond_hparams_names = ['w_prior_scale', 'a_prior_scale', 'kernel_amplitude', 'kernel_length_scale', 'eta']
  config.etas = [0., 0.25, 0.5, 0.75, 1.]
  config.prior_hparams_fixed = [5., 10., 1., 0.5, 1., 1., 0.2, 0.3]
  config.num_samples_amortisation_plot = 1_000

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


  # Use random location for anchor profiles for evaluation
  config.include_random_anchor = False


  # Random seed
  config.seed = 1

  return config
