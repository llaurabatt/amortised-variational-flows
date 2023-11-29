"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'vmp_flow_allhp'

  # Dataset to use
  config.dataset_id = 'coarsen_all_items'

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

  # Define priors
  config.prior_hparams = ml_collections.ConfigDict()
  config.prior_hparams.mu_prior_concentration = 1.
  config.prior_hparams.mu_prior_rate = 0.5
  config.prior_hparams.zeta_prior_a = 1.
  config.prior_hparams.zeta_prior_b = 1.
  config.prior_hparams.w_prior_scale = 5.
  config.prior_hparams.a_prior_scale = 10.

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
  config.gp_jitter = 1e-3

  # Number of training steps to run.
  config.training_steps = 100_000 

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 3e-4,
      'warmup_steps': 3_000,
      'transition_steps': 10_000,
      'decay_rate': 0.6,
      'transition_begin': 0,
      'staircase': False,
      'end_value': None,
  }

  # Optimizer for searching hp
  config.optim_kwargs_hp = ml_collections.ConfigDict()
  config.optim_kwargs_hp.learning_rate = 1e-4
  config.hp_star_steps = 10_000
  config.cond_hparams_names = ['w_prior_scale', 'a_prior_scale', 'kernel_amplitude', 'kernel_length_scale', 'eta']

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

  # Number of posteriors samples to approximate the variational loss (ELBO).
  config.num_samples_elbo = 3
  config.num_samples_gamma_profiles = 3

  # How often to evaluate the model.
  config.eval_steps = config.training_steps // 2#5 
  config.num_samples_eval = 500
  config.num_samples_mse = 2_000
  config.eval_last = True 
  config.max_steps_nan = 1_000

  # How often to log images to monitor convergence.
  config.log_img_steps = config.training_steps // 2#5 
  config.log_img_at_end = True  
  config.save_samples = False # FLIPPED
  config.path_mcmc_img = ''

  # Number of samples used in the plots.
  config.num_samples_plot = 10_000 #2_000 for basis fields
  config.num_samples_chunk_plot = 500 #100 for basis fields

  # Floating profiles to plot in grid
  config.lp_floating_grid10 = [
      136, 234, 1002, 501, 236, 237, 319, 515, 699, 755
  ]
# TRAIN LPs: [  81   82   85   87   89   90   91   94   96   97  105  113  145  162
#   195  211  233  251  308  311  313  317  318  349  360  366  373  378
#   380  415  417  418  445  451  542  585  586  612  731  777 1001 1012
#  1102 1122 1126 1127 1128 1133 1135 1136 1140 1141 1200 1201 1202 1203
#  1245 1257 1259 1284 1285 1286 1287 1306 1307 1308 1321 1323 1324 1325
#  1326 1331 1334 1335 1337 1338 1340 1342 1343 1355]
#  VAL LPs: [  83  104  138  294  301  348  377  441  732 1132 1198 1199 1204 1301
#  1327 1329 1330 1332 1345 1348]
#  TEST LPs: [  84   88  133  139  307  363  446  448  472  544  617  770  814 1125
#  1134 1142 1205 1302 1339 1341]
#  FLOATING LPs: [   1    2    3    4    5   16   17   23   26   29   30   32   36   38
#    43   45   46   49   51   52   54   55   56   60   61   62   65   68
#    69   70   71   73   75   79   80   99  100  106  110  114  115  130
#   136  140  154  164  165  167  168  169  175  177  180  181  183  184
#   186  188  189  192  193  194  196  198  200  201  202  204  206  207
#   210  212  213  215  217  219  220  221  222  223  225  226  227  234
#   235  236  237  238  240  243  246  247  257  259  260  277  278  287
#   299  300  302  303  314  316  319  320  322  325  357  358  361  365
#   382  398  405  410  411  419  422  423  425  426  432  434  435  454
#   461  473  474  476  477  479  488  491  492  494  495  496  497  498
#   500  501  503  504  505  506  507  508  509  510  511  512  514  515
#   516  517  519  527  529  530  531  534  536  537  539  540  541  549
#   550  551  552  553  554  556  557  558  559  560  561  577  578  579
#   580  581  582  583  584  587  588  591  593  597  603  605  652  661
#   676  677  678  699  704  709  714  715  717  718  726  729  730  736
#   737  738  742  752  755  761  763  764  766  767  804  901  905  908
#   910  912  913  927 1002 1300 1352 4218 4239 4245 4285 4286 4289 4675
#  4682 4685 7550 7591 7592 7593 7600 7610 7980]
  # config.lp_random_anchor_10 = [83, 104, 138, 251, 307, 349, 378, 441, 732, 1132]
  config.lp_random_anchor_10 = None

  # eta shown in figures
  config.eta_plot = [1.0] #[0.001, 0.25, 0.5, 0.75, 1.0]
  config.prior_hparams_plot = [[5., 10., 1., 0.5, 1., 1., 0.2, 0.3],
                               [1., 4., 1., 0.5, 1., 1., 0.5, 0.9],
                               [8., 15., 1., 0.5, 1., 1., 0.1, 0.05]]

  # How often to save model checkpoints.
  config.checkpoint_steps = config.training_steps // 5
  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Number of samples of eta for Meta-Posterior training
  config.eta_sampling_a = 0.5
  config.eta_sampling_b = 0.5

  # Use random location for anchor profiles for evaluation
  config.include_random_anchor = False
  # Metric for Hyperparameter Optimization
  config.synetune_metric = "mean_dist_anchor_val_min"

  # Random seed
  config.seed = 1

  return config
