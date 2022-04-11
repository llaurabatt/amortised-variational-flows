"""A simple example of variational SMI on the Random Effects model."""

import math
import pathlib

from absl import logging

import numpy as np
import scipy

from matplotlib import pyplot as plt

# from clu import metric_writers
from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

from tensorflow_probability.substrates import jax as tfp

import plot
from train_flow import (get_inducing_points, load_data, q_distr_global,
                        q_distr_loc_floating, q_distr_loc_random_anchor,
                        sample_all_flows, elbo_estimate,
                        error_locations_estimate)

from modularbayes import metaposterior
from modularbayes import utils
from modularbayes.utils.training import TrainState
from modularbayes.typing import (Any, Array, Batch, ConfigDict, Dict, List,
                                 Mapping, Optional, PRNGKey, SummaryWriter,
                                 Tuple, Union)

kernels = tfp.math.psd_kernels

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def make_optimizer(
    lr_schedule_name,
    lr_schedule_kwargs,
    grad_clip_value,
) -> optax.GradientTransformation:
  """Define optimizer to train the VHP map."""
  schedule = getattr(optax, lr_schedule_name)(**lr_schedule_kwargs)

  optimizer = optax.chain(*[
      optax.clip_by_global_norm(max_norm=grad_clip_value),
      optax.adabelief(learning_rate=schedule),
  ])
  return optimizer


# Define Meta-Posterior map
# Produce flow parameters as a function of eta
@hk.without_apply_rng
@hk.transform
def vmp_map(eta, vmp_map_name, vmp_map_kwargs, params_flow_init):
  return getattr(metaposterior, vmp_map_name)(
      **vmp_map_kwargs, params_flow_init=params_flow_init)(
          eta)


def loss(
    params_tuple: Tuple[hk.Params],
    batch: Batch,
    prng_key: PRNGKey,
    num_samples_eta: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    kernel_name: str,
    kernel_kwargs: Dict[str, Any],
    gp_jitter: float,
    include_random_anchor: bool,
    num_samples_gamma_profiles: int,
    num_samples_flow: int,
    vmp_map_name: str,
    vmp_map_kwargs: Dict[str, Any],
    params_flow_init_list: List[hk.Params],
    eta_sampling_a: float,
    eta_sampling_b: float,
) -> Array:
  """Define training loss function."""

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample eta values for floating profiles
  key_eta = next(prng_seq)

  eta_vmp = jax.random.beta(
      key=key_eta,
      a=eta_sampling_a,
      b=eta_sampling_b,
      shape=(num_samples_eta, 1),
  )

  # Get variational parameters corresponding to each eta
  params_flow_vmap_list = [
      vmp_map.apply(
          params,
          eta=eta_vmp,
          vmp_map_name=vmp_map_name,
          vmp_map_kwargs=vmp_map_kwargs,
          params_flow_init=params_flow_init,
      ) for params, params_flow_init in zip(params_tuple, params_flow_init_list)
  ]

  # Sample from posterior #

  # Use same key for every eta
  # (same key implies same samples from the base distribution)
  key_flow = next(prng_seq)

  # Generate sample from posteriors
  q_distr_out_vmap = jax.vmap(lambda params_tuple: sample_all_flows(
      params_tuple=params_tuple,
      batch=batch,
      prng_key=key_flow,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples_flow,),
      include_random_anchor=include_random_anchor,
      kernel_name=kernel_name,
      kernel_kwargs=kernel_kwargs,
      num_samples_gamma_profiles=num_samples_gamma_profiles,
      gp_jitter=gp_jitter,
  ))(
      params_flow_vmap_list)

  # Compute ELBO #

  # Use etas to create matrix of eta's per profile
  num_profiles_total = (
      flow_kwargs.num_profiles_anchor + flow_kwargs.num_profiles_floating)
  eta_profiles = jnp.broadcast_to(eta_vmp,
                                  (num_samples_eta, num_profiles_total))
  eta_profiles = jnp.where(
      jnp.arange(num_profiles_total).reshape(1, -1) <
      flow_kwargs.num_profiles_anchor,
      1.,
      eta_profiles,
  )
  smi_eta_vmap = {
      'profiles': eta_profiles,
      'items': jnp.ones((num_samples_eta, len(flow_kwargs.num_forms_tuple))),
  }
  # globals().update(vmp_map_kwargs)

  elbo_dict = jax.vmap(lambda q_distr_out_vmap_i, smi_eta_i: elbo_fn(
      q_distr_out=q_distr_out_vmap_i,
      smi_eta=smi_eta_i,
      batch=batch,
      include_random_anchor=include_random_anchor,
  ))(q_distr_out_vmap, smi_eta_vmap)

  # Our loss is the Negative ELBO
  loss_avg = -(
      jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2'] +
                  elbo_dict['stage_3']))

  return loss_avg


def plot_vmp_map(
    state: TrainState,
    vmp_map_name: str,
    vmp_map_kwargs: Mapping[str, Any],
    params_flow_init: hk.Params,
    lambda_idx: Union[int, List[int]],
    eta_grid: Array,
    constant_lambda_ignore_plot: bool,
):
  """Visualize VMP map."""

  assert eta_grid.ndim == 2

  params_flow_grid = vmp_map.apply(
      state.params,
      eta=eta_grid,
      vmp_map_name=vmp_map_name,
      vmp_map_kwargs=vmp_map_kwargs,
      params_flow_init=params_flow_init,
  )

  num_leaves = len(jax.tree_util.tree_leaves(params_flow_grid))
  max_num_leaves = 20
  # Variational parameters
  lambda_all = [
      x.reshape((x.shape[0], -1)) for x in jax.tree_util.tree_leaves(
          params_flow_grid)[:min(max_num_leaves, num_leaves)]
  ]
  lambda_all = jnp.concatenate(lambda_all, axis=-1)

  # Ignore flat functions of eta
  if constant_lambda_ignore_plot:
    lambda_all = lambda_all[:,
                            jnp.where(
                                jnp.square(lambda_all - lambda_all[[0], :]).sum(
                                    axis=0) > 0.)[0]]

  fig, axs = plt.subplots(
      nrows=1,
      ncols=len(lambda_idx),
      figsize=(4 * (len(lambda_idx)), 3),
  )
  if len(lambda_idx) == 1:
    axs = [axs]

  if not lambda_all.shape[1] > 0:
    return fig, axs

  # Plot vmp-map as a function of eta
  for i, idx_i in enumerate(lambda_idx):
    axs[i].plot(eta_grid, lambda_all[:, idx_i])
    axs[i].set_xlabel('eta')
    axs[i].set_ylabel(f'lambda_{idx_i}')

  fig.tight_layout()

  return fig, axs


def log_images(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    config: ConfigDict,
    show_basis_fields: bool,
    show_linguistic_fields: bool,
    num_loc_random_anchor_plot: Optional[int],
    num_loc_floating_plot: Optional[int],
    show_mixing_weights: bool,
    show_loc_given_y: bool,
    use_gamma_anchor: bool,
    show_vmp_map: bool,
    eta_grid_len: int,
    params_flow_init_list: List[hk.Params],
    show_eval_metric: bool,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  # config.eta_plot = [0.001] + np.round(np.linspace(0., 1., 11)[1:], 2).tolist()

  eta_plot = jnp.array(config.eta_plot).reshape(-1, 1)

  assert eta_plot.ndim == 2

  # Produce flow parameters as a function of eta
  params_flow_vmap_list = [
      vmp_map.apply(
          state.params,
          eta=eta_plot,
          vmp_map_name=config.vmp_map_name,
          vmp_map_kwargs=config.vmp_map_kwargs,
          params_flow_init=params_flow_init,
      ) for state, params_flow_init in zip(state_list, params_flow_init_list)
  ]

  # Use same key for every eta
  # (same key implies same samples from the base distribution)
  key_flow = next(prng_seq)

  # Generate sample from posteriors
  q_distr_out_vmap = jax.vmap(lambda params_tuple: sample_all_flows(
      params_tuple=params_tuple,
      batch=batch,
      prng_key=key_flow,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_plot,),
      include_random_anchor=config.include_random_anchor,
      kernel_name=config.kernel_name,
      kernel_kwargs=config.kernel_kwargs,
      num_samples_gamma_profiles=(config.num_samples_gamma_profiles
                                  if use_gamma_anchor else 0),
      gp_jitter=config.gp_jitter,
  ))(
      params_flow_vmap_list)

  # Plot posterior samples
  for i, eta_i in enumerate(eta_plot):
    # i = 0; eta_i = eta_plot[i]
    posterior_sample_dict_i = jax.tree_util.tree_map(
        lambda x: x[i],  # pylint: disable=cell-var-from-loop
        q_distr_out_vmap['posterior_sample'],
    )

    plot.posterior_samples(
        posterior_sample_dict=posterior_sample_dict_i,
        batch=batch,
        prng_key=next(prng_seq),
        kernel_name=config.kernel_name,
        kernel_kwargs=config.kernel_kwargs,
        gp_jitter=config.gp_jitter,
        step=state_list[0].step,
        show_basis_fields=show_basis_fields,
        show_linguistic_fields=show_linguistic_fields,
        num_loc_random_anchor_plot=num_loc_random_anchor_plot,
        num_loc_floating_plot=num_loc_floating_plot,
        show_mixing_weights=show_mixing_weights,
        show_loc_given_y=show_loc_given_y,
        suffix=f"eta_floating_{float(eta_i):.3f}",
        summary_writer=summary_writer,
        workdir_png=workdir_png,
        use_gamma_anchor=use_gamma_anchor,
    )

  ### Visualize meta-posterior map along the Eta space ###
  if show_vmp_map:
    # Define elements to grate grid of eta values
    eta_grid = jnp.linspace(0, 1, eta_grid_len).reshape(-1, 1)

    images = []
    for i, state, params_flow_init in zip(
        range(len(state_list)),
        state_list,
        params_flow_init_list,
    ):
      plot_name = 'lalme_vmp_map_' + [
          'global', 'loc_floating', 'loc_random_anchor'
      ][i]
      fig, _ = plot_vmp_map(
          state=state,
          vmp_map_name=config.vmp_map_name,
          vmp_map_kwargs=config.vmp_map_kwargs,
          params_flow_init=params_flow_init,
          lambda_idx=np.array(config.lambda_idx_plot),
          eta_grid=eta_grid,
          constant_lambda_ignore_plot=config.constant_lambda_ignore_plot,
      )
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))

      if summary_writer:
        images.append(utils.plot_to_image(fig))

    # Logging VMP-map plots
    if summary_writer:
      plot_name = 'lalme_vmp_map'
      summary_writer.image(
          tag=plot_name,
          image=utils.misc.normalize_images(images),
          step=state_list[0].step,
      )

  ### Evaluation metrics ###
  if show_eval_metric:

    eta_grid_eval = jnp.linspace(0, 1, eta_grid_len).reshape(-1, 1)
    images = []

    # Produce flow parameters as a function of eta
    error_loc_dict = {
        'eta_floating': [],
        'distance_floating': [],
    }
    if config.include_random_anchor:
      error_loc_dict['distance_random_anchor'] = []

    key_flow = next(prng_seq)
    for i in range(eta_grid_len):
      eta_i = eta_grid_eval[[i], :]
      params_flow_eta_grid_eval = [
          vmp_map.apply(
              state.params,
              eta=eta_i,
              vmp_map_name=config.vmp_map_name,
              vmp_map_kwargs=config.vmp_map_kwargs,
              params_flow_init=params_flow_init,
          )
          for state, params_flow_init in zip(state_list, params_flow_init_list)
      ]

      # Use same key for every eta
      # (same key implies same samples from the base distribution)
      q_distr_out_i = jax.vmap(lambda params_tuple_i: sample_all_flows(
          params_tuple=params_tuple_i,
          batch=batch,
          prng_key=key_flow,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          sample_shape=(config.num_samples_eval,),
          include_random_anchor=config.include_random_anchor,
          kernel_name=config.kernel_name,
          kernel_kwargs=config.kernel_kwargs,
          num_samples_gamma_profiles=config.num_samples_gamma_profiles,
          gp_jitter=config.gp_jitter,
      ))(
          params_flow_eta_grid_eval)

      # Error on posterior location
      error_loc_dict['eta_floating'].append(float(eta_i))
      error_loc_dict_i = jax.vmap(
          compute_error_locations,
          in_axes=[0, None])(q_distr_out_i['posterior_sample'], batch)
      error_loc_dict['distance_floating'].append(
          float(error_loc_dict_i['distance_floating']))
      if config.include_random_anchor:
        error_loc_dict['distance_random_anchor'].append(
            float(error_loc_dict_i['distance_random_anchor']))

    plot_name = 'lalme_vmp_distance_floating'
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    # Plot distance_floating as a function of eta
    axs.plot(error_loc_dict['eta_floating'],
             error_loc_dict['distance_floating'])
    axs.set_xlabel('eta_floating')
    axs.set_ylabel('Mean distance')
    axs.set_title('Mean distance for floating profiles\n' +
                  '(posterior vs. linguistic guess)')
    fig.tight_layout()
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      images.append(utils.plot_to_image(fig))

    if config.include_random_anchor:
      plot_name = 'lalme_vmp_distance_random_anchor'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance_floating as a function of eta
      axs.plot(error_loc_dict['eta_floating'],
               error_loc_dict['distance_random_anchor'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean distance')
      axs.set_title('Mean distance for anchor profiles\n' +
                    '(posterior vs. real location)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(utils.plot_to_image(fig))

      if summary_writer:
        plot_name = 'lalme_vmp_distance'
        summary_writer.image(
            tag=plot_name,
            image=utils.misc.normalize_images(images),
            step=state_list[0].step,
        )


def train_and_evaluate(config: ConfigDict, workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)

  # Full dataset used everytime
  # No batching for now
  dataset = load_data(config=config)
  # Add some parameters to config
  config.num_profiles = dataset['num_profiles']
  config.num_profiles_anchor = dataset['num_profiles_anchor']
  config.num_profiles_floating = dataset['num_profiles_floating']
  config.num_forms_tuple = dataset['num_forms_tuple']
  config.num_inducing_points = math.prod(config.flow_kwargs.inducing_grid_shape)

  # For training, we need a Dictionary compatible with jit
  # we remove string vector
  train_ds = {k: v for k, v in dataset.items() if k not in ['items', 'forms']}

  # Compute GP covariance between anchor profiles
  train_ds['cov_anchor'] = getattr(
      kernels, config.kernel_name)(**config.kernel_kwargs).matrix(
          x1=train_ds['loc'][:train_ds['num_profiles_anchor'], :],
          x2=train_ds['loc'][:train_ds['num_profiles_anchor'], :],
      )

  train_ds = get_inducing_points(dataset=train_ds, config=config)

  # These parameters affect the dimension of the flow
  # so they are also part of the flow parameters
  config.flow_kwargs.num_profiles_anchor = dataset['num_profiles_anchor']
  config.flow_kwargs.num_profiles_floating = dataset['num_profiles_floating']
  config.flow_kwargs.num_forms_tuple = dataset['num_forms_tuple']
  config.flow_kwargs.num_inducing_points = int(
      math.prod(config.flow_kwargs.inducing_grid_shape))
  config.flow_kwargs.is_smi = True
  # Get locations bounds
  # These define the range of values produced by the posterior of locations
  loc_bounds = np.stack(
      [dataset['loc'].min(axis=0), dataset['loc'].max(axis=0)],
      axis=1).astype(np.float32)
  config.flow_kwargs.loc_x_range = tuple(loc_bounds[0])
  config.flow_kwargs.loc_y_range = tuple(loc_bounds[1])

  # smi_eta = {'groups':jnp.ones((1,2))}
  # elbo_fn(config.params_flow, next(prng_seq), train_ds, smi_eta)

  # Get examples of the output tree to be produced by the meta functions
  # Global parameters
  params_flow_init_list = []

  state_flow_init_path_list = [
      config.state_global_init, config.state_loc_floating_init
  ]
  if config.include_random_anchor:
    state_flow_init_path_list.append(config.state_loc_random_anchor_init)

  if state_flow_init_path_list[0] == '':
    params_flow_init_list.append(
        hk.transform(q_distr_global).init(
            next(prng_seq),
            flow_name=config.flow_name,
            flow_kwargs=config.flow_kwargs,
            sample_shape=(config.num_samples_elbo,),
        ))
  else:
    state_flow_init = utils.load_ckpt(path=state_flow_init_path_list[0])
    params_flow_init_list.append(state_flow_init.params)
    del state_flow_init

  # Get an initial sample of global parameters
  # (used below to initialize floating locations)
  global_params_base_sample_init = hk.transform(q_distr_global).apply(
      params_flow_init_list[0],
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_elbo,),
  )['global_params_base_sample']

  # Location floating profiles
  if state_flow_init_path_list[1] == '':
    params_flow_init_list.append(
        hk.transform(q_distr_loc_floating).init(
            next(prng_seq),
            flow_name=config.flow_name,
            flow_kwargs=config.flow_kwargs,
            global_params_base_sample=global_params_base_sample_init,
        ))
  else:
    state_flow_init = utils.load_ckpt(path=state_flow_init_path_list[1])
    params_flow_init_list.append(state_flow_init.params)
    del state_flow_init

  # Location random anchor profiles
  if config.include_random_anchor:
    if state_flow_init_path_list[2] == '':
      params_flow_init_list.append(
          hk.transform(q_distr_loc_random_anchor).init(
              next(prng_seq),
              flow_name=config.flow_name,
              flow_kwargs=config.flow_kwargs,
              global_params_base_sample=global_params_base_sample_init,
          ))
    else:
      state_flow_init = utils.load_ckpt(path=state_flow_init_path_list[2])
      params_flow_init_list.append(state_flow_init.params)
      del state_flow_init

  ### Set Variational Meta-Posterior Map ###

  # eta knots
  if config.vmp_map_name in ['VmpGP', 'VmpCubicSpline']:
    # grid of percentile values from a beta distribution
    config.vmp_map_kwargs['eta_knots'] = scipy.stats.beta.ppf(
        jnp.linspace(0, 1., config.vmp_map_kwargs.num_knots).reshape(-1, 1),
        a=config.eta_sampling_a,
        b=config.eta_sampling_b,
    ).tolist()
    del config.vmp_map_kwargs['num_knots']

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')

  state_list = []
  state_name_list = []

  state_name_list.append('global')
  state_list.append(
      utils.initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=vmp_map,
          forward_fn_kwargs={
              'eta': jnp.ones((config.num_samples_eta, 1)),
              'vmp_map_name': config.vmp_map_name,
              'vmp_map_kwargs': config.vmp_map_kwargs,
              'params_flow_init': params_flow_init_list[0],
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  state_name_list.append('loc_floating')
  state_list.append(
      utils.initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=vmp_map,
          forward_fn_kwargs={
              'eta': jnp.ones((config.num_samples_eta, 1)),
              'vmp_map_name': config.vmp_map_name,
              'vmp_map_kwargs': config.vmp_map_kwargs,
              'params_flow_init': params_flow_init_list[1],
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0 and state_list[0].step < config.training_steps:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(utils.flatten_dict(config))
  else:
    summary_writer = None

  # Print a useful summary of the execution of the VHP-map architecture.
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda state_i, params_flow_init_i: vmp_map.apply(
          state_i.params,
          eta=jnp.ones((config.num_samples_eta, 1)),
          vmp_map_name=config.vmp_map_name,
          vmp_map_kwargs=config.vmp_map_kwargs,
          params_flow_init=params_flow_init_i),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  logging.info('VMP-MAP GLOBAL PARAMETERS:')
  summary = tabulate_fn_(state_list[0], params_flow_init_list[0])
  for line in summary.split("\n"):
    logging.info(line)

  logging.info('VMP-MAP LOCATION FLOATING PROFILES:')
  summary = tabulate_fn_(state_list[1], params_flow_init_list[1])
  for line in summary.split("\n"):
    logging.info(line)

  if config.include_random_anchor:
    state_name_list.append('loc_random_anchor')
    state_list.append(
        utils.initial_state_ckpt(
            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
            forward_fn=vmp_map,
            forward_fn_kwargs={
                'eta': jnp.ones((config.num_samples_eta, 1)),
                'vmp_map_name': config.vmp_map_name,
                'vmp_map_kwargs': config.vmp_map_kwargs,
                'params_flow_init': params_flow_init_list[2],
            },
            prng_key=next(prng_seq),
            optimizer=make_optimizer(**config.optim_kwargs),
        ))
    logging.info('VMP-MAP RANDOM LOCATION OF ANCHOR PROFILES:')
    summary = tabulate_fn_(state_list[2], params_flow_init_list[2])
    for line in summary.split("\n"):
      logging.info(line)

  ### Training VMP map ###
  update_states_jit = lambda state_list, batch, prng_key: utils.update_states(
      state_list=state_list,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer(**config.optim_kwargs),
      loss_fn=loss,
      loss_fn_kwargs={
          'num_samples_eta': config.num_samples_eta,
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'kernel_name': config.kernel_name,
          'kernel_kwargs': config.kernel_kwargs,
          'gp_jitter': config.gp_jitter,
          'include_random_anchor': config.include_random_anchor,
          'num_samples_gamma_profiles': config.num_samples_gamma_profiles,
          'num_samples_flow': config.num_samples_elbo,
          'vmp_map_name': config.vmp_map_name,
          'vmp_map_kwargs': config.vmp_map_kwargs,
          'params_flow_init_list': params_flow_init_list,
          'eta_sampling_a': config.eta_sampling_a,
          'eta_sampling_b': config.eta_sampling_b,
      },
  )
  # globals().update(loss_fn_kwargs)

  update_states_jit = jax.jit(update_states_jit)

  if state_list[0].step < config.training_steps:
    logging.info('Training VMP-map...')

  while state_list[0].step < config.training_steps:

    # Plots to monitor training
    if (config.log_img_steps
        is not None) and ((state_list[0].step in [0, 1]) or
                          (state_list[0].step % config.log_img_steps == 0)):
      # print("Logging images...\n")
      log_images(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
          config=config,
          show_basis_fields=False,
          show_linguistic_fields=False,
          num_loc_random_anchor_plot=5,
          num_loc_floating_plot=5,
          show_mixing_weights=False,
          show_loc_given_y=False,
          use_gamma_anchor=False,
          show_vmp_map=True,
          eta_grid_len=10,
          params_flow_init_list=params_flow_init_list,
          show_eval_metric=True,
          summary_writer=summary_writer,
          workdir_png=workdir,
      )
      plt.close()

    # Log learning rate
    summary_writer.scalar(
        tag='learning_rate',
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(state_list[0].step),
        step=state_list[0].step,
    )

    state_list, metrics = update_states_jit(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
    )

    # The computed training loss would correspond to the model before update
    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state_list[0].step - 1,
    )

    if state_list[0].step == 2:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    if state_list[0].step % config.eval_steps == 0:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    if (state_list[0].step) % config.checkpoint_steps == 0:
      for state, state_name in zip(state_list, state_name_list):
        utils.save_checkpoint(
            state=state,
            checkpoint_dir=f'{checkpoint_dir}/{state_name}',
            keep=config.checkpoints_keep,
        )

    # Wait until computations are done before the next step
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info('Final training step: %i', state_list[0].step)

  # Saving checkpoint at the end of the training process
  # (in case training_steps is not multiple of checkpoint_steps)
  for state, state_name in zip(state_list, state_name_list):
    utils.save_checkpoint(
        state=state,
        checkpoint_dir=f'{checkpoint_dir}/{state_name}',
        keep=config.checkpoints_keep,
    )

  # Last plot of posteriors
  log_images(
      state_list=state_list,
      batch=train_ds,
      prng_key=jax.random.PRNGKey(config.seed),
      config=config,
      show_basis_fields=True,
      show_linguistic_fields=True,
      num_loc_random_anchor_plot=20,
      num_loc_floating_plot=20,
      show_mixing_weights=False,
      show_loc_given_y=False,
      use_gamma_anchor=False,
      show_vmp_map=True,
      eta_grid_len=20,
      params_flow_init_list=params_flow_init_list,
      show_eval_metric=True,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )
  plt.close()

  return state


# # For debugging
# config = get_config()
# # workdir = pathlib.Path.home() / 'smi/output/debug'
# workdir = pathlib.Path.home() / 'smi/output/lalme/spline/vmp_gp'
# config.state_global_init = str(pathlib.Path.home() /
#         ('smi/output/lalme/spline/' +
#          'eta_floating_0.500/checkpoints/global/ckpt_030000'))
# config.state_loc_floating_init = str(pathlib.Path.home() /
#         ('smi/output/lalme/spline/' +
#          'eta_floating_0.500/checkpoints/loc_floating/ckpt_030000'))
# config.state_loc_random_anchor_init = str(pathlib.Path.home() /
#         ('smi/output/lalme/spline/' +
#          'eta_floating_0.500/checkpoints/loc_random_anchor/ckpt_030000'
#         ))
# train_and_evaluate(config, workdir)
