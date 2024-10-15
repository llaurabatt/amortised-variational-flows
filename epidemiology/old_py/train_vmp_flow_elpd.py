"""A simple example of a flow model trained on Epidemiology data."""
import pathlib

from absl import logging

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

import flows
import log_prob_fun
import plot
from train_flow import load_dataset, make_optimizer

from modularbayes._src.utils.training import TrainState
from modularbayes import (plot_to_image, normalize_images, flatten_dict, initial_state_ckpt, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict, List,
                                      Mapping, Optional, PRNGKey, SmiEta, 
                                      SummaryWriter, Tuple)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def q_distr_phi(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta: Array,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  # Define normalizing flows
  q_distr = getattr(flows, flow_name + '_phi')(**flow_kwargs)

  num_samples = eta.shape[0]

  # Sample from flows
  (phi_sample, phi_log_prob_posterior,
   phi_base_sample) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=[eta, None],
   )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows.split_flow_phi(
          samples=phi_sample,
          **flow_kwargs,
      ))

  # sample from base distribution that generated phi
  q_distr_out['phi_base_sample'] = phi_base_sample

  # log P(phi)
  q_distr_out['phi_log_prob'] = phi_log_prob_posterior

  return q_distr_out


def q_distr_theta(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    phi_base_sample: Array,
    eta: Array,
    is_aux: bool,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  num_samples = phi_base_sample.shape[0]

  # Define normalizing flows
  q_distr = getattr(flows, flow_name + '_theta')(**flow_kwargs)

  # Sample from flow
  (theta_sample, theta_log_prob_posterior) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
      context=[eta, phi_base_sample],
  )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows.split_flow_theta(
          samples=theta_sample,
          is_aux=is_aux,
          **flow_kwargs,
      ))

  # log P(theta|phi)
  q_distr_out['theta_' + ('aux_' if is_aux else '') +
              'log_prob'] = theta_log_prob_posterior

  return q_distr_out


def sample_all_flows(
    params_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    smi_eta: SmiEta,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  prng_seq = hk.PRNGSequence(prng_key)

  # phi
  q_distr_out = hk.transform(q_distr_phi).apply(
      params_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      eta=smi_eta['modules'],
  )

  # theta
  q_distr_out_theta = hk.transform(q_distr_theta).apply(
      params_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      phi_base_sample=q_distr_out['phi_base_sample'],
      eta=smi_eta['modules'],
      is_aux=False,
  )
  q_distr_out['posterior_sample'].update(q_distr_out_theta['posterior_sample'])
  q_distr_out['theta_log_prob'] = q_distr_out_theta['theta_log_prob']

  q_distr_out_theta_aux = hk.transform(q_distr_theta).apply(
      params_tuple[2],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      phi_base_sample=q_distr_out['phi_base_sample'],
      eta=smi_eta['modules'],
      is_aux=True,
  )
  q_distr_out['posterior_sample'].update(
      q_distr_out_theta_aux['posterior_sample'])
  q_distr_out['theta_aux_log_prob'] = q_distr_out_theta_aux[
      'theta_aux_log_prob']

  return q_distr_out


def elbo_estimate_along_eta(
    params_tuple: Tuple[hk.Params],
    batch: Batch,
    prng_key: PRNGKey,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta_sampling_a: float,
    eta_sampling_b: float,
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample eta values (only for Y module)
  etas_elbo = jax.random.beta(
      key=next(prng_seq),
      a=eta_sampling_a,
      b=eta_sampling_b,
      shape=(num_samples, 1),
  )
  # Set eta_z=1
  etas_elbo = jnp.concatenate([jnp.ones_like(etas_elbo), etas_elbo], axis=-1)
  smi_eta_elbo = {'modules': etas_elbo}

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      prng_key=next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      smi_eta=smi_eta_elbo,
  )

  shared_params_names = [
      'phi',
  ]
  refit_params_names = [
      'theta',
  ]

  # ELBO stage 1: Power posterior
  posterior_sample_dict_stg1 = {}
  for key in shared_params_names:
    posterior_sample_dict_stg1[key] = q_distr_out['posterior_sample'][key]
  for key in refit_params_names:
    posterior_sample_dict_stg1[key] = q_distr_out['posterior_sample'][key +
                                                                      '_aux']

  log_prob_joint_stg1 = jax.vmap(
      lambda posterior_sample_i, smi_eta_i: log_prob_fun.log_prob_joint(
          batch=batch,
          posterior_sample_dict=posterior_sample_i,
          smi_eta=smi_eta_i,
      ))(
          jax.tree_map(lambda x: jnp.expand_dims(x, 1),
                       posterior_sample_dict_stg1),
          smi_eta_elbo,
      )

  log_q_stg1 = q_distr_out['phi_log_prob'] + q_distr_out['theta_aux_log_prob']

  #TODO: check reshape
  elbo_stg1 = log_prob_joint_stg1.reshape(-1) - log_q_stg1

  # ELBO stage 2: Refit theta
  posterior_sample_dict_stg2 = {}
  for key in shared_params_names:
    posterior_sample_dict_stg2[key] = jax.lax.stop_gradient(
        q_distr_out['posterior_sample'][key])
  for key in refit_params_names:
    posterior_sample_dict_stg2[key] = q_distr_out['posterior_sample'][key]

  log_prob_joint_stg2 = jax.vmap(
      lambda posterior_sample_i: log_prob_fun.log_prob_joint(
          batch=batch,
          posterior_sample_dict=posterior_sample_i,
          smi_eta=None,
      ))(
          jax.tree_map(lambda x: jnp.expand_dims(x, 1),
                       posterior_sample_dict_stg2))

  log_q_stg2 = (
      jax.lax.stop_gradient(q_distr_out['phi_log_prob']) +
      q_distr_out['theta_log_prob'])

  # TODO: check reshape
  elbo_stg2 = log_prob_joint_stg2.reshape(-1) - log_q_stg2

  elbo_dict = {'stage_1': elbo_stg1, 'stage_2': elbo_stg2}

  return elbo_dict


def loss(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_estimate_along_eta(
      params_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


#########################################################################################################
def compute_elpd(
    posterior_sample_dict: Dict[str, Any],
    batch: Batch,
) -> Mapping[str, Array]:
  """Compute ELPD.

  Estimates the ELPD based on two Monte Carlo approximations:
    1) Using WAIC.

  Args:
    posterior_sample_dict: Dictionary of posterior samples.
    batch: Batch of data (the one that was for training).

  Returns:
    Dictionary of ELPD estimates, with keys:
      - 'elpd_waic_pointwise': WAIC-based estimate.
  """

  # Initialize dictionary for output
  elpd_out = {}

  num_samples, _ = posterior_sample_dict['phi'].shape # TOCHECK: is it correct?

  ### WAIC ###
#   # Compute LPD

  loglik_pointwise_insample = log_prob_fun.log_lik_vectorised(
      batch['Z'],
      batch['Y'],
      batch['N'],
      batch['T'],
      posterior_sample_dict['phi'],
      posterior_sample_dict['theta'],
  ).sum(2)

  lpd_pointwise = jax.scipy.special.logsumexp(
      loglik_pointwise_insample, axis=0) - jnp.log(num_samples)
  elpd_out['lpd_pointwise'] = lpd_pointwise

  p_waic_pointwise = jnp.var(loglik_pointwise_insample, axis=0)
  elpd_out['p_waic_pointwise'] = p_waic_pointwise

  elpd_waic_pointwise = lpd_pointwise - p_waic_pointwise
  elpd_out['elpd_waic_pointwise'] = elpd_waic_pointwise


  loglik_pointwise_y_insample = log_prob_fun.log_lik_y_vectorised(
      batch['Y'],
      batch['T'],
      posterior_sample_dict['phi'],
      posterior_sample_dict['theta'],
  )
  lpd_pointwise_y = jax.scipy.special.logsumexp(
      loglik_pointwise_y_insample, axis=0) - jnp.log(num_samples)
  elpd_out['lpd_pointwise_y'] = lpd_pointwise_y

  p_waic_pointwise_y = jnp.var(loglik_pointwise_y_insample, axis=0)
  elpd_out['p_waic_pointwise_y'] = p_waic_pointwise_y

  elpd_waic_pointwise_y = lpd_pointwise_y - p_waic_pointwise_y
  elpd_out['elpd_waic_pointwise_y'] = elpd_waic_pointwise_y


  loglik_pointwise_z_insample = log_prob_fun.log_lik_z_vectorised(
      batch['Z'],
      batch['N'],
      posterior_sample_dict['phi'],
  )
  lpd_pointwise_z = jax.scipy.special.logsumexp(
      loglik_pointwise_z_insample, axis=0) - jnp.log(num_samples)
  elpd_out['lpd_pointwise_z'] = lpd_pointwise_z

  p_waic_pointwise_z = jnp.var(loglik_pointwise_z_insample, axis=0)
  elpd_out['p_waic_pointwise_z'] = p_waic_pointwise_z

  elpd_waic_pointwise_z = lpd_pointwise_z - p_waic_pointwise_z
  elpd_out['elpd_waic_pointwise_z'] = elpd_waic_pointwise_z

  return elpd_out


compute_elpd_jit = jax.jit(compute_elpd)


def elpd_estimate_pointwise(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    num_samples: int,
    eta: Array,
):
  q_distr_out_i = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=prng_key,  # same key to reduce variance of posterior along eta
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      smi_eta={
          'modules': jnp.broadcast_to(eta, (num_samples, 2)) # num modules
      },  
)
  elpd_dict = compute_elpd_jit(
      posterior_sample_dict=q_distr_out_i['posterior_sample'],
      batch=batch,
  )
  return elpd_dict

def elpd_points(
    state_list: List[TrainState],
    batch: Batch, # train_ds
    prng_key: PRNGKey,
    config: ConfigDict,
    eta_grid: Array,
    use_vmap: bool = True,
):
  """Visualize ELPD surface as function of eta."""

  assert eta_grid.ndim == 2

  num_groups, *grid_shape = eta_grid.shape

  # TODO: vmap implementation produces RuntimeError: RESOURCE_EXHAUSTED
  lpd_pointwise_all_eta = []
  p_waic_pointwise_all_eta = []
  elpd_waic_pointwise_all_eta = []

  lpd_pointwise_y_all_eta = []
  p_waic_pointwise_y_all_eta = []
  elpd_waic_pointwise_y_all_eta = []

  lpd_pointwise_z_all_eta = []
  p_waic_pointwise_z_all_eta = []
  elpd_waic_pointwise_z_all_eta = []

  if use_vmap:
    # Faster option: using vmap
    # Sometimes fail due to RuntimeError: RESOURCE_EXHAUSTED
    eta_grid_reshaped = eta_grid.reshape(num_groups, -1).T

    elpd_dict_all = jax.vmap(lambda eta_i: elpd_estimate_pointwise(
        state_list=state_list,
        batch=batch,
        prng_key=prng_key,
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        num_samples=config.num_samples_elpd,
        eta=eta_i,
    ))(eta_grid_reshaped)

    print('vmap done')
        

    lpd_pointwise_all_eta = elpd_dict_all['lpd_pointwise']
    p_waic_pointwise_all_eta = elpd_dict_all['p_waic_pointwise']
    elpd_waic_pointwise_all_eta = elpd_dict_all['elpd_waic_pointwise']

    lpd_pointwise_y_all_eta = elpd_dict_all['lpd_pointwise_y']
    p_waic_pointwise_y_all_eta = elpd_dict_all['p_waic_pointwise_y']
    elpd_waic_pointwise_y_all_eta = elpd_dict_all['elpd_waic_pointwise_y']

    lpd_pointwise_z_all_eta = elpd_dict_all['lpd_pointwise_z']
    p_waic_pointwise_z_all_eta = elpd_dict_all['p_waic_pointwise_z']
    elpd_waic_pointwise_z_all_eta = elpd_dict_all['elpd_waic_pointwise_z']
  else:
    # Slower option: for loop
    # Takes longer to compute
    eta_grid_reshaped = eta_grid.reshape(num_groups, -1).T
    for eta_i in zip(eta_grid_reshaped):
      # eta_i = (eta_grid.reshape(num_groups, -1).T)[0]

      elpd_dict_i = elpd_estimate_pointwise(
          state_list=state_list,
          batch=batch,
          prng_key=prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          num_samples=config.num_samples_elpd,
          eta=eta_i,
      )
      lpd_pointwise_all_eta.append(elpd_dict_i['lpd_pointwise'])
      p_waic_pointwise_all_eta.append(elpd_dict_i['p_waic_pointwise'])
      elpd_waic_pointwise_all_eta.append(elpd_dict_i['elpd_waic_pointwise'])

      lpd_pointwise_y_all_eta.append(elpd_dict_i['lpd_pointwise_y'])
      p_waic_pointwise_y_all_eta.append(elpd_dict_i['p_waic_pointwise_y'])
      elpd_waic_pointwise_y_all_eta.append(elpd_dict_i['elpd_waic_pointwise_y'])

      lpd_pointwise_z_all_eta.append(elpd_dict_i['lpd_pointwise_z'])
      p_waic_pointwise_z_all_eta.append(elpd_dict_i['p_waic_pointwise_z'])
      elpd_waic_pointwise_z_all_eta.append(elpd_dict_i['elpd_waic_pointwise_z'])

    lpd_pointwise_all_eta = jnp.stack(lpd_pointwise_all_eta, axis=0)
    p_waic_pointwise_all_eta = jnp.stack(p_waic_pointwise_all_eta, axis=0)
    elpd_waic_pointwise_all_eta = jnp.stack(elpd_waic_pointwise_all_eta, axis=0)

    lpd_pointwise_y_all_eta = jnp.stack(lpd_pointwise_y_all_eta, axis=0)
    p_waic_pointwise_y_all_eta = jnp.stack(p_waic_pointwise_y_all_eta, axis=0)
    elpd_waic_pointwise_y_all_eta = jnp.stack(elpd_waic_pointwise_y_all_eta, axis=0)

    lpd_pointwise_z_all_eta = jnp.stack(lpd_pointwise_z_all_eta, axis=0)
    p_waic_pointwise_z_all_eta = jnp.stack(p_waic_pointwise_z_all_eta, axis=0)
    elpd_waic_pointwise_z_all_eta = jnp.stack(elpd_waic_pointwise_z_all_eta, axis=0)

  # Add pointwise elpd and lpd across observations
  lpd_all_eta = lpd_pointwise_all_eta.sum(axis=-1).reshape(grid_shape)
  p_waic_all_eta = p_waic_pointwise_all_eta.sum(axis=-1).reshape(grid_shape)
  elpd_waic_all_eta = elpd_waic_pointwise_all_eta.sum(
      axis=-1).reshape(grid_shape)

  lpd_y_all_eta = lpd_pointwise_y_all_eta.sum(axis=-1).reshape(grid_shape)
  p_waic_y_all_eta = p_waic_pointwise_y_all_eta.sum(axis=-1).reshape(grid_shape)
  elpd_waic_y_all_eta = elpd_waic_pointwise_y_all_eta.sum(
      axis=-1).reshape(grid_shape)

  lpd_z_all_eta = lpd_pointwise_z_all_eta.sum(axis=-1).reshape(grid_shape)
  p_waic_z_all_eta = p_waic_pointwise_z_all_eta.sum(axis=-1).reshape(grid_shape)
  elpd_waic_z_all_eta = elpd_waic_pointwise_z_all_eta.sum(
      axis=-1).reshape(grid_shape)

  elpd_surface_dict = {'lpd_all_eta':lpd_all_eta,'p_waic_all_eta':p_waic_all_eta,
  'elpd_waic_all_eta':elpd_waic_all_eta, 'lpd_y_all_eta':lpd_y_all_eta,
  'p_waic_y_all_eta':p_waic_y_all_eta, 'elpd_waic_y_all_eta':elpd_waic_y_all_eta,
  'lpd_z_all_eta':lpd_z_all_eta, 'p_waic_z_all_eta':p_waic_z_all_eta,
  'elpd_waic_z_all_eta':elpd_waic_z_all_eta}

  return elpd_surface_dict


def log_images(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    config: ConfigDict,
    show_elpd: bool,
    eta_grid_len: int,
    summary_writer: Optional[SummaryWriter],
    workdir_png: Optional[str],
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  eta_plot = jnp.array(config.eta_plot)

  assert eta_plot.ndim == 2

  # Plot posterior samples
  key_flow = next(prng_seq)
  for i in range(eta_plot.shape[0]):
    # Sample from flow
    q_distr_out = sample_all_flows(
        params_tuple=[state.params for state in state_list],
        prng_key=key_flow,
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        smi_eta={
            'modules':
                jnp.broadcast_to(eta_plot[[i], :], (config.num_samples_plot,) +
                                 eta_plot.shape[1:])
        },
)

    plot.posterior_samples(
        posterior_sample_dict=q_distr_out['posterior_sample'],
        step=state_list[0].step,
        summary_writer=summary_writer,
        eta=eta_plot[i][1],
        workdir_png=workdir_png,
    )

  ### ELPD ###

  # Define elements to grate grid of eta values

  eta_base = np.array([1., 0.])
  eta_grid_base = np.tile(eta_base, [eta_grid_len + 1, 1]).T


  if show_elpd:

    images = []
    prng_key_elpd = next(prng_seq)

    ########### vary eta_1 only
    eta_grid_mini =  np.linspace(0., 1., eta_grid_len + 1)
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [1]
    eta_grid[eta_grid_x_y_idx, :] = eta_grid_mini

    elpd_dict = elpd_points(
        state_list=state_list,
        batch=batch, # train_ds
        prng_key=prng_key_elpd,
        config=config,
        eta_grid=eta_grid,
    )


    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(4 * 3, 4*3))
    fig.suptitle('Eta_1')

    for mod_ix, mod_name in enumerate(['', '_y', '_z']):
        for i, metric in enumerate([elpd_dict[f'lpd{mod_name}_all_eta'],
                                    -elpd_dict[f'p_waic{mod_name}_all_eta'],
                                    elpd_dict[f'elpd_waic{mod_name}_all_eta'],]):
            axs[mod_ix,i].set_title(['Full likelihood', 'Y module', 'Z module'][mod_ix])
            axs[mod_ix,i].plot(eta_grid[eta_grid_x_y_idx[0]],
                -metric)
            axs[mod_ix,i].set_xlabel('eta_1')
            axs[mod_ix,i].set_ylabel(['- LPD', 'p_WAIC', '- ELPD WAIC'][i])

    plt.tight_layout()

    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / ('elpd_eta1' + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    if summary_writer:
      plot_name = 'rnd_eff_elpd_surface'
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=state_list[0].step,
      )








#########################################################################################################

# def log_images(
#     state_list: List[TrainState],
#     prng_key: PRNGKey,
#     config: ConfigDict,
#     num_samples_plot: int,
#     summary_writer: Optional[SummaryWriter] = None,
#     workdir_png: Optional[str] = None,
# ) -> None:
#   """Plots to monitor during training."""

#   prng_seq = hk.PRNGSequence(prng_key)

#   eta_plot = jnp.array(config.eta_plot)

#   assert eta_plot.ndim == 2

#   # Plot posterior samples
#   key_flow = next(prng_seq)
#   for i in range(eta_plot.shape[0]):
#     # Sample from flow
#     q_distr_out = sample_all_flows(
#         params_tuple=[state.params for state in state_list],
#         prng_key=key_flow,
#         flow_name=config.flow_name,
#         flow_kwargs=config.flow_kwargs,
#         smi_eta={
#             'modules':
#                 jnp.broadcast_to(eta_plot[[i], :],
#                                  (num_samples_plot,) + eta_plot.shape[1:])
#         },
#     )

#     plot.posterior_samples(
#         posterior_sample_dict=q_distr_out['posterior_sample'],
#         summary_writer=summary_writer,
#         step=state_list[0].step,
#         eta=eta_plot[i][1],
#         workdir_png=workdir_png,
#     )


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
  # Small data, no need to batch
  train_ds = load_dataset()

  phi_dim = train_ds['Z'].shape[0]
  theta_dim = 2

  # phi_dim and theta_dim are also arguments of the flow,
  # as they define its dimension
  config.flow_kwargs.phi_dim = phi_dim
  config.flow_kwargs.theta_dim = theta_dim
  # Also is_smi modifies the dimension of the flow, due to the duplicated params
  config.flow_kwargs.is_smi = True

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))
  else:
    summary_writer = None

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state_list = []
  state_name_list = []

  state_name_list.append('phi')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(q_distr_phi),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'eta': jnp.ones((config.num_samples_elbo, 2)),
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # Get an initial sample of phi
  # (used below to initialize theta)
  phi_base_sample_init = hk.transform(q_distr_phi).apply(
      state_list[0].params,
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      eta=jnp.ones((config.num_samples_elbo, 2)),
  )['phi_base_sample']

  state_name_list.append('theta')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(q_distr_theta),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'phi_base_sample': phi_base_sample_init,
              'eta': jnp.ones((config.num_samples_elbo, 2)),
              'is_aux': False,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  if config.flow_kwargs.is_smi:
    state_name_list.append('theta_aux')
    state_list.append(
        initial_state_ckpt(
            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
            forward_fn=hk.transform(q_distr_theta),
            forward_fn_kwargs={
                'flow_name': config.flow_name,
                'flow_kwargs': config.flow_kwargs,
                'phi_base_sample': phi_base_sample_init,
                'eta': jnp.ones((config.num_samples_elbo, 2)),
                'is_aux': True,
            },
            prng_key=next(prng_seq),
            optimizer=make_optimizer(**config.optim_kwargs),
        ))

  # Print a useful summary of the execution of the flow architecture.
  logging.info('FLOW PHI:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: hk.transform(q_distr_phi).apply(
          params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          eta=jnp.ones((config.num_samples_elbo, 2)),
      ),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_list[0].params, next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)

  logging.info('FLOW THETA:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: hk.transform(q_distr_theta).apply(
          params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          phi_base_sample=phi_base_sample_init,
          eta=jnp.ones((config.num_samples_elbo, 2)),
          is_aux=False,
      ),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_list[1].params, next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)

  # Jit function to update training states
  update_states_jit = lambda state_list, batch, prng_key: update_states(
      state_list=state_list,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer(**config.optim_kwargs),
      loss_fn=loss,
      loss_fn_kwargs={
          'num_samples': config.num_samples_elbo,
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'eta_sampling_a': config.eta_sampling_a,
          'eta_sampling_b': config.eta_sampling_b,
      },
  )
  update_states_jit = jax.jit(update_states_jit)

  elbo_validation_jit = lambda state_list, batch, prng_key: elbo_estimate_along_eta(
      params_tuple=tuple(state.params for state in state_list),
      batch=batch,
      prng_key=prng_key,
      num_samples=config.num_samples_eval,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      eta_sampling_a=1.,
      eta_sampling_b=1.,
  )
  elbo_validation_jit = jax.jit(elbo_validation_jit)

  if state_list[0].step < config.training_steps:
    logging.info('Training Variational Meta-Posterior (VMP-flow)...')

  # Reset random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  while state_list[0].step < config.training_steps:

    # Plots to monitor training
    if ((state_list[0].step == 0) or
        (state_list[0].step % config.log_img_steps == 0)):
      # print("Logging images...\n")
      log_images(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
          config=config,
          show_elpd=False,
          eta_grid_len=15,
          summary_writer=summary_writer,
          workdir_png=workdir,
      )

    # Log learning rate
    summary_writer.scalar(
        tag='learning_rate',
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(state_list[0].step),
        step=state_list[0].step,
    )

    # SGD step
    state_list, metrics = update_states_jit(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
    )
    # The computed training loss corresponds to the model before update
    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state_list[0].step - 1,
    )

    if state_list[0].step == 1:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    # Metrics for evaluation
    if state_list[0].step % config.eval_steps == 0:

      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

      elbo_validation_dict = elbo_validation_jit(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
      )
      for k, v in elbo_validation_dict.items():
        summary_writer.scalar(
            tag=f'elbo_{k}',
            value=v.mean(),
            step=state_list[0].step,
        )

    if state_list[0].step % config.checkpoint_steps == 0:
      for state_i, state_name_i in zip(state_list, state_name_list):
        save_checkpoint(
            state=state_i,
            checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
            keep=config.checkpoints_keep,
        )

    # Wait until computations are done before the next step
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info('Final training step: %i', state_list[0].step)

  # Saving checkpoint at the end of the training process
  # (in case training_steps is not multiple of checkpoint_steps)
  for state_i, state_name_i in zip(state_list, state_name_list):
    save_checkpoint(
        state=state_i,
        checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
        keep=config.checkpoints_keep,
    )

  # Last plot of posteriors
  log_images(
      state_list=state_list,
      batch=train_ds,
      prng_key=next(prng_seq),
      config=config,
      show_elpd=True,
      eta_grid_len=15,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )

  return state_list
