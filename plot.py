"""Plot methods for the LALME model."""

import pathlib

import numpy as np
import scipy

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns

import jax
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp

import log_prob_fun

from modularbayes import log1mexpm, plot_to_image, normalize_images
from modularbayes._src.typing import (Any, Array, Batch, Dict, Kernel, List,
                                      Mapping, Optional, PRNGKey, SummaryWriter,
                                      Tuple)

kernels = tfp.math.psd_kernels

Axes = matplotlib.axes.Axes
Figure = matplotlib.figure.Figure
JointGrid = sns.JointGrid


def plot_linguistic_field(
    posterior_sample_dict: Dict[str, Any],
    data: Dict[str, Any],
    items_id: List[str],
    forms_id: List[str],
    item: int,
    show_loc_anchor: bool = True,
    alpha_loc_anchor: float = 0.3,
    show_loc_floating_bayes: bool = False,
    alpha_loc_floating_bayes: Optional[float] = 0.01,
    show_loc_floating_linguist: bool = True,
    use_gamma_anchor: bool = False,
) -> Tuple[Figure, Axes]:

  # Total number of forms for this item
  num_forms = data['num_forms_tuple'][item]

  if use_gamma_anchor:
    # Samples of GPs on inducing points and anchor profiles
    # Note: gamma_anchor has an additional axis at dimension 1,
    #   num_samples_gamma_profiles, we only take the first sample
    gamma = np.concatenate([
        posterior_sample_dict['gamma_inducing'],
        posterior_sample_dict['gamma_anchor'][:, 0, ...],
    ],
                           axis=-1)
    # Locations of inducing points and anchor profiles
    loc = np.concatenate([
        data['loc_inducing'],
        data['loc'][:data['num_profiles_anchor']],
    ],
                         axis=0)
  else:
    gamma = posterior_sample_dict['gamma_inducing']
    loc = data['loc_inducing']

  # Get Probability fields
  # Linear transformation of GPs
  phi_prob_ = np.einsum(
      "sbf,sbp->sfp",
      posterior_sample_dict['mixing_weights_list'][item],
      gamma,
  ) + np.expand_dims(
      posterior_sample_dict['mixing_offset_list'][item], axis=-1)
  # softmax transform over the form axis.
  phi_prob = scipy.special.softmax(phi_prob_, axis=1)
  # Average over samples
  phi_prob_mean = phi_prob.mean(axis=0)

  # Create plotting grid
  axs_nrows = int(np.ceil(np.sqrt(num_forms)))
  axs_ncols = int(np.ceil(np.sqrt(num_forms)))
  # fig, ax = plt.subplots()
  fig, axs = plt.subplots(
      nrows=axs_nrows,
      ncols=axs_ncols,
      figsize=(3. * axs_ncols, 2.5 * axs_nrows),
  )
  fig.suptitle(f"item:{str(items_id[item])}")

  # Plot each form on a separate panel
  for f_, ax in enumerate(axs.reshape(-1)[:num_forms]):
    # Plot the mean of linguistic field
    x_plot, y_plot = np.split(loc, 2, axis=-1)
    z_plot = np.round(phi_prob_mean[f_], 2)
    cntr2 = ax.tricontour(
        x_plot.squeeze(), y_plot.squeeze(), z_plot, cmap="Greys")
    if len(cntr2.levels) > 1:
      fig.colorbar(cntr2, ax=ax)
    else:
      ax.text(
          x_plot.max(),
          y_plot.max(),
          cntr2.levels[0],
          horizontalalignment='right',
          verticalalignment='top',
          transform=ax.transAxes)

    if show_loc_floating_bayes:
      # Plot floating profiles location
      # Infered by the model
      for p in range(data['num_profiles_floating']):
        f_in_profile = (
            data['y'][item][f_][data['num_profiles_anchor'] + p] == 1)
        if f_in_profile:
          ax.scatter(
              posterior_sample_dict['loc_floating'][:, p, 0],
              posterior_sample_dict['loc_floating'][:, p, 1],
              c='darkgreen',
              marker=f"${p}$",
              alpha=alpha_loc_floating_bayes,
              label='floating (bayes)')

    if show_loc_floating_linguist:
      # Plot floating profiles location
      # Assigned by linguists
      floating_with_f = np.where((data['y'][item][f_]).astype(bool) & (
          np.arange(data['num_profiles']) >= data['num_profiles_anchor']))[0]
      if len(floating_with_f) > 0:
        ax.scatter(
            data['loc'][floating_with_f, 0],
            data['loc'][floating_with_f, 1],
            c='blue',
            marker='x',
            # alpha=alpha_anchor,
            label='anchor')

    # Plot anchor profiles location
    if show_loc_anchor:
      anchors_with_f = np.where((data['y'][item][f_]).astype(bool) & (
          np.arange(data['num_profiles']) < data['num_profiles_anchor']))[0]
      if len(anchors_with_f) > 0:
        ax.scatter(
            data['loc'][anchors_with_f, 0],
            data['loc'][anchors_with_f, 1],
            c='red',
            marker='o',
            alpha=alpha_loc_anchor,
            label='anchor')

    # Set panel title
    ax.set_title(f"form:{str(forms_id[item][f_])}")

  plt.tight_layout()
  return fig, axs


def plot_basis_fields(
    posterior_sample_dict: Dict[str, Any],
    data: Dict[str, Any],
    use_gamma_anchor: bool = False,
) -> Tuple[Figure, Axes]:

  # Total number of base GPs
  num_basis_gps = posterior_sample_dict['gamma_inducing'].shape[1]

  if use_gamma_anchor:
    # Samples of GPs on inducing points and anchor profiles
    # Note: gamma_anchor has an additional axis at dimension 1,
    #   num_samples_gamma_profiles, we only take the first sample
    gamma = np.concatenate([
        posterior_sample_dict['gamma_inducing'],
        posterior_sample_dict['gamma_anchor'][:, 0, ...],
    ],
                           axis=-1)
    # Locations of inducing points and anchor profiles
    loc = np.concatenate([
        data['loc_inducing'],
        data['loc'][:data['num_profiles_anchor']],
    ],
                         axis=0)
  else:
    gamma = posterior_sample_dict['gamma_inducing']
    loc = data['loc_inducing']

  # Average over samples
  gamma_mean = gamma.mean(axis=0)

  # Create plotting grid
  # axs_nrows = int(np.ceil(np.sqrt(num_basis_gps)))
  # axs_ncols = int(np.ceil(np.sqrt(num_basis_gps)))
  axs_nrows = 1
  axs_ncols = num_basis_gps
  # fig, ax = plt.subplots()
  fig, axs = plt.subplots(
      nrows=axs_nrows,
      ncols=axs_ncols,
      figsize=(3. * axs_ncols, 2.5 * axs_nrows),
  )
  fig.suptitle("Basis Fields")

  # For each base GP
  for f, ax in enumerate(axs.reshape(-1)[:num_basis_gps]):
    # Plot the mean of the basis field
    x_plot, y_plot = np.split(loc, 2, axis=-1)
    z_plot = np.round(gamma_mean[f], 2)
    cntr2 = ax.tricontour(
        x_plot.squeeze(), y_plot.squeeze(), z_plot, cmap="Greys")
    if len(cntr2.levels) > 1:
      fig.colorbar(cntr2, ax=ax)
    else:
      ax.text(
          x_plot.max(),
          y_plot.max(),
          cntr2.levels[0],
          horizontalalignment='right',
          verticalalignment='top',
          transform=ax.transAxes)

  plt.tight_layout()
  return fig, axs


def plot_profile_location(
    posterior_sample_dict: Dict[str, Any],
    data: Dict[str, Any],
    profiles_id: List[str],
    profile: int,
    loc_type: str = 'floating',
    posterior_sample_dict_2: Optional[Dict[str, Any]] = None,
    kde_x_range: Optional[Tuple[float]] = (0., 1.),
    kde_y_range: Optional[Tuple[float]] = (0., 1.),
):

  assert loc_type in ['floating', 'random_anchor']

  fig, ax = plt.subplots(figsize=(5, 5))

  # Samples from posterior Profile locations
  assert ('loc_' + loc_type) in posterior_sample_dict
  loc_x, loc_y = [
      x.flatten() for x in np.split(
          posterior_sample_dict['loc_' + loc_type][:, profile, :],
          2,
          -1,
      )
  ]
  # sns.kdeplot(
  #     x=loc_x,
  #     y=loc_y,
  #     shade=True,
  #     cmap="PuBu",
  #     ax=ax,
  # )
  _ = ax.hist2d(
      x=loc_x,
      y=loc_y,
      bins=20,
      cmap="Blues",
      range=[[0., 1.], [0., 1.]],
  )

  # Additional Samples from posterior Profile locations
  if posterior_sample_dict_2 is not None:
    assert ('loc_' + loc_type) in posterior_sample_dict_2
    loc_x, loc_y = [
        x.flatten() for x in np.split(
            posterior_sample_dict_2['loc_' + loc_type][:, profile, :],
            2,
            -1,
        )
    ]
    # Peform the kernel density estimate
    xx, yy = np.mgrid[kde_x_range[0]:kde_x_range[1]:100j,
                      kde_y_range[0]:kde_y_range[1]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([loc_x, loc_y])
    kernel = scipy.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    cset = ax.contour(xx, yy, f, cmap='copper')
    ax.clabel(cset, inline=1)

  data_loc_idx = (data['num_profiles_anchor']
                  if loc_type == 'floating' else 0) + profile

  # Add Location of Profile from LALME data
  ax.scatter(
      x=data['loc'][[data_loc_idx], 0],
      y=data['loc'][[data_loc_idx], 1],
      marker="X",
      s=200,
      color="red",
  )
  ax.set_xlim((0, 1))
  ax.set_ylim((0, 1))
  ax.set_title(loc_type + ' profile ' + str(profiles_id[data_loc_idx]))

  return fig, ax


def plot_mixing_weights(
    posterior_sample_dict: Dict[str, Any],
    item: int,
) -> Tuple[Figure, Axes]:
  mixing_weights = posterior_sample_dict['mixing_weights_list'][item]
  mixing_weights_mean = mixing_weights.mean(axis=0)
  _, num_basis_gps, num_forms = mixing_weights.shape
  fig, axs = plt.subplots(
      num_basis_gps,
      num_forms,
      figsize=(2. * num_forms, 1.5 * num_basis_gps),
  )
  for b in range(num_basis_gps):
    for f in range(num_forms):
      violin_dict = axs[b, f].violinplot(dataset=mixing_weights[:, b, f])
      violin_dict['bodies'][0].set_edgecolor('black')
      violin_dict['bodies'][0].set_facecolor(
          colors.to_hex(
              plt.get_cmap('coolwarm')(
                  scipy.special.expit(-mixing_weights_mean[b, f]))))
      axs[b, f].set_xticks([])

  return fig, axs


def plot_profile_location_given_y(
    y: List[Array],
    posterior_sample_dict: Dict[str, Any],
    loc_inducing: Array,
    kernel: Kernel,
    prng_key: PRNGKey,
    num_loc_plot: int = 1000,
    loc_candidate: Optional[Array] = None,
    gp_jitter: float = 1e-2,
) -> Tuple[Figure, Axes]:

  # p=0
  # y = [y_i[:,[p]] for y_i in batch['y']]
  # loc_candidate = batch['loc'][[p],:]
  # loc_inducing = batch['loc_inducing']
  # kernel = get_kernel()
  # num_loc_plot = 1000
  # eps = 1e-2

  num_profiles_plot = y[0].shape[1]
  num_inducing_points = loc_inducing.shape[0]

  assert all([y_i.shape[1] == num_profiles_plot for y_i in y])

  # Only one loc_candidate per profile is allowed for now
  if loc_candidate is not None:
    assert loc_candidate.shape == (num_profiles_plot, 2)

  prng_key, key_loc = jax.random.split(prng_key)
  loc_plot = jax.random.uniform(key=key_loc, shape=(num_loc_plot, 2))

  cov_loc_plot = kernel.matrix(
      x1=loc_plot,
      x2=loc_plot,
  ) + gp_jitter * jnp.eye(num_loc_plot)
  cov_loc_plot_inducing = kernel.matrix(x1=loc_plot, x2=loc_inducing)

  cov_inducing = kernel.matrix(
      x1=loc_inducing,
      x2=loc_inducing,
  ) + gp_jitter * jnp.eye(num_inducing_points)
  # Inverse of covariance of inducing values
  cov_inducing_chol = jnp.linalg.cholesky(cov_inducing)
  cov_inducing_chol_inv = jax.scipy.linalg.solve_triangular(
      a=cov_inducing_chol,
      b=jnp.eye(num_inducing_points),
      lower=True,
  )
  cov_inducing_inv = jnp.matmul(
      cov_inducing_chol_inv.T, cov_inducing_chol_inv, precision='highest')

  p_loc_plot_given_inducing = log_prob_fun.gp_F_given_U(
      u=posterior_sample_dict['gamma_inducing'],
      cov_x=jnp.expand_dims(cov_loc_plot, axis=0),
      cov_x_z=jnp.expand_dims(cov_loc_plot_inducing, axis=0),
      cov_z_inv=cov_inducing_inv,
      gp_jitter=gp_jitter,
  )
  prng_key, key_gp_loc_plot = jax.random.split(prng_key)
  # Sample base GPs on the loc_plot locations
  phi_loc_plot = p_loc_plot_given_inducing.sample(seed=key_gp_loc_plot)

  # Compute log_prob(X, Y| parameters) for each profile in y
  log_prob_y_all_profiles = []
  for p in range(num_profiles_plot):
    # Select one of the profiles and replicate it num_loc_plot times
    y_p = [
        jnp.broadcast_to(y_i[:, [p]], (y_i.shape[0], num_loc_plot)) for y_i in y
    ]
    # Compute log_prob(Y| parameters)
    log_prob_y_equal_1_pointwise_list = jax.vmap(
        log_prob_fun.log_prob_y_equal_1)(
            gamma=phi_loc_plot,
            mixing_weights_list=posterior_sample_dict['mixing_weights_list'],
            mixing_offset_list=posterior_sample_dict['mixing_offset_list'],
            mu=posterior_sample_dict['mu'],
            zeta=posterior_sample_dict['zeta'],
        )
    log_prob_y_item_profile = jnp.stack([
        jnp.where(y_p_i, log_prob_y_eq_1_i,
                  log1mexpm(-log_prob_y_eq_1_i)).sum(axis=1) for y_p_i,
        log_prob_y_eq_1_i in zip(y_p, log_prob_y_equal_1_pointwise_list)
    ],
                                        axis=1)
    # Sum over item dimension
    log_prob_y_p_samples = log_prob_y_item_profile.sum(axis=1)
    # Average over posterior samples
    log_prob_y_p = log_prob_y_p_samples.mean(axis=0)
    # Normalize over locations, so they add up to 1
    log_prob_y_p_normalized = log_prob_y_p - jax.scipy.special.logsumexp(
        log_prob_y_p)
    # assert (jnp.exp(log_prob_y_p_normalized).sum() - 1.) < 1e-4

    # Append to the list with all profiles
    log_prob_y_all_profiles.append(log_prob_y_p_normalized)

  # Create plotting grid
  axs_nrows = int(np.ceil(np.sqrt(num_profiles_plot)))
  axs_ncols = int(np.ceil(np.sqrt(num_profiles_plot)))

  # fig, ax = plt.subplots()
  fig, axs = plt.subplots(
      nrows=axs_nrows,
      ncols=axs_ncols,
      figsize=(3. * axs_ncols, 2.5 * axs_nrows),
  )
  axs = np.array(axs)
  x_plot, y_plot = np.split(loc_plot, 2, axis=-1)
  # Plot probabilities for each profile
  for p, ax in enumerate(axs.reshape(-1)[:num_profiles_plot]):
    z_plot = np.round(jnp.exp(log_prob_y_all_profiles[p]), 3)
    cntr2 = ax.tricontour(
        x_plot.squeeze(), y_plot.squeeze(), z_plot, cmap="Greys")
    if len(cntr2.levels) > 1:
      fig.colorbar(cntr2, ax=ax)
    else:
      ax.text(
          x_plot.max(),
          y_plot.max(),
          cntr2.levels[0],
          horizontalalignment='right',
          verticalalignment='top',
          transform=ax.transAxes)

    if loc_candidate is not None:
      ax.scatter(
          x=loc_candidate[p, 0],
          y=loc_candidate[p, 1],
          marker=7,
          color='red',
      )
  plt.tight_layout()

  return fig, axs


def posterior_samples(
    posterior_sample_dict: Mapping[str, Any],
    batch: Batch,
    prng_key: PRNGKey,
    kernel_name: str,
    kernel_kwargs: Dict[str, Any],
    gp_jitter: Optional[float],
    step: int,
    profiles_id: List[str],
    items_id: List[str],
    forms_id: List[str],
    show_basis_fields: bool,
    show_linguistic_fields: bool,
    num_loc_random_anchor_plot: Optional[int],
    num_loc_floating_plot: Optional[int],
    show_mixing_weights: bool,
    show_loc_given_y: bool,
    suffix: Optional[str] = None,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
    use_gamma_anchor: bool = False,
) -> None:
  """Visualise samples from the approximate posterior distribution."""

  # Plot basis fields
  if show_basis_fields:
    plot_name = "lalme_basis_fields"
    plot_name = plot_name + ("" if (suffix is None) else ("_" + suffix))
    fig, _ = plot_basis_fields(
        posterior_sample_dict=posterior_sample_dict,
        data=batch,
    )
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      summary_writer.image(plot_name, plot_to_image(fig), step=step)
    # images.append(plot_to_image(fig))

  # Plot linguistic fields
  if show_linguistic_fields:
    images = []
    assert len(batch['num_forms_tuple']) == len(forms_id)
    assert len(forms_id) == len(items_id)
    for i_, item_id in enumerate(items_id):
      plot_name = f"linguistic_field_{str(item_id).replace('/', '-')}"
      if suffix is not None:
        plot_name += ("_" + suffix)
      fig, _ = plot_linguistic_field(
          posterior_sample_dict=posterior_sample_dict,
          data=batch,
          items_id=items_id,
          forms_id=forms_id,
          item=i_,
          show_loc_anchor=True,
          alpha_loc_anchor=0.3,
          show_loc_floating_bayes=False,
          alpha_loc_floating_bayes=0.01,
          show_loc_floating_linguist=False,
          use_gamma_anchor=use_gamma_anchor,
      )
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      # summary_writer.image(plot_name, plot_to_image(fig), step=step)
      images.append(plot_to_image(fig))

    if summary_writer:
      plot_name = "lalme_linguistic_fields"
      if suffix is not None:
        plot_name += ("_" + suffix)
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )
      del images

  # Plot anchor profiles locations
  if num_loc_random_anchor_plot is not None:
    images = []
    profiles_plot = range(
        min(num_loc_random_anchor_plot, batch['num_profiles_anchor']))
    for p in profiles_plot:
      plot_name = f"anchor_profile_{profiles_id[p]:03d}_random_loc"
      if suffix is not None:
        plot_name += ("_" + suffix)
      fig, _ = plot_profile_location(
          posterior_sample_dict=posterior_sample_dict,
          data=batch,
          profiles_id=profiles_id,
          profile=p,
          loc_type='random_anchor',
      )
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      # summary_writer.image(plot_name, plot_to_image(fig), step=step)
      images.append(plot_to_image(fig))

    if summary_writer:
      plot_name = "lalme_anchor_profiles_locations"
      if suffix is not None:
        plot_name += ("_" + suffix)
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )
      del images

  # Plot floating profiles locations
  if num_loc_floating_plot is not None:
    images = []
    profiles_plot = range(
        min(num_loc_floating_plot, batch['num_profiles_floating']))
    for p in profiles_plot:
      plot_name = f"floating_profile_{profiles_id[batch['num_profiles_anchor']+p]:03d}_loc"
      if suffix is not None:
        plot_name += ("_" + suffix)
      fig, _ = plot_profile_location(
          posterior_sample_dict=posterior_sample_dict,
          data=batch,
          profiles_id=profiles_id,
          profile=p,
          loc_type='floating',
      )
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      # summary_writer.image(plot_name, plot_to_image(fig), step=step)
      images.append(plot_to_image(fig))

    if summary_writer:
      plot_name = "lalme_floating_profiles_locations"
      if suffix is not None:
        plot_name += ("_" + suffix)
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )
      del images

  # Plot mixing weights
  if show_mixing_weights:
    images = []
    for i_ in range(len(batch['num_forms_tuple'])):
      plot_name = f"lalme_mixing_weights_{i_}"
      fig, _ = plot_mixing_weights(
          posterior_sample_dict=posterior_sample_dict,
          item=i_,
      )
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      # summary_writer.image(plot_name, plot_to_image(fig), step=step)
      images.append(plot_to_image(fig))

    if summary_writer:
      plot_name = "lalme_mixing_weights"
      if suffix is not None:
        plot_name += ("_" + suffix)
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )
      del images

  # Plot log_prob(X|Y)
  if show_loc_given_y:
    plot_name = "lalme_loc_given_y"
    prng_key, key_loc_given_y = jax.random.split(prng_key)
    fig, _ = plot_profile_location_given_y(
        y=batch['y'],
        posterior_sample_dict=posterior_sample_dict,
        loc_inducing=batch['loc_inducing'],
        kernel=getattr(kernels, kernel_name)(**kernel_kwargs),
        prng_key=key_loc_given_y,
        num_loc_plot=1000,
        loc_candidate=batch['loc'],
        gp_jitter=gp_jitter,
    )
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      summary_writer.image(plot_name, plot_to_image(fig), step=step)


def posterior_samples_compare(
    batch: Batch,
    # prng_key: PRNGKey,
    posterior_sample_dict_1: Mapping[str, Any],
    posterior_sample_dict_2: Mapping[str, Any],
    step: int,
    profiles_id: List[str],
    num_loc_floating_plot: Optional[int],
    suffix: Optional[str] = None,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
):
  """Plot comparison two sets of posterior samples.

  This method is mainly intended to compare MCMC vs Variational.
  The first dictionary of samples is plotted as a heatmap, the second dictionary
  overplaced as level curves.
  """

  # Plot floating profiles locations
  if num_loc_floating_plot is not None:
    profiles_plot = range(
        min(num_loc_floating_plot, batch['num_profiles_floating']))

    images = []
    for p in profiles_plot:
      plot_name = f"floating_profile_{profiles_id[batch['num_profiles_anchor']+p]:03d}_loc_2_samples"
      if suffix is not None:
        plot_name += ("_" + suffix)
      fig, _ = plot_profile_location(
          posterior_sample_dict=posterior_sample_dict_1,
          data=batch,
          profiles_id=profiles_id,
          profile=p,
          loc_type='floating',
          posterior_sample_dict_2=posterior_sample_dict_2,
          kde_x_range=(0., 1.),
          kde_y_range=(0., 0.8939394),
      )
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      images.append(plot_to_image(fig))
    if summary_writer:
      plot_name = "floating_profiles_locations_2_samples"
      if suffix is not None:
        plot_name += ("_" + suffix)
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )
      del images
