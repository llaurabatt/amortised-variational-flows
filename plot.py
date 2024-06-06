"""Plotting methods for the LALME model."""

import pathlib

import numpy as np
import scipy

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors
import seaborn as sns

import arviz as az
from arviz import InferenceData

from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp
import wandb

from log_prob_fun import (ModelParamsGlobal, ModelParamsLocations,
                          ModelParamsGammaProfiles)
# from train_flow_allhp import (error_locations_estimate)

from modularbayes import plot_to_image, normalize_images
from modularbayes._src.typing import (Any, Array, Dict, List, Mapping, Optional,
                                      SummaryWriter, Tuple)

from misc import clean_filename

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


def plot_basis_fields_az(
    lalme_az: InferenceData,
    lalme_dataset: Dict[str, Any],
    loc_inducing: Array,
    use_gamma_anchor: bool = False,
) -> Tuple[Figure, Axes]:

  # Total number of base GPs
  num_basis_gps = lalme_az.posterior['gamma_inducing'].shape[-2]

  if use_gamma_anchor:
    # Samples of GPs on inducing points and anchor profiles
    # Note: gamma_anchor has an additional axis at dimension 1,
    #   num_samples_gamma_profiles, we only take the first sample
    gamma = np.concatenate([
        lalme_az.posterior['gamma_inducing'].values,
        lalme_az.posterior['gamma_anchor'].values,
    ],
                           axis=-1)
    # Locations of inducing points and anchor profiles
    loc = np.concatenate([
        loc_inducing,
        lalme_dataset['loc'][:lalme_dataset['num_profiles_anchor']],
    ],
                         axis=0)
  else:
    gamma = lalme_az.posterior['gamma_inducing'].values
    loc = loc_inducing

  # Average over samples
  gamma_mean = jnp.mean(gamma, axis=[0, 1])

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


def profile_locations_grid(
    lalme_dataset: Dict[str, Any],
    profiles_id: List[int],
    var_name: str = 'loc_floating',
    coord: str = "LP_floating",
    suptitle: Optional[str] = None,
    nrows: Optional[int] = None,
    scatter_kwargs: Optional[Dict[str, Any]] = None,
    wass_dists: Optional[dict] = None,
    lalme_az: Optional[InferenceData] = None,
    lalme_az_2: Optional[InferenceData] = None,
    lalme_az_list: Optional[List] = None,
    prior_hparams_str_list: Optional[List] = None,
    MSEs_dict: Optional[Dict] = None,

):

  assert((((lalme_az is not None) or (lalme_az_2 is not None)) and (lalme_az_list is None))
    or
    (((lalme_az is None) and (lalme_az_2 is None)) and (lalme_az_list is not None))
  )

  assert ((((lalme_az_list is not None) and (prior_hparams_str_list is not None)) or ((lalme_az_list is None) and (prior_hparams_str_list is None))))
  # Subplots layout
  if nrows is None:
    nrows = int(np.sqrt(len(profiles_id)))
  ncols = len(profiles_id) // nrows
  ncols += 1 if len(profiles_id) % nrows else 0

  fig, axs = plt.subplots(
      nrows,
      ncols,
      figsize=(ncols * 3 + 1, nrows * 3),
      squeeze=False,
      sharex=True,
      sharey=True,
  )
  for i, lp_ in enumerate(profiles_id):
    
    p_ = np.where(lalme_dataset['LP'] == lp_)[0][0]
    if lalme_az is not None:
      az.plot_pair(
          lalme_az,
          var_names=[var_name],
          coords={coord: lp_},
          kind='scatter',
          scatter_kwargs=scatter_kwargs,
          ax=axs[i // ncols, i % ncols],
      )
    if lalme_az_2 is not None:
      az.plot_pair(
          lalme_az_2,
          var_names=[var_name],
          coords={coord: lp_},
          kind=["kde"],
          kde_kwargs={
              "fill_last": False,
              "hdi_probs": [0.05, 0.5, 0.95]
          },
          ax=axs[i // ncols, i % ncols],
      )
    if lalme_az_list is not None:
      colors = ['blue', 'orange', 'green', 'black', 'purple']      
      for j_ix, lalme_az_j in enumerate(lalme_az_list):
          az.plot_pair(
          lalme_az_j,
          var_names=[var_name],
          coords={coord: lp_},
          kind='scatter',
          # color=colors[j_ix],
          scatter_kwargs=scatter_kwargs,
          ax=axs[i // ncols, i % ncols],
          )

    axs[i // ncols, i % ncols].scatter(
        x=lalme_dataset['loc'][[p_], 0],
        y=lalme_dataset['loc'][[p_], 1],
        marker="X",
        s=200,
        color="red",
    )
    axs[i // ncols, i % ncols].set_xlim([0, 1])
    axs[i // ncols, i % ncols].set_ylim([0, 1])
    axs[i // ncols, i % ncols].set_xlabel("")
    axs[i // ncols, i % ncols].set_ylabel("")
    axs[i // ncols, i % ncols].set_title(f"Profile: {lp_}" + \
                                         (f", MSE {float(MSEs_dict[lp_]):.2f}" if MSEs_dict is not None else "") + \
                                         (f", WD {float(wass_dists[lp_]):.2f}" if wass_dists is not None else ""))
    if lalme_az_list is not None:
      legend_patches = [Patch(facecolor=colors[j], label=prior_hparams_str_list[j]) for j in jnp.arange(len(prior_hparams_str_list))]
      plt.figlegend(handles=legend_patches, loc='lower center', ncols=3)

  if suptitle:
    fig.suptitle(suptitle)
  fig.tight_layout()
  if lalme_az_list is not None:
    fig.subplots_adjust(left=None, bottom=0.15, right=None, top=0.92, wspace=0.3, hspace=0.4)

  return fig, axs

def profile_locations_img_level_curves(
    lalme_dataset: Dict[str, Any],
    profiles_id: List[int],
    img:str,
    var_name: str = 'loc_floating',
    coord: str = "LP_floating",
    suptitle: Optional[str] = None,
    nrows: Optional[int] = None,
    lalme_az: Optional[InferenceData] = None,
    lalme_az_2: Optional[InferenceData] = None,
):

  assert(((lalme_az is not None) or (lalme_az_2 is not None)))

  # Subplots layout
  if nrows is None:
    nrows = int(np.sqrt(len(profiles_id)))
  ncols = len(profiles_id) // nrows
  ncols += 1 if len(profiles_id) % nrows else 0


  fig = plt.figure(figsize=(ncols * 3 + 1, nrows * 3))

  fig_ax = fig.add_axes([0, 0, 1, 1], zorder=1)
  im = plt.imread(img)
  fig_ax.imshow(im, extent=[0, ncols, 0, nrows], aspect='auto')
  fig_ax.axis('off')


  for i, lp_ in enumerate(profiles_id):
      ax = fig.add_subplot(nrows, ncols, i+1, zorder=2, facecolor="none")
      p_ = np.where(lalme_dataset['LP'] == lp_)[0][0]
      if lalme_az is not None:
        az.plot_pair(
            lalme_az,
            var_names=[var_name],
            coords={coord: lp_},
            kind=["kde"],
            kde_kwargs={
                "fill_last": False,
                "hdi_probs": [0.05, 0.5, 0.95]
            },
            ax=ax,
        )
      if lalme_az_2 is not None:
        az.plot_pair(
            lalme_az_2,
            var_names=[var_name],
            coords={coord: lp_},
            kind=["kde"],
            kde_kwargs={
                "fill_last": False,
                "hdi_probs": [0.05, 0.5, 0.95]
            },
            ax=ax,
        )

      # Set limits, labels, title
      ax.set_xlim([0, 1])
      ax.set_ylim([0, 1])
      ax.set_xlabel("")
      ax.set_ylabel("")
      ax.set_xticks([])
      ax.set_yticks([])
      ax.spines["top"].set_visible(False)
      ax.spines["right"].set_visible(False)
      ax.spines["bottom"].set_visible(False)
      ax.spines["left"].set_visible(False)
  fig.subplots_adjust(left=0.03,right=0.98,bottom=0.07,top=0.935)

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


def lalme_plots_arviz(
    lalme_az: InferenceData,
    lalme_dataset: Dict[str, Any],
    step: Optional[int] = 0,
    show_mu: bool = False,
    show_zeta: bool = False,
    show_basis_fields: bool = False,
    show_W_items: Optional[List[str]] = None,
    show_a_items: Optional[List[str]] = None,
    mcmc_img: Optional[str] = None,
    lp_floating: Optional[List[int]] = None,
    lp_floating_aux_traces: Optional[List[int]] = None,
    lp_floating_traces: Optional[List[int]] = None,
    lp_floating_aux_grid10: Optional[List[int]] = None,
    lp_floating_grid10: Optional[List[int]] = None,
    lp_anchor_val_grid30: Optional[List[int]] = None,
    lp_anchor_val_grid28: Optional[List[int]] = None,
    lp_anchor_val_grid21: Optional[List[int]] = None,
    lp_anchor_val_grid10: Optional[List[int]] = None,
    lp_random_anchor: Optional[List[int]] = None,
    lp_random_anchor_grid10: Optional[List[int]] = None,
    lp_anchor_val: Optional[List[int]] = None,
    lp_anchor_test: Optional[List[int]] = None,
    loc_inducing: Optional[Array] = None,
    workdir_png: Optional[str] = None,
    summary_writer: Optional[SummaryWriter] = None,
    use_wandb: Optional[bool]=False,
    suffix: str = '',
    scatter_kwargs={'alpha': 0.07},
    MSEs_anchor_val_dict: Optional[Dict] = None,
):
  

  if show_mu:
    axs = az.plot_trace(
        lalme_az,
        var_names=["mu"],
        compact=False,
    )
    max_ = float(lalme_az.posterior.mu.max())
    for axs_i in axs:
      axs_i[0].set_xlim([0, max_])
    plt.tight_layout()
    if workdir_png:
      plot_name = "lalme_mu_trace"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = "lalme_mu_trace"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if show_zeta:
    axs = az.plot_trace(
        lalme_az,
        var_names=["zeta"],
        compact=False,
    )
    for axs_i in axs:
      axs_i[0].set_xlim([0, 1])
    plt.tight_layout()
    if workdir_png:
      plot_name = "lalme_zeta_trace"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = "lalme_zeta_trace"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if show_basis_fields:
    assert loc_inducing is not None, "loc_inducing must be provided to plot basis fields"
    fig, _ = plot_basis_fields_az(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        loc_inducing=loc_inducing,
    )
    plot_name = "lalme_basis_fields"
    plot_name += suffix
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      summary_writer.image(plot_name, plot_to_image(fig), step=step)

  if show_W_items is not None:
    idx_ = np.intersect1d(
        show_W_items, lalme_dataset['items'], return_indices=True)[2]
    images = []
    for i in idx_:
      axs = az.plot_forest(lalme_az, var_names=[f"W_{i}"], ess=True)
      plt.suptitle(f"LMC weights {lalme_dataset['items'][i]}")
      plt.tight_layout()
      if workdir_png:
        plot_name = f"lalme_W_{i}"
        plot_name += suffix
        plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      images.append(plot_to_image(None))

    if summary_writer:
      plot_name = "lalme_W"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )

  if show_a_items is not None:
    idx_ = np.intersect1d(
        show_a_items, lalme_dataset['items'], return_indices=True)[2]
    images = []
    for i in idx_:
      axs = az.plot_forest(lalme_az, var_names=[f"a_{i}"], ess=True)
      plt.suptitle(f"LMC offsets {lalme_dataset['items'][i]}")
      plt.tight_layout()
      if workdir_png:
        plot_name = f"lalme_a_{i}"
        plot_name += suffix
        plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      images.append(plot_to_image(None))

    if summary_writer:
      plot_name = "lalme_a"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )

  if lp_floating is not None:
    images = []
    for lp_ in lp_floating:
      # p = lalme_dataset['num_profiles_anchor']
      p_ = np.where(lalme_dataset['LP'] == lp_)[0][0]
      axs = az.plot_pair(
          lalme_az,
          var_names=["loc_floating"],
          coords={"LP_floating": lp_},
          kind='scatter',
          scatter_kwargs=scatter_kwargs,
          marginals=True,
      )

      axs[1, 0].scatter(
          x=lalme_dataset['loc'][[p_], 0],
          y=lalme_dataset['loc'][[p_], 1],
          marker="X",
          s=200,
          color="red",
      )
      axs[0, 0].set_xlim([0, 1])
      axs[1, 0].set_xlim([0, 1])
      axs[1, 1].set_ylim([0, 1])
      plt.suptitle(f"Profile: {lp_}")

      if workdir_png:
        plot_name = f"floating_profile_{lp_:03d}_loc"
        plot_name += suffix
        plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
        # summary_writer.image(plot_name, plot_to_image(fig), step=step)
      images.append(plot_to_image(None))

    if summary_writer:
      plot_name = "lalme_loc_floating"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )
      del images

  if lp_floating_traces is not None:
    axs = az.plot_trace(
        lalme_az,
        var_names=["loc_floating"],
        coords={
            "LP_floating": lp_floating_traces,
            'coords': ['x']
        },
        compact=False,
    )
    for axs_i in axs:
      axs_i[0].set_xlim([0, 1])
    plt.tight_layout()

    if workdir_png:
      plot_name = "lalme_loc_floating_traces_x"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = "lalme_loc_floating_traces_x"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )
    axs = az.plot_trace(
        lalme_az,
        var_names=["loc_floating"],
        coords={
            "LP_floating": lp_floating_traces,
            'coords': ['y']
        },
        compact=False,
    )
    for axs_i in axs:
      axs_i[0].set_xlim([0, 1])
    plt.tight_layout()

    if workdir_png:
      plot_name = "lalme_loc_floating_traces_y"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = "lalme_loc_floating_traces_y"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

      

  if lp_floating_aux_traces is not None:
    axs = az.plot_trace(
        lalme_az,
        var_names=["loc_floating_aux"],
        coords={
            "LP_floating": lp_floating_aux_traces,
            'coords': ['x']
        },
        compact=False,
    )
    for axs_i in axs:
      axs_i[0].set_xlim([0, 1])
    plt.tight_layout()

    if workdir_png:
      plot_name = "lalme_loc_floating_aux_traces_x"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = "lalme_loc_floating_aux_traces_x"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

    axs = az.plot_trace(
        lalme_az,
        var_names=["loc_floating_aux"],
        coords={
            "LP_floating": lp_floating_aux_traces,
            'coords': ['y']
        },
        compact=False,
    )
    for axs_i in axs:
      axs_i[0].set_xlim([0, 1])
    plt.tight_layout()

    if workdir_png:
      plot_name = "lalme_loc_floating_aux_traces_y"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = "lalme_loc_floating_aux_traces_y"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if lp_floating_aux_grid10 is not None:
    fig, axs = profile_locations_grid(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_floating_aux_grid10,
        var_name='loc_floating_aux',
        coord="LP_floating",
        nrows=2,
        scatter_kwargs=scatter_kwargs,
    )
    if workdir_png:
      plot_name = "lalme_floating_aux_profiles_grid"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_floating_aux_profiles_grid"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )


  if lp_floating_grid10 is not None:

    fig, axs = profile_locations_grid(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_floating_grid10,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=2,
        scatter_kwargs=scatter_kwargs,
    )
    if workdir_png:
      plot_name = "lalme_floating_profiles_grid"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_floating_profiles_grid"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

    if mcmc_img is not None:
      fig, axs = profile_locations_img_level_curves(
        img = mcmc_img,
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_floating_grid10,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=2,
    )
      if workdir_png:
        plot_name = "lalme_floating_profiles_grid_mcmc_compare"
        plot_name += suffix
        plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      
      if use_wandb:
        image = plot_to_image(fig)
        images = wandb.Image(image, caption="VI vs MCMC")
        wandb.log({"VIvsMCMC": images}, step=step)

        # image = plot_to_image(fig)

        # if summary_writer:
        #   plot_name = "lalme_floating_profiles_grid"
        #   plot_name += suffix
        #   summary_writer.image(
        #       tag=plot_name,
        #       image=image,
        #       step=step,
        #   )


  if lp_random_anchor is not None:
    images = []
    for lp_ in lp_random_anchor:
      # p = lalme_dataset['num_profiles_anchor']
      p_ = np.where(lalme_dataset['LP'] == lp_)[0][0]
      axs = az.plot_pair(
          lalme_az,
          var_names=["loc_random_anchor"],
          coords={"LP_anchor": lp_},
          kind='scatter',
          scatter_kwargs=scatter_kwargs,
          marginals=True,
      )
      axs[1, 0].scatter(
          x=lalme_dataset['loc'][[p_], 0],
          y=lalme_dataset['loc'][[p_], 1],
          marker="X",
          s=200,
          color="red",
      )
      axs[0, 0].set_xlim([0, 1])
      axs[1, 0].set_xlim([0, 1])
      axs[1, 1].set_ylim([0, 1])
      plt.suptitle(f"Profile: {lp_}")

      if workdir_png:
        plot_name = f"anchor_profile_{lp_:03d}_loc"
        plot_name += suffix
        plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
        # summary_writer.image(plot_name, plot_to_image(fig), step=step)
      images.append(plot_to_image(None))

    if summary_writer:
      plot_name = "lalme_loc_random_anchor"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )
      del images

  if lp_random_anchor_grid10 is not None:
    fig, axs = profile_locations_grid(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_random_anchor_grid10,
        var_name='loc_random_anchor',
        coord="LP_anchor",
        nrows=2,
        scatter_kwargs=scatter_kwargs,
    )
    if workdir_png:
      plot_name = "lalme_random_anchor_profiles_grid"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_random_anchor_profiles_grid"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if lp_anchor_val is not None:
    # lp_anchor_val = np.setdiff1d(lp_anchor_val, [104, 138, 1198, 1301, 1348])
    fig, axs = profile_locations_grid(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_anchor_val,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=(len(lp_anchor_val) // 5 + (1 if len(lp_anchor_val) % 5 else 0)),
        scatter_kwargs=scatter_kwargs,
        MSEs_dict=MSEs_anchor_val_dict,
    )
    if workdir_png:
      plot_name = "lalme_lp_anchor_val_grid"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_lp_anchor_val_grid"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if lp_anchor_val_grid30 is not None:
    # LP_ixs = jnp.where(LPs==lp_anchor_val_grid30)[0]
    # MSEs = error_loc_out_pointwise['dist_floating'][LP_ixs]
    # MSEs_dict = {lp: mse for lp, mse in zip(lp_floating_grid10, MSEs)}
    # lp_anchor_val = np.setdiff1d(lp_anchor_val, [104, 138, 1198, 1301, 1348])
    fig, axs = profile_locations_grid(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_anchor_val_grid30,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=10,
        scatter_kwargs=scatter_kwargs,
        MSEs_dict=MSEs_anchor_val_dict,
    )
    if workdir_png:
      plot_name = "lalme_lp_anchor_val_grid30"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_lp_anchor_val_grid30"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if lp_anchor_val_grid28 is not None:
    # lp_anchor_val = np.setdiff1d(lp_anchor_val, [104, 138, 1198, 1301, 1348])
    fig, axs = profile_locations_grid(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_anchor_val_grid28,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=4,
        scatter_kwargs=scatter_kwargs,
        MSEs_dict=MSEs_anchor_val_dict,
    )
    if workdir_png:
      plot_name = "lalme_lp_anchor_val_grid28"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_lp_anchor_val_grid28"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if lp_anchor_val_grid21 is not None:
    # lp_anchor_val = np.setdiff1d(lp_anchor_val, [104, 138, 1198, 1301, 1348])
    fig, axs = profile_locations_grid(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_anchor_val_grid21,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=3,
        scatter_kwargs=scatter_kwargs,
        MSEs_dict=MSEs_anchor_val_dict,
    )
    if workdir_png:
      plot_name = "lalme_lp_anchor_val_grid21"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_lp_anchor_val_grid21"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if lp_anchor_val_grid10 is not None:
    # lp_anchor_val = np.setdiff1d(lp_anchor_val, [104, 138, 1198, 1301, 1348])
    fig, axs = profile_locations_grid(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_anchor_val_grid10,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=2,
        scatter_kwargs=scatter_kwargs,
        MSEs_dict=MSEs_anchor_val_dict,
    )
    if workdir_png:
      plot_name = "lalme_lp_anchor_val_grid10"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_lp_anchor_val_grid10"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if lp_anchor_test is not None:
    fig, axs = profile_locations_grid(
        lalme_az=lalme_az,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_anchor_test,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=(len(lp_anchor_test) // 5 +
               (1 if len(lp_anchor_test) % 5 else 0)),
        scatter_kwargs=scatter_kwargs,
    )
    if workdir_png:
      plot_name = "lalme_lp_anchor_test_grid"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_lp_anchor_test_grid"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )


def posterior_samples_compare(
    # prng_key: PRNGKey,
    lalme_az_1: Mapping[str, Any],
    lalme_az_2: Mapping[str, Any],
    lalme_dataset: Dict[str, Any],
    step: int,
    lp_floating_grid10: Optional[List[int]] = None,
    show_mu: bool = False,
    show_zeta: bool = False,
    wass_dists: Optional[dict] = None,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
    suffix: str = '',
    scatter_kwargs={'alpha': 0.07},
    data_labels=["lalme_az_1", "lalme_az_2"],
):
  """Plot comparison two sets of posterior samples.

  This method is mainly intended to compare MCMC vs Variational.
  The first dictionary of samples is plotted as a heatmap, the second dictionary
  overplaced as level curves.
  """

  if lp_floating_grid10 is not None:
    fig, _ = profile_locations_grid(
        lalme_az=lalme_az_1,
        lalme_az_2=lalme_az_2,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_floating_grid10,
        wass_dists=wass_dists,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=2,
        scatter_kwargs=scatter_kwargs,
    )
    if workdir_png:
      plot_name = "lalme_floating_profiles_grid_compare"
      plot_name += suffix
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_floating_profiles_grid_compare"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if show_mu:
    axs = az.plot_density(
        [lalme_az_1, lalme_az_2],
        data_labels=data_labels,
        var_names=['mu'],
        grid=(1, lalme_dataset['num_items']),
        figsize=(3 * lalme_dataset['num_items'], 2.5),
        hdi_prob=1.0,
        shade=0.2,
    )
    max_ = float(
        max(lalme_az_1.posterior.mu.max(), lalme_az_2.posterior.mu.max()))
    max_ = 40.
    for axs_i in axs[0]:
      axs_i.set_xlim([0, max_])
    plt.tight_layout()
    if workdir_png:
      plot_name = "lalme_mu_compare"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      image = plot_to_image(None)

    if summary_writer:
      plot_name = "lalme_mu_compare"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if show_zeta:
    axs = az.plot_density(
        [lalme_az_1, lalme_az_2],
        data_labels=data_labels,
        var_names=['zeta'],
        grid=(1, lalme_dataset['num_items']),
        figsize=(3 * lalme_dataset['num_items'], 2.5),
        hdi_prob=1.0,
        shade=0.2,
    )
    for axs_i in axs[0]:
      axs_i.set_xlim([0, 1])
    plt.tight_layout()
    if workdir_png:
      plot_name = "lalme_zeta_compare"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      image = plot_to_image(None)

    if summary_writer:
      plot_name = "lalme_zeta_compare"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )


def posterior_samples_level_curves(
    # prng_key: PRNGKey,
    lalme_az_1: Mapping[str, Any],
    lalme_dataset: Dict[str, Any],
    step: int,
    lp_floating_grid10: Optional[List[int]] = None,
    show_mu: bool = False,
    show_zeta: bool = False,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
    suffix: str = '',
    scatter_kwargs={'alpha': 0.07},
    data_labels=["lalme_az_1", "lalme_az_2"],
):
  """Plot level curves from posterior samples.
  """

  if lp_floating_grid10 is not None:
    fig, _ = profile_locations_grid(
        lalme_az=lalme_az_1,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_floating_grid10,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=2,
        scatter_kwargs=scatter_kwargs,
    )
    if workdir_png:
      plot_name = "lalme_floating_profiles_level_curves"
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_floating_profiles_level_curves"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if show_mu:
    axs = az.plot_density(
        [lalme_az_1, lalme_az_2],
        data_labels=data_labels,
        var_names=['mu'],
        grid=(1, lalme_dataset['num_items']),
        figsize=(3 * lalme_dataset['num_items'], 2.5),
        hdi_prob=1.0,
        shade=0.2,
    )
    max_ = float(
        max(lalme_az_1.posterior.mu.max(), lalme_az_2.posterior.mu.max()))
    max_ = 40.
    for axs_i in axs[0]:
      axs_i.set_xlim([0, max_])
    plt.tight_layout()
    if workdir_png:
      plot_name = "lalme_mu_level_curves"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      image = plot_to_image(None)

    if summary_writer:
      plot_name = "lalme_mu_level_curves"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if show_zeta:
    axs = az.plot_density(
        [lalme_az_1, lalme_az_2],
        data_labels=data_labels,
        var_names=['zeta'],
        grid=(1, lalme_dataset['num_items']),
        figsize=(3 * lalme_dataset['num_items'], 2.5),
        hdi_prob=1.0,
        shade=0.2,
    )
    for axs_i in axs[0]:
      axs_i.set_xlim([0, 1])
    plt.tight_layout()
    if workdir_png:
      plot_name = "lalme_zeta_level_curves"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      image = plot_to_image(None)

    if summary_writer:
      plot_name = "lalme_zeta_level_curves"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )







def lalme_az_from_samples(
    lalme_dataset: Dict[str, Any],
    model_params_global: ModelParamsGlobal,
    model_params_locations: Optional[ModelParamsLocations] = None,
    model_params_gamma: Optional[ModelParamsGammaProfiles] = None,
    thinning: Optional[int] = 1,
) -> InferenceData:
  """Converts a posterior dictionary to an ArviZ InferenceData object.

  Args:
    posterior_dict: Dictionary of posterior samples.

  Returns:
    ArviZ InferenceData object.
  """

  items_ = [clean_filename(i) for i in lalme_dataset['items']]

  ### Global parameters
  assert model_params_global.mu.ndim == 3 and (
      model_params_global.mu.shape[0] < model_params_global.mu.shape[1]), (
          "Arrays in model_params_global" +
          "are expected to have shapes: (num_chains, num_samples, ...)")

    
  samples_dict = model_params_global._asdict()
  num_chains = samples_dict['mu'].shape[0]
  num_samples = samples_dict['mu'].shape[1]
  thin_idxs = jnp.where(jnp.arange(num_samples)%thinning==0)[0]

  samples_dict.update(
      {f"W_{i}": x for i, x in enumerate(samples_dict['mixing_weights_list'])})
  del samples_dict['mixing_weights_list']
  samples_dict.update(
      {f"a_{i}": x for i, x in enumerate(samples_dict['mixing_offset_list'])})
  del samples_dict['mixing_offset_list']

  num_basis_gps, num_inducing_points = samples_dict['gamma_inducing'].shape[-2:]

  coords_lalme = {
      "items": items_,
      "gp_basis": range(num_basis_gps),
      "loc_inducing": range(num_inducing_points),
  }
  dims_lalme = {
      "mu": ["items"],
      "zeta": ["items"],
      "gamma_inducing": ["gp_basis", "loc_inducing"],
  }
  for i, item_i in enumerate(items_):
    forms_i = [clean_filename(f_i) for f_i in lalme_dataset['forms'][i]]
    coords_lalme.update({f"forms_{item_i}": forms_i})
    dims_lalme.update({f"W_{i}": ["gp_basis", f"forms_{item_i}"]})
    dims_lalme.update({f"a_{i}": [f"forms_{item_i}"]})

  ### Linguistic Profiles Locations
  if model_params_locations is not None:
    coords_lalme.update({
        "LP_anchor":
            lalme_dataset['LP'][:lalme_dataset['num_profiles_anchor']],
        "LP_floating":
            lalme_dataset['LP'][-lalme_dataset['num_profiles_floating']:],
        "coords": ["x", "y"],
    })
    samples_dict.update(model_params_locations._asdict())
    if model_params_locations.loc_floating is not None:
      assert model_params_locations.loc_floating.ndim == 4 and (
          model_params_locations.loc_floating.shape[0] <
          model_params_locations.loc_floating.shape[1]), (
              "Arrays in model_params_locations" +
              "are expected to have shapes:" + "(num_chains, num_samples, ...)")
      dims_lalme.update({"loc_floating": ["LP_floating", "coords"]})
    if model_params_locations.loc_floating_aux is not None:
      assert model_params_locations.loc_floating_aux.ndim == 4 and (
          model_params_locations.loc_floating_aux.shape[0] <
          model_params_locations.loc_floating_aux.shape[1]), (
              "Arrays in model_params_locations" +
              "are expected to have shapes:" + "(num_chains, num_samples, ...)")
      dims_lalme.update({"loc_floating_aux": ["LP_floating", "coords"]})
    if model_params_locations.loc_random_anchor is not None:
      assert model_params_locations.loc_random_anchor.ndim == 4 and (
          model_params_locations.loc_random_anchor.shape[0] <
          model_params_locations.loc_random_anchor.shape[1]), (
              "Arrays in model_params_locations" +
              "are expected to have shapes:" + "(num_chains, num_samples, ...)")
      dims_lalme.update({"loc_random_anchor": ["LP_anchor", "coords"]})

  ### Gamma fields on profiles locations
  if model_params_gamma is not None:
    assert model_params_gamma.gamma_anchor.ndim == 4 and (
        model_params_gamma.gamma_anchor.shape[0] <
        model_params_gamma.gamma_anchor.shape[1]), (
            "Arrays in model_params_gamma" +
            "are expected to have shapes: (num_chains, num_samples, ...)")

    samples_dict.update(model_params_gamma._asdict())
    if model_params_gamma.gamma_anchor is not None:
      dims_lalme.update({"gamma_anchor": ["gp_basis", "LP_anchor"]})
    if model_params_gamma.gamma_floating is not None:
      dims_lalme.update({"gamma_floating": ["gp_basis", "LP_floating"]})
    if model_params_gamma.gamma_floating_aux is not None:
      dims_lalme.update({"gamma_floating_aux": ["gp_basis", "LP_floating"]})
    if model_params_gamma.gamma_random_anchor is not None:
      dims_lalme.update({"gamma_random_anchor": ["gp_basis", "LP_anchor"]})

  # Remove empty entries in samples_dict
  for k, v in samples_dict.copy().items():
    if v is None:
      del samples_dict[k]

  if thinning!=1:
    samples_dict = {i:x[:,thin_idxs,...] for i,x in samples_dict.items()}

  shape_match = [((x.shape[0]== num_chains) and (x.shape[1]== thin_idxs.shape[0])) for x in samples_dict.values()]
  assert sum(shape_match) == len(shape_match)

  lalme_az = az.convert_to_inference_data(
      samples_dict,
      coords=coords_lalme,
      dims=dims_lalme,
  )

  return lalme_az




def lalme_priorhparam_compare_plots_arviz(
    lalme_az_list: List[InferenceData],
    lalme_dataset: Dict[str, Any],
    prior_hparams_str_list: List[str],
    step: Optional[int] = 0,
    show_basis_fields: bool = False,
    show_W_items: Optional[List[str]] = None,
    show_a_items: Optional[List[str]] = None,
    lp_anchor_val_grid10: Optional[List[int]] = None,
    lp_floating_grid10: Optional[List[int]] = None,
    lp_random_anchor_grid10: Optional[List[int]] = None,
    lp_anchor_val: Optional[List[int]] = None,
    lp_anchor_test: Optional[List[int]] = None,
    loc_inducing: Optional[Array] = None,
    workdir_png: Optional[str] = None,
    summary_writer: Optional[SummaryWriter] = None,
    suffix: str = 'priorhp_compare',
    scatter_kwargs={'alpha': 0.07},
):

  if show_basis_fields:
    assert loc_inducing is not None, "loc_inducing must be provided to plot basis fields"
    for lalme_az_j, prior_hparams_j in zip(lalme_az_list,prior_hparams_str_list):
      fig, _ = plot_basis_fields_az(
          lalme_az=lalme_az_j,
          lalme_dataset=lalme_dataset,
          loc_inducing=loc_inducing,
      )
      plot_name = f"lalme_basis_fields_{prior_hparams_j}"
      plot_name += suffix
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        summary_writer.image(plot_name, plot_to_image(fig), step=step)

  if show_W_items is not None:
    idx_ = np.intersect1d(
        show_W_items, lalme_dataset['items'], return_indices=True)[2]
    images = []
    colors = ['blue', 'orange', 'green', 'black', 'purple']
    # The kind="forestplot" generates credible intervals, where the central points are 
    # the estimated posterior means, the thick lines are the central quartiles, 
    # and the thin lines represent the 100 x hdiprob)% highest density intervals.
    # for i in [31,56,60,61,67,70]:
      
    #   W_samples_forms = lalme_az_list[0].posterior[f'W_{i}'].squeeze()
    #   fig, ax = plt.subplots(1, W_samples_forms.shape[-2], figsize=(10,5))      
    #   for b_ix, a in enumerate(ax.flatten()):
    #     W_samples = W_samples_forms[:,b_ix,:]
    #     sns.boxplot(W_samples, ax=a)
         

    #   if workdir_png:
    #     plot_name = f"lalme_W_{i}_{prior_hparams_str_list[0]}"
    #     plot_name += suffix
    #     plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    #   images.append(plot_to_image(None))
    for i in [60,61]:#[31,56,60,61,67,70]
      axs = az.plot_forest(lalme_az_list, model_names=prior_hparams_str_list, 
                            var_names=[f"W_{i}"],  kind='forestplot',
                            colors=colors[:len(lalme_az_list)],
                            figsize=(8, lalme_az_list[0].posterior[f'W_{i}'].shape[-1]*4))

      axs[0].legend(loc=None) # makes legend disappear, as legend=False does not work
      axs[0].set_title('')
      legend_patches = [Patch(facecolor=colors[j], label=prior_hparams_str_list[j]) for j in np.arange(len(prior_hparams_str_list))]
      plt.figlegend(handles=legend_patches, loc='lower center', ncols=2, fontsize=11)
      # plt.suptitle(f"LMC weights {lalme_dataset['items'][i]}")
      plt.tight_layout() 
      plt.subplots_adjust(left=None, 
                          bottom=max((-0.02*lalme_az_list[0].posterior[f'W_{i}'].shape[-1]+0.19,0)), 
                          right=None, top=0.98, wspace=0.3, hspace=0.4)            

      if workdir_png:
        plot_name = f"lalme_W_{i}_2_"
        plot_name += suffix
        plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      images.append(plot_to_image(None))

    if summary_writer:
      plot_name = "lalme_W"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )

  if show_a_items is not None:
    idx_ = np.intersect1d(
        show_a_items, lalme_dataset['items'], return_indices=True)[2]
    images = []
    for i in idx_:
      axs = az.plot_forest(lalme_az_list, model_names=prior_hparams_str_list, 
                            var_names=[f"a_{i}"],  kind='forestplot',
                            colors=colors[:len(lalme_az_list)],
                            figsize=(8, lalme_az_list[0].posterior[f'W_{i}'].shape[-1]))

      axs[0].legend(loc=None) # makes legend disappear, as legend=False does not work
      axs[0].set_title('')
      legend_patches = [Patch(facecolor=colors[j], label=prior_hparams_str_list[j]) for j in np.arange(len(prior_hparams_str_list))]
      plt.figlegend(handles=legend_patches, loc='lower center', ncols=2, fontsize=11)
      # plt.suptitle(f"LMC offsets {lalme_dataset['items'][i]}")
      plt.tight_layout()   
      plt.subplots_adjust(left=None, 
                          bottom=max((-0.02*lalme_az_list[0].posterior[f'W_{i}'].shape[-1]+0.19,0)), 
                          right=None, top=0.98, wspace=0.3, hspace=0.4)    
      if workdir_png:
        plot_name = f"lalme_a_{i}"
        plot_name += suffix
        plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      images.append(plot_to_image(None))

    if summary_writer:
      plot_name = "lalme_a"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )

  if lp_anchor_val_grid10:
    for lalme_az_j, prior_hparams_j in zip(lalme_az_list,prior_hparams_str_list):
      fig, axs = profile_locations_grid(
          lalme_az=lalme_az_j,
          lalme_dataset=lalme_dataset,
          profiles_id=lp_anchor_val_grid10,
          var_name='loc_floating',
          coord="LP_floating",
          nrows=2,
          scatter_kwargs=scatter_kwargs,
      )
      if workdir_png:
        plot_name = f"lalme_val_anchor_profiles_grid_{prior_hparams_j}"
        plot_name += suffix
        plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      image = plot_to_image(fig)

      if summary_writer:
        plot_name = f"lalme_val_anchor_profiles_grid_{prior_hparams_j}"
        plot_name += suffix
        summary_writer.image(
            tag=plot_name,
            image=image,
            step=step,
        )

  if lp_floating_grid10 is not None:
    fig, axs = profile_locations_grid(
        lalme_az_list=lalme_az_list,
        prior_hparams_str_list=prior_hparams_str_list,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_floating_grid10,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=2,
        scatter_kwargs=scatter_kwargs,
    )
    if workdir_png:
      plot_name = "lalme_floating_profiles_grid"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_floating_profiles_grid"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if lp_random_anchor_grid10 is not None:
    fig, axs = profile_locations_grid(
        lalme_az_list=lalme_az_list,
        prior_hparams_str_list=prior_hparams_str_list,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_random_anchor_grid10,
        var_name='loc_random_anchor',
        coord="LP_anchor",
        nrows=2,
        scatter_kwargs=scatter_kwargs,
    )
    if workdir_png:
      plot_name = "lalme_random_anchor_profiles_grid"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_random_anchor_profiles_grid"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if lp_anchor_val is not None:
    # lp_anchor_val = np.setdiff1d(lp_anchor_val, [104, 138, 1198, 1301, 1348])
    fig, axs = profile_locations_grid(
        lalme_az_list=lalme_az_list,
        prior_hparams_str_list=prior_hparams_str_list,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_anchor_val,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=(len(lp_anchor_val) // 5 + (1 if len(lp_anchor_val) % 5 else 0)),
        scatter_kwargs=scatter_kwargs,
    )
    if workdir_png:
      plot_name = "lalme_lp_anchor_val_grid"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_lp_anchor_val_grid"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if lp_anchor_test is not None:
    fig, axs = profile_locations_grid(
        lalme_az_list=lalme_az_list,
        prior_hparams_str_list=prior_hparams_str_list,
        lalme_dataset=lalme_dataset,
        profiles_id=lp_anchor_test,
        var_name='loc_floating',
        coord="LP_floating",
        nrows=(len(lp_anchor_test) // 5 +
               (1 if len(lp_anchor_test) % 5 else 0)),
        scatter_kwargs=scatter_kwargs,
    )
    if workdir_png:
      plot_name = "lalme_lp_anchor_test_grid"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = "lalme_lp_anchor_test_grid"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )
