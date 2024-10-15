"""Plot methods for the epidemiology model."""
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") #, module='seaborn')

import pathlib

import numpy as np
import jax 
import jax.numpy as jnp
import pandas as pd
from scipy.stats import wasserstein_distance

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import itertools
import pickle
import sympy

from modularbayes._src.utils.training import TrainState
from modularbayes import colour_fader, plot_to_image, normalize_images

from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Mapping, Optional, PRNGKey,
                                      SummaryWriter, Tuple, List)

JointGrid = sns.JointGrid
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256) 


def plot_phi(
    phi: Array,
    theta: Array,
    priorhp:Array,
    eta: Optional[float] = None,
    xlim: Optional[Tuple[int]] = None,
    ylim: Optional[Tuple[int]] = None,
) -> Tuple[Figure, Axes]:
  """Relationship between HPV prevalence vs Cancer Incidence ."""

  n_samples, phi_dim = phi.shape

  loglambda = theta[:, [0]] + theta[:, [1]] * phi

  if xlim is None:
    xlim = (None, None)
  if ylim is None:
    ylim = (None, None)

  fig, ax = plt.subplots(figsize=(6, 4))
  for m in range(phi_dim):
    ax.scatter(
        phi[:, m],
        loglambda[:, m],
        alpha=np.clip(100 / n_samples, 0., 1.),
        label=f'{m+1}')
  ax.set(
      title="HPV/Cancer model" + f"_c1_{priorhp[0]:.3f}" + f"_c2_{priorhp[1]:.3f}" + ("" if eta is None else f"\n eta {eta:.3f}"),
      xlabel='phi',
      ylabel='theta_0 + theta_1 * phi',
      xlim=xlim,
      ylim=ylim,
  )
  leg = ax.legend(title='Group', loc='lower center', ncol=4, fontsize='x-small')
  for lh in leg.legendHandles:
    lh.set_alpha(1)

  return fig, ax


def plot_theta(
    theta: Array,
    betahp:Array,
    eta: Optional[float] = None,
    xlim: Optional[Tuple[int]] = None,
    ylim: Optional[Tuple[int]] = None,
) -> JointGrid:
  """Posterior distribution of theta in the epidemiology model."""

  n_samples, theta_dim = theta.shape

  colour = colour_fader('black', 'gold', (eta if eta is not None else 1.0))

  posterior_samples_df = pd.DataFrame(
      theta, columns=[f"theta_{i}" for i in range(1, theta_dim + 1)])
  # fig, ax = plt.subplots(figsize=(10, 10))
  pars = {
      'alpha': np.clip(100 / n_samples, 0., 1.),
      'colour': colour,
  }

  # warnings.simplefilter(action='ignore', category=FutureWarning)
  grid = sns.JointGrid(
      x='theta_1',
      y='theta_2',
      data=posterior_samples_df,
      xlim=xlim,
      ylim=ylim,
      height=5)
  g = grid.plot_joint(
      sns.scatterplot,
      data=posterior_samples_df,
      alpha=pars['alpha'],
      color=pars['colour'])
  # warnings.simplefilter(action='ignore', category=FutureWarning)
  sns.kdeplot(
      posterior_samples_df['theta_1'],
      ax=g.ax_marg_x,
      # shade=True,
      color=pars['colour'],
      legend=False)
  
  sns.kdeplot(
      posterior_samples_df['theta_2'],
      ax=g.ax_marg_y,
      # shade=True,
      color=pars['colour'],
      legend=False,
      vertical=True,
      )
  # Add title
  g.fig.subplots_adjust(top=0.9)
  g.fig.suptitle("HPV/Cancer model" + f"_conc1_{betahp[0]:.3f}" + f"_conc0_{betahp[1]:.3f}" +
                 ("" if eta is None else f"\n eta {eta:.3f}"))

  return grid


def posterior_samples(
    posterior_sample_dict: Mapping[str, Any],
    step: int,
    betahp: Array,
    summary_writer: Optional[SummaryWriter] = None,
    eta: Optional[float] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Visualise samples from the approximate posterior distribution."""

  images = []

  # Plot relation: prevalence (lambda) vs incidence (phi)
  plot_name = "epidemiology_phi"
  plot_name = plot_name + f"_conc1_{betahp[0]:.3f}" + f"_conc2_{betahp[1]:.3f}" + ("" if (eta is None) else f"_eta_{eta:.3f}")
  fig, _ = plot_phi(
      posterior_sample_dict['phi'],
      posterior_sample_dict['theta'],
      eta=1.0 if (eta is None) else float(eta),
      betahp=betahp,
      xlim=[0, 0.3],
      ylim=[-2.5, 2],
  )
  if workdir_png:
    fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
  # summary_writer.image(plot_name, plot_to_image(fig), step=step)
  images.append(plot_to_image(fig))

  # Plot relation: theta_2 vs theta_1
  plot_name = "epidemiology_theta"
  plot_name = plot_name + f"_conc1_{betahp[0]:.3f}" + f"_conc2_{betahp[1]:.3f}" + ("" if (eta is None) else f"_eta_{eta:.3f}")
  grid = plot_theta(
      posterior_sample_dict['theta'],
      eta=1.0 if (eta is None) else float(eta),
      betahp=betahp,
      xlim=[-3, -1],
      ylim=[5, 35],
  )
  fig = grid.fig
  if workdir_png:
    fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
  # summary_writer.image(plot_name, plot_to_image(fig), step=step)
  images.append(plot_to_image(fig))

  # Log all images
  if summary_writer:
    plot_name = "epidemiology_posterior_samples"
    plot_name = plot_name + f"_conc1_{betahp[0]:.3f}" + f"_conc2_{betahp[1]:.3f}" + ("" if (eta is None) else f"_eta_{eta:.3f}")
    summary_writer.image(
        tag=plot_name,
        image=normalize_images(images),
        step=step,
        max_outputs=len(images),
    )


    
def plot_final_posterior_vs_true(par_name_plot: str,
                                 true_params: dict,
                                 par_samples: Optional[Array] = None,
                                 compare_samples: Optional[Array] = None,
                                 alternative_samples: Optional[Array] = None,
                                 mcmc_samples: Optional[Array] = None,
                                 ):
    try:    
        n_samples, n_groups = par_samples.shape
    except:
        try:
            n_samples, n_groups = compare_samples.shape
        except:
            try:
                n_samples, n_groups = alternative_samples.shape
            except:
                n_samples, n_groups = mcmc_samples.shape

    n_plots_per_row = min(n_groups, 10)
    n_rows = (n_groups + n_plots_per_row - 1) // n_plots_per_row

    if n_groups > 10 and not sympy.ntheory.isprime(n_groups):
        n_plots_per_row = n_groups // n_rows
    elif n_groups > 10 and sympy.intheory.isprime(n_groups):
        n_plots_per_row = n_groups // (n_rows - 1)

    mpl.rcParams['font.size'] = 20
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['legend.fontsize'] = 20

    fig, ax = plt.subplots(n_rows, n_plots_per_row, figsize=(19, 3*n_rows), sharey=True)
    
    
    for i in range(n_rows):
        for j in range(n_plots_per_row):
            col = i * n_plots_per_row + j
            try:
                a = ax[i,j]
            except:
                a = ax[j]
            if col < n_groups:
                if par_samples is not None:
                    sns.kdeplot(par_samples[col], ax=a, 
                                color='red',  linewidth=3., 
                                linestyle='dotted', 
                                label=r'VP at $\psi_{True}$')
                a.axvline(true_params['mu'][col], color='red', linewidth=1., label='True parameter value')
                a.set_xlabel(fr'${par_name_plot}_{{{col}}}$')
                if alternative_samples is not None:
                    sns.kdeplot(alternative_samples[col], ax=a, linestyle='dotted',
                                 linewidth=3., alpha=0.7, label=r'VMP at alternative $\psi_{alt}$')
                if compare_samples is not None:
                    sns.kdeplot(compare_samples[col], ax=a, label=fr'VMP at selected $\psi^\ast$', 
                                linewidth=3.,color='black', linestyle='dashed', alpha=0.7)
                if mcmc_samples is not None:
                    sns.kdeplot(mcmc_samples[col], ax=a, label=r'MCMC at $\psi_{True}$', 
                                linewidth=3., alpha=0.7)

                # a.set_xlim(-7, 7)
            else:
                fig.delaxes(a)

    plt.tight_layout()

    # Add legend
    try:
        a0 = ax[0,0]
    except:
        a0 = ax[0]
    handles, labels = a0.get_legend_handles_labels()
    if n_rows==1:
        plt.subplots_adjust(bottom=0.5)
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.25), ncol=4)
    else:
        plt.subplots_adjust(bottom=0.1)
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)

    return fig



def plot_final_posterior_vs_true_OLD(mu_samples: Array,
                             sigma_samples: Array,
                             true_params: dict):
    
    n_samples, n_groups = mu_samples.shape
    fig, ax = plt.subplots(2,n_groups, figsize=(15,3), sharey=True)

    for col in jnp.arange(n_groups):
        sns.kdeplot(mu_samples[col], ax=ax[0, col])
        ax[0,col].axvline(true_params['mu'][col], color='red', linewidth=0.5)
        ax[0,col].set_xlabel(fr'$\mu_{col}$', fontsize=12)
        
        sns.kdeplot(sigma_samples[col], ax=ax[1, col])
        ax[1,col].axvline(true_params['sigma'][col], color='red', linewidth=0.5)
        ax[1,col].set_xlabel(fr'$\sigma_{col}$', fontsize=12)

    plt.tight_layout()

    return fig

def old_plot_optim_hparams_vs_true(path: str,
                               init_names: list,
                               optimiser_name: str,
                               true_vals: dict,
                               hp_names: list,
                               ):

    names_latex = {'mu_prior_mean_m': 'm',
                'mu_prior_scale_s':'s',
                'sigma_prior_concentration':'$g_1$',
                'sigma_prior_scale':'$g_2$',
                }
    try:
        old_to_new = {'mu_loc':'mu_prior_mean_m', 'mu_scale':'mu_prior_scale_s', 'sigma_conc':'sigma_prior_concentration', 'sigma_scale':'sigma_prior_scale'}
        true_vals = dict((old_to_new[key], value) for (key, value) in true_vals.items()).copy()
    except: 
        pass

    n_plots = len(hp_names)+1
    fig, ax = plt.subplots(int(n_plots/2)+int(n_plots%2>0), 2, 
                        figsize=(7,3.5*(int(n_plots/2)+int(n_plots%2>0))))

    for init_type in init_names:
        try:
            with open(path + f'/hp_info_allhps_{init_type}_{optimiser_name}_elbo.sav', 'rb') as fr:
                res_elbo = pickle.load(fr)
            with open(path + f'/hp_info_allhps_{init_type}_{optimiser_name}_elpd_waic.sav', 'rb') as fr:
                res_waic = pickle.load(fr)
            loss_type = ''
        except:
            try:
                with open(path + f'/hp_info_allhps_{init_type}_{optimiser_name}_elbo.sav', 'rb') as fr:
                    res_elbo = pickle.load(fr)
                loss_type = '_elbo'
            except:
                with open(path + f'/hp_info_allhps_{init_type}_{optimiser_name}_elpd_waic.sav', 'rb') as fr:
                    res_waic = pickle.load(fr)
                loss_type = '_elpd_waic'
        for a_ix, a in enumerate(ax.flatten()):
            if a_ix==0:
                try:
                    a2 = a.twinx()
                    l1 = a.plot(jnp.array(res_elbo['loss']), alpha=0.7, label='negative elbo')
                    a.set_ylabel('negative elbo')
                    l2 = a2.plot(jnp.array(res_waic['loss']), alpha=0.7, label='negative waic')
                    a2.set_ylabel('negative waic')
                    a.legend( handles=l1+l2)
                except:
                    try:
                       a.plot(jnp.array(res_elbo['loss']), alpha=0.7, label='negative elbo')
                    except:
                       a.plot(jnp.array(res_waic['loss']), alpha=0.7, label='negative waic')  
                a.set_xlabel('Iterations')
                a.set_title('Training loss')
                # if loglik_type=='z':
                #     a.set_ylim(top=85)
                # a.set_ylim(0.25, 0.26)
            elif a_ix <= n_plots-1: 
                try:
                    a2 = a.twinx() 
                    l1 = a.plot(jnp.array(res_elbo['params'])[:,a_ix-1], label='negative elbo')
                    a.set_ylabel('negative elbo')
                    l2 = a.plot(jnp.array(res_waic['params'])[:,a_ix-1], label='negative waic')
                    a2.set_ylabel('negative waic')
                    a.legend( handles=l1+l2)
                except:
                    try:
                        a.plot(jnp.array(res_elbo['params'])[:,a_ix-1], label='elbo')
                    except:    
                        a.plot(jnp.array(res_waic['params'])[:,a_ix-1], label='waic')
                hp_name = hp_names[a_ix-1]
                a.set_title('Trace plot for '+ names_latex[hp_name])
                a.set_xlabel('Iterations')
                a.axhline(true_vals[hp_name], color='black')
            else:
                a.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
    plt.savefig(path + f'/synth_hp_tuning_prior_hparams_{optimiser_name}{loss_type}.png')
    return fig

def plot_optim_hparams_vs_true(path: str,
                               init_names: list,
                               optimiser_name: str,
                               true_vals: dict,
                               hp_names: list,
                               loss_type: str,
                               ):

    names_latex = {'mu_prior_mean_m': 'm',
                'mu_prior_scale_s':'s',
                'sigma_prior_concentration':'$g_1$',
                'sigma_prior_scale':'$g_2$',
                }
    try:
        old_to_new = {'mu_loc':'mu_prior_mean_m', 'mu_scale':'mu_prior_scale_s', 'sigma_conc':'sigma_prior_concentration', 'sigma_scale':'sigma_prior_scale'}
        true_vals = dict((old_to_new[key], value) for (key, value) in true_vals.items()).copy()
    except: 
        pass

    n_plots = len(hp_names)+1
    fig, ax = plt.subplots(int(n_plots/2)+int(n_plots%2>0), 2, 
                        figsize=(7,3.5*(int(n_plots/2)+int(n_plots%2>0))))

    for init_type in init_names:
        with open(path + f'/hp_info_allhps_{init_type}_{optimiser_name}_{loss_type}.sav', 'rb') as fr:
            res = pickle.load(fr)
        for a_ix, a in enumerate(ax.flatten()):
            if a_ix==0:
                a.plot(jnp.array(res['loss']), alpha=0.7, label=f'negative {loss_type}')
                a.set_xlabel('Iterations')
                a.set_title('Training loss')
                # if loglik_type=='z':
                #     a.set_ylim(top=85)
                # a.set_ylim(0.25, 0.26)
            elif a_ix <= n_plots-1: 
                a.plot(jnp.array(res['params'])[:,a_ix-1], label=f'negative {loss_type}')
                hp_name = hp_names[a_ix-1]
                a.set_title('Trace plot for '+ names_latex[hp_name])
                a.set_xlabel('Iterations')
                a.axhline(true_vals[hp_name], color='black')
            else:
                a.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
    plt.savefig(path + f'/synth_hp_tuning_prior_hparams_{optimiser_name}_{loss_type}.png')
    return fig


