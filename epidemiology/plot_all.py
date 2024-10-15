"""Plot methods for the epidemiology model."""
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") #, module='seaborn')

import pathlib

import numpy as np
import jax 
import jax.numpy as jnp
import pandas as pd
import pickle
from scipy.stats import wasserstein_distance

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib as mpl
import itertools
import ot

from modularbayes._src.utils.training import TrainState
from modularbayes import colour_fader, plot_to_image, normalize_images

from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Mapping, Optional, PRNGKey,
                                      SummaryWriter, Tuple, List)

JointGrid = sns.JointGrid
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256) 

def plot_phi_loglambda(
    posterior_sample_dict: Mapping,
    priorhps: Mapping,
    smi_etas: Mapping,
    priorhp_toplot: Optional[Mapping] = {'eta_bayes':['priorhp_ones', 'priorhp_converged_bayes'],
                                         'eta_cut':['priorhp_ones', 'priorhp_converged_cut']},
    xlim: Optional[Tuple[int]] = None,
    ylim: Optional[Tuple[int]] = None,
) -> Figure:
  """Relationship between HPV prevalence vs Cancer Incidence ."""

  assert len(priorhp_toplot['eta_bayes'])==len(priorhp_toplot['eta_cut']), "Different length of lists under priorhp_toplot"
  col_n = len(priorhp_toplot['eta_bayes'])

  if xlim is None:
    xlim = (None, None)
  if ylim is None:
    ylim = (None, None)
  
  fig, ax = plt.subplots(col_n, 2, figsize=(8,6), sharex=True, sharey=True)
  plt.suptitle("HPV/Cancer model" , fontsize=13)
  
  for eta_ix, (eta_k, eta_v) in enumerate(smi_etas.items()):
    priorhp_ks = priorhp_toplot[eta_k]
    for p_ix, priorhp_k in enumerate(priorhp_ks):

        theta = posterior_sample_dict[eta_k][priorhp_k]['theta']
        phi = posterior_sample_dict[eta_k][priorhp_k]['phi']
        n_samples, phi_dim = phi.shape
        
        loglambda = theta[:, [0]] + theta[:, [1]] * phi
        for m in range(phi_dim):
           ax[p_ix, eta_ix].scatter(
              phi[:, m],
              loglambda[:, m],
              alpha=np.clip(100 / n_samples, 0., 1.),
              label=(f"{m+1}" if p_ix==eta_ix==0 else ""))
        ax[p_ix, eta_ix].set_title(f"eta {eta_v:.4f}" + f"\n c1: {priorhps[priorhp_k][0]:.2f}" + f" c2: {priorhps[priorhp_k][1]:.2f}", fontsize=10)
        ax[p_ix, eta_ix].set(
        #    title= f"eta {eta_v:.4f}" + f"\n c1: {priorhps[priorhp_k][0]:.2f}" + f" c2: {priorhps[priorhp_k][1]:.2f}",
           xlabel='phi',
           ylabel='theta_0 + theta_1 * phi',
        #    xlim=xlim,
        #    ylim=ylim,
           )
        
  leg = fig.legend(title='Group', loc='lower center', ncol=7, fontsize='x-small')
  for lh in leg.legendHandles:
    lh.set_alpha(1)

  fig.tight_layout()
  fig.subplots_adjust(left=0.05, bottom=0.2, right=None, top=0.85, wspace=0.3, hspace=0.5)

  return fig


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

def plot_elpd_one_variable(
      elpd_surface_dict: Mapping[str, Any],
      suptitle: str,
      xlabel: str,
      x_values: Array,
      indx: int,
      is_long:Optional[bool]=True,
):
    if is_long:
        fig, axs = plot_elpd_one_variable_long(
            elpd_surface_dict=elpd_surface_dict,
            suptitle=suptitle,
            xlabel=xlabel,
            x_values=x_values,
            indx=indx,
        )
    else:
        fig, axs = plot_elpd_one_variable_short(
            elpd_surface_dict=elpd_surface_dict,
            suptitle=suptitle,
            xlabel=xlabel,
            x_values=x_values,
            indx=indx,
        )
    
    return fig, axs


def plot_elpd_one_variable_long(
    elpd_surface_dict: Mapping[str, Any],
    suptitle: str,
    xlabel: str,
    x_values: Array,
    indx: int,
) -> Tuple[Figure, Axes]:
    # plot eta fixing hp
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(4 * 3, 4*3))
    fig.suptitle(suptitle)

    for mod_ix, mod_name in enumerate(['', '_y', '_z']):
        for i, metric in enumerate([elpd_surface_dict[f'lpd{mod_name}_all_eta'],
                                    -elpd_surface_dict[f'p_waic{mod_name}_all_eta'],
                                    elpd_surface_dict[f'elpd_waic{mod_name}_all_eta'],]):
            axs[mod_ix,i].set_title(['Full likelihood', 'Y module', 'Z module'][mod_ix])
            axs[mod_ix,i].plot(x_values,
                -metric.mean(indx))
            axs[mod_ix,i].set_xlabel(xlabel)
            axs[mod_ix,i].set_ylabel(['- LPD', 'p_WAIC', '- ELPD WAIC'][i])

    plt.tight_layout()
    return fig, axs

def plot_elpd_one_variable_short(
    elpd_surface_dict: Mapping[str, Any],
    suptitle: str,
    xlabel: str,
    x_values: Array,
    indx: int,
) -> Tuple[Figure, Axes]:
    # plot eta fixing hp
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4*1, 4*2))
    fig.suptitle(suptitle)

    for mod_ix, mod_name in enumerate(['_y', '_z']):
            axs[mod_ix].set_title(['Y module', 'Z module'][mod_ix])
            axs[mod_ix].plot(x_values,
                -elpd_surface_dict[f'elpd_waic{mod_name}_all_eta'].mean(indx))
            axs[mod_ix].set_xlabel(xlabel)
            axs[mod_ix].set_ylabel('- ELPD WAIC')

    plt.tight_layout()
    return fig, axs

def plot_posterior_phi_hprange(
      plot_two:bool,
      posterior_sample_dict,
      eta,
      priorhps,
      priorhp_main: Optional[Mapping[str, Mapping]] = 
      {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
        'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
):
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['legend.fontsize'] = 16

    colors = ['green', 'black', 'red', 'orange', 'pink']
    eta_k = eta[0]
    eta_v = eta[1]
    if plot_two:
                # Phis
        fig_phi, ax_phi = plt.subplots(1,2, figsize=(10, 7), sharex=True)
        ax_phi_flattened = ax_phi.flatten()
        priorhps_red = {k:v for k,v in priorhps.items() if k not in ['priorhp_ones']}
        for p_ix, (priorhp_k, priorhp_v) in enumerate(priorhps_red.items()):

            phi = posterior_sample_dict[eta_k][priorhp_k]['phi']       
            # phi plot
            for phi_ix, phi_no in enumerate([2,9]):
                if priorhp_k == priorhp_main['main'][eta_k]:
                    sns.kdeplot(phi[:,phi_no], ax=ax_phi_flattened[phi_ix], color='black')
                else:
                    sns.kdeplot(phi[:,phi_no], ax=ax_phi_flattened[phi_ix], color=colors[p_ix], alpha=0.3)
                ax_phi_flattened[phi_ix].set_xlabel(fr'$\delta_{{{phi_no+1}}}$', fontsize=15)
                ax_phi_flattened[phi_ix].set_ylabel(fr'Density', fontsize=15)  
                ax_phi_flattened[phi_ix].xaxis.set_tick_params(which='both', labelbottom=True)
        labels = [fr'$c_1$: {priorhps_red[k][0]}, $c_2$: {priorhps_red[k][1]}' for k in priorhps_red.keys()]

        fig_phi.legend(labels,
            loc='lower center', ncol=4, fontsize=11)

        fig_phi.tight_layout()


        fig_phi.subplots_adjust(left=None, bottom=0.15, right=None, top=0.9, wspace=0.2, hspace=0.3)

    else:
        # Phis
        fig_phi, ax_phi = plt.subplots(2,2, figsize=(10, 10), sharex=True)
        ax_phi_flattened = ax_phi.flatten()
        for p_ix, (priorhp_k, priorhp_v) in enumerate(priorhps.items()):

            phi = posterior_sample_dict[eta_k][priorhp_k]['phi']       
            # phi plot
            for phi_ix, phi_no in enumerate([7,8,9,12]):
                if priorhp_k == priorhp_main['main'][eta_k]:
                    sns.kdeplot(phi[:,phi_no], ax=ax_phi_flattened[phi_ix], color='black')
                else:
                    sns.kdeplot(phi[:,phi_no], ax=ax_phi_flattened[phi_ix], color=colors[p_ix], alpha=0.3)
                ax_phi_flattened[phi_ix].set_title(fr'$\phi_{{{phi_no+1}}}$', fontsize=15)

                ax_phi_flattened[phi_ix].xaxis.set_tick_params(which='both', labelbottom=True)
        if eta_v==0.0001:
            fig_phi.suptitle(r'$\phi$ distributions for eta $\approx 0$', fontsize=20)
        else:
            fig_phi.suptitle(fr'$\phi$ distributions for eta {eta_v:.3f}', fontsize=20)
        labels = [fr'$c_1$: {priorhps[k][0]}, $c_2$: {priorhps[k][1]}' for k in priorhps.keys()]

        fig_phi.legend(labels,
            loc='lower center', ncol=3, fontsize='medium')

        fig_phi.tight_layout()


        fig_phi.subplots_adjust(left=None, bottom=0.15, right=None, top=0.9, wspace=0.2, hspace=0.3)
    return fig_phi

def plot_posterior_phi_hprange_singlephi(
      posterior_sample_dict,
      phi_ix,
      eta,
      priorhps,
      priorhp_main: Optional[Mapping[str, Mapping]] = 
      {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
        'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
):

    colors = ['green', 'black', 'red', 'orange', 'pink']
    eta_k = eta[0]
    eta_v = eta[1]

    # Phis
    fig_phi, ax_phi = plt.subplots(figsize=(10, 10))
    
    for p_ix, (priorhp_k, priorhp_v) in enumerate(priorhps.items()):  
        phi = posterior_sample_dict[eta_k][priorhp_k]['phi'] 
        # phi plot
        if priorhp_k == priorhp_main['main'][eta_k]:
            sns.kdeplot(phi[:,phi_ix], ax=ax_phi, color='black')
        else:
            sns.kdeplot(phi[:,phi_ix], ax=ax_phi, color=colors[p_ix], alpha=0.3)
        ax_phi.set_title(fr'$\phi_{{{phi_ix+1}}}$', fontsize=15)

        ax_phi.xaxis.set_tick_params(which='both', labelbottom=True)
    if eta_v==0.0001:
        fig_phi.suptitle(r'$\phi$ distribution for eta $\approx 0$', fontsize=20)
    else:
        fig_phi.suptitle(fr'$\phi$ distribution for eta {eta_v:.3f}', fontsize=20)
    labels = [fr'$c_1$: {priorhps[k][0]}, $c_2$: {priorhps[k][1]}' for k in priorhps.keys()]

    fig_phi.legend(labels,
        loc='lower center', ncol=3, fontsize='medium')

    fig_phi.tight_layout()


    fig_phi.subplots_adjust(left=None, bottom=0.15, right=None, top=0.9, wspace=0.2, hspace=0.3)
    return fig_phi
    

def plot_posterior_theta_hprange(
      posterior_sample_dfs,
      smi_etas,
      priorhps,
      priorhp_main: Optional[Mapping[str, Mapping]] = 
      {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
        'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
):

    n_samples = posterior_sample_dfs['eta_bayes']['priorhp_converged_bayes'].shape[0]

    pars = {'alpha': np.clip(100 / n_samples, 0., 1.),
            # 'colour': colour,
        }
    df_main = pd.concat([posterior_sample_dfs['eta_bayes'][priorhp_main['main']['eta_bayes']], #4
                             posterior_sample_dfs['eta_cut'][priorhp_main['main']['eta_cut']]]) #0
    warnings.simplefilter(action='ignore', category=FutureWarning)
    grid = sns.JointGrid(
                        x='theta_1',
                        y='theta_2',
                        data=df_main,
                        hue='eta1',
                        xlim=[-3, -1], #[-3, 2],
                        ylim=[5, 35], #[-1, 35],
                        height=5)
    g = grid.plot_joint(sns.scatterplot, alpha=pars['alpha'],)
    g.ax_joint.get_legend().set_title(r'$\eta$ values')
    g.ax_joint.set_xlabel(r"$\theta_1$")
    g.ax_joint.set_ylabel(r"$\theta_2$")
    sns.kdeplot(
        posterior_sample_dfs['eta_bayes'][priorhp_main['main']['eta_bayes']]['theta_1'], #4
        ax=g.ax_marg_x,
        #legend=False,
        )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.kdeplot(
        posterior_sample_dfs['eta_bayes'][priorhp_main['main']['eta_bayes']]['theta_2'], #4
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 1, $c_1$: {priorhps[priorhp_main['main']['eta_bayes']][0]}, $c_2$: {priorhps[priorhp_main['main']['eta_bayes']][1]} ",
        vertical=True,
        )
    
    
    sns.kdeplot(
        posterior_sample_dfs['eta_cut'][priorhp_main['main']['eta_cut']]['theta_1'], #0
        ax=g.ax_marg_x,
        #legend=False,
        )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.kdeplot(
        posterior_sample_dfs['eta_cut'][priorhp_main['main']['eta_cut']]['theta_2'], #0
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 0, $c_1$: {priorhps[priorhp_main['main']['eta_cut']][0]}, $c_2$: {priorhps[priorhp_main['main']['eta_cut']][1]}",
        vertical=True,
        )
    
    sns.kdeplot(
        posterior_sample_dfs['eta_bayes'][priorhp_main['secondary']['eta_bayes']]['theta_1'],
        ax=g.ax_marg_x,
        alpha=0.3,
        color='#1f77b4',
        #legend=False,
        )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.kdeplot(
        posterior_sample_dfs['eta_bayes'][priorhp_main['secondary']['eta_bayes']]['theta_2'],
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 1, $c_1$: {priorhps[priorhp_main['secondary']['eta_bayes']][0]}, $c_2$: {priorhps[priorhp_main['secondary']['eta_bayes']][1]} ",
        vertical=True,
        alpha=0.3,
        color='#1f77b4',
        )
            
    sns.kdeplot(
        posterior_sample_dfs['eta_cut'][priorhp_main['secondary']['eta_cut']]['theta_1'],
        ax=g.ax_marg_x,
        alpha=0.3,
        color='orange',
        #legend=False,
        )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.kdeplot(
        posterior_sample_dfs['eta_cut'][priorhp_main['secondary']['eta_cut']]['theta_2'],
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 0, $c_1$: {priorhps[priorhp_main['secondary']['eta_cut']][0]}, $c_2$: {priorhps[priorhp_main['secondary']['eta_cut']][1]}",
        vertical=True,
        alpha=0.3,
        color='orange',
        )
    

    # Add title
    g.fig.subplots_adjust(top=0.9)
    # g.fig.suptitle(r'Joint SMI posterior for $\theta_1$ and $\theta_2$', fontsize=13)

    plt.legend(loc='upper right', bbox_to_anchor=(0,-0.2), ncols=2, fontsize=9)

    # g.ax_marg_y.legend(loc='lower left')
    sns.move_legend(g.ax_joint, "upper right")
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.3, right=None, top=0.99, wspace=0.5, hspace=0.5)

    return g.fig


def plot_posterior_theta_hprange_vsmcmc(
      posterior_sample_dfs,
      mcmc_dfs,
      smi_etas,
      priorhps,
      priorhp_main: Optional[Mapping[str, Mapping]] = 
      {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
        'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
):

    n_samples = posterior_sample_dfs['eta_bayes']['priorhp_converged_bayes'].shape[0]

    pars = {'alpha': np.clip(100 / n_samples, 0., 1.),
            # 'colour': colour,
        }
    df_main = pd.concat([posterior_sample_dfs['eta_bayes'][priorhp_main['main']['eta_bayes']], #4
                             posterior_sample_dfs['eta_cut'][priorhp_main['main']['eta_cut']]]) #0
    
    mcmc_1 = pd.DataFrame(mcmc_dfs['eta1'], columns=['theta_1', 'theta_2'])
    mcmc_1['eta1'] = '1 (mcmc)'
    mcmc_0 = pd.DataFrame(mcmc_dfs['eta0001'], columns=['theta_1', 'theta_2'])
    mcmc_0['eta1'] = '0 (mcmc)' 
    df_main = pd.concat([mcmc_1, mcmc_0, df_main]).copy()

    mcmc1 = df_main[df_main.eta1=='1 (mcmc)'][['theta_1', 'theta_2']].values 
    mcmc0 = df_main[df_main.eta1=='0 (mcmc)'][['theta_1', 'theta_2']].values 
    vmp1 = df_main[df_main.eta1=='= 1'][['theta_1', 'theta_2']].values
    vmp0 = df_main[df_main.eta1=='= 0'][['theta_1', 'theta_2']].values
    wass = {}
    for theta_ix in [0,1]:
        w1 = wasserstein_distance(mcmc1[:,theta_ix],vmp1[:,theta_ix])
        w0 = wasserstein_distance(mcmc0[:,theta_ix],vmp0[:,theta_ix])
        wass[f'theta{theta_ix}_w1'] = w1
        wass[f'theta{theta_ix}_w0'] = w0

    warnings.simplefilter(action='ignore', category=FutureWarning)
    grid = sns.JointGrid(
                        x='theta_1',
                        y='theta_2',
                        data=df_main,
                        hue='eta1',
                        xlim=[-3, -1], #[-3, 2],
                        ylim=[5, 35], #[-1, 35],
                        height=5)
    g = grid.plot_joint(sns.scatterplot, alpha=np.array([0.3]*n_samples*2 + [0.3]*n_samples*2), 
                        s=5, edgecolor=None, palette=['yellow', 'black', '#1f77b4', 'orange'])
    g.ax_joint.get_legend().set_title(r'$\eta$ values')
    g.ax_joint.set_xlabel(r"$\theta_1$")
    g.ax_joint.set_ylabel(r"$\theta_2$")
    sns.kdeplot(
        posterior_sample_dfs['eta_bayes'][priorhp_main['main']['eta_bayes']]['theta_1'], #4
        ax=g.ax_marg_x,
        #legend=False,
        )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.kdeplot(
        posterior_sample_dfs['eta_bayes'][priorhp_main['main']['eta_bayes']]['theta_2'], #4
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 1, $c_1$: {priorhps[priorhp_main['main']['eta_bayes']][0]}, $c_2$: {priorhps[priorhp_main['main']['eta_bayes']][1]} ",
        vertical=True,
        )
    
    
    sns.kdeplot(
        posterior_sample_dfs['eta_cut'][priorhp_main['main']['eta_cut']]['theta_1'], #0
        ax=g.ax_marg_x,
        #legend=False,
        )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.kdeplot(
        posterior_sample_dfs['eta_cut'][priorhp_main['main']['eta_cut']]['theta_2'], #0
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 0, $c_1$: {priorhps[priorhp_main['main']['eta_cut']][0]}, $c_2$: {priorhps[priorhp_main['main']['eta_cut']][1]}",
        vertical=True,
        )
    
    sns.kdeplot(
        posterior_sample_dfs['eta_bayes'][priorhp_main['secondary']['eta_bayes']]['theta_1'],
        ax=g.ax_marg_x,
        alpha=0.3,
        color='#1f77b4',
        #legend=False,
        )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.kdeplot(
        posterior_sample_dfs['eta_bayes'][priorhp_main['secondary']['eta_bayes']]['theta_2'],
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 1, $c_1$: {priorhps[priorhp_main['secondary']['eta_bayes']][0]}, $c_2$: {priorhps[priorhp_main['secondary']['eta_bayes']][1]} ",
        vertical=True,
        alpha=0.3,
        color='#1f77b4',
        )
            
    sns.kdeplot(
        posterior_sample_dfs['eta_cut'][priorhp_main['secondary']['eta_cut']]['theta_1'],
        ax=g.ax_marg_x,
        alpha=0.3,
        color='orange',
        #legend=False,
        )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.kdeplot(
        posterior_sample_dfs['eta_cut'][priorhp_main['secondary']['eta_cut']]['theta_2'],
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 0, $c_1$: {priorhps[priorhp_main['secondary']['eta_cut']][0]}, $c_2$: {priorhps[priorhp_main['secondary']['eta_cut']][1]}",
        vertical=True,
        alpha=0.3,
        color='orange',
        )
    

    # Add title
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(rf"Joint SMI posterior for $\theta_1$ and $\theta_2$" +  " \n "+ fr"$\eta=1$: WD $\theta_1$ = {wass['theta0_w1']:.2f}, WD $\theta_2$ = {wass['theta1_w1']:.2f}" +" \n " + fr"$\eta=0$: WD $\theta_1$ = {wass['theta0_w0']:.2f}, WD $\theta_2$ = {wass['theta1_w0']:.2f}", fontsize=10)

    plt.legend(loc='upper right')
    # g.ax_marg_y.legend(loc='lower left')
    sns.move_legend(g.ax_joint, "upper left", fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.5, hspace=0.5)

    return g.fig


def plot_hprange_betapriors(
      priorhps,
      priorhp_main: Optional[Mapping[str, Mapping]] = 
      {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
        'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
                ):
    
    x = jnp.linspace(0,1,50)
    colors = {('main', 'eta_bayes'):['#1f77b4', 1],
              ('main', 'eta_cut'):['orange', 1],
              ('secondary', 'eta_bayes'):['#1f77b4', 0.3],
              ('secondary', 'eta_cut'):['orange', 0.3]}
    fig, ax = plt.subplots()
    fig.suptitle(r'Prior distributions for $\phi_i$')
    for i in itertools.product(('main', 'secondary'), ('eta_bayes', 'eta_cut')):
       c1 = priorhps[priorhp_main[i[0]][i[1]]][0]
       c2 = priorhps[priorhp_main[i[0]][i[1]]][1]

       ax.plot(x, jax.scipy.stats.beta.pdf(x, c1, c2), color=colors[i][0],
                alpha=colors[i][1], 
                label=fr"$c_1$: {priorhps[priorhp_main[i[0]][i[1]]][0]}, $c_2$: {priorhps[priorhp_main[i[0]][i[1]]][1]}",
                )
    ax.set_xlabel(r"$\phi_i$")
    ax.legend()
      
    return fig

def plot_posterior_phi_etarange(
      posterior_sample_dict,
      smi_etas,
      priorhp,
      priorhp_main: Optional[Mapping[str, Mapping]] = 
      {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
        'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
):

    priorhp_k = priorhp[0]
    priorhp_v = priorhp[1]

    # Phis
    fig_phi, ax_phi = plt.subplots(2,2, figsize=(10, 10), sharex=True)
    ax_phi_flattened = ax_phi.flatten()
    for eta_ix, (eta_k, eta_v) in enumerate(smi_etas.items()):

        phi = posterior_sample_dict[eta_k][priorhp_k]['phi']       
        # phi plot
        for phi_ix, phi_no in enumerate([7,8,9,12]):
            sns.kdeplot(phi[:,phi_no], ax=ax_phi_flattened[phi_ix],  alpha=0.8)
            ax_phi_flattened[phi_ix].set_title(fr'$\phi_{{{phi_no+1}}}$', fontsize=15)

            ax_phi_flattened[phi_ix].xaxis.set_tick_params(which='both', labelbottom=True)

    fig_phi.suptitle(f'Phi distributions for priorhp $c_1$: {priorhp_v[0]}, $c_2$: {priorhp_v[1]}', fontsize=20)
    labels = [fr'$\eta_1$: {smi_etas[k]}' for k in smi_etas.keys()]

    fig_phi.legend(labels,
        loc='lower center', ncol=2, fontsize='medium')

    fig_phi.tight_layout()


    fig_phi.subplots_adjust(left=None, bottom=0.15, right=None, top=0.9, wspace=0.2, hspace=0.3)
    return fig_phi


    # Plot the ELPD surface.
def plot_elpd_surface(
    elpd_surface_dict,
    eta_grid,
    eta_grid_x_y_idx,
    xlab,
    ylab,
    suptitle,
):
    fig, axs = plt.subplots(
        nrows=3, ncols=2, figsize=(2 * 3, 3*3)) #, subplot_kw={"projection": "3d"})

    fig.suptitle(suptitle, fontsize=15)
    for mod_ix, mod_name in enumerate(['_y', '_y', '_z']):
        for i, metric in enumerate([elpd_surface_dict[f'lpd{mod_name}_all_eta'],
                                    elpd_surface_dict[f'elpd_waic{mod_name}_all_eta'],]):
            axs[mod_ix,i].set_title([f"Y module \n {['- LPD', '- ELPD WAIC'][i]}", 
                                     f"Logged Y module\n {['- LPD', '- ELPD WAIC'][i]}", 
                                     f"Z module\n {['- LPD', '- ELPD WAIC'][i]}"][mod_ix])
            if mod_ix==1:
                cp = axs[mod_ix,i].contourf(eta_grid[eta_grid_x_y_idx[0]],
                eta_grid[eta_grid_x_y_idx[1]],
                np.log(-metric),
                levels=15, cmap=cmap)
            else:
                cp = axs[mod_ix,i].contourf(eta_grid[eta_grid_x_y_idx[0]],
                eta_grid[eta_grid_x_y_idx[1]],
                -metric,
                levels=15, cmap=cmap)
            # axs[mod_ix,i].plot_surface(
            #     eta_grid[eta_grid_x_y_idx[0]],
            #     eta_grid[eta_grid_x_y_idx[1]],
            #     -metric,
            #     cmap=matplotlib.cm.inferno,
            # )
            # axs[mod_ix,i].view_init(30, 225)
            axs[mod_ix,i].set_xlabel(xlab)
            axs[mod_ix,i].set_ylabel(ylab)
            fig.colorbar(cp, ax=axs[mod_ix,i])


    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.2, hspace=0.6)

    return fig


def lambda_loss_two_stages(
        info_dict,
        step_now,
        ):

        fig, ax = plt.subplots()
        fig.suptitle('Negative training loss')
        n_steps = len(info_dict['lambda_training_loss'])
        ax.plot(np.arange(step_now-n_steps, step_now), 
                np.array(info_dict['elbo_stage1']).mean(axis=1), color='black', alpha=0.5, label='stage 1')
        ax.set_title('stage 1 vs stage 2')
        ax.set_ylabel('stage 1')
        ax.legend()
        ax0=ax.twinx()
        ax0.plot(np.arange(step_now-n_steps, step_now), 
                np.array(info_dict['elbo_stage2']).mean(axis=1), color='red',linestyle='dotted', alpha=1, label='stage 2')
        ax0.set_ylabel('stage 2')
        ax0.legend(loc=4)
        
        return fig

def lambda_loss(
        info_dict,
        step_now,
        loss_low_treshold,
        loss_high_treshold,
        ):

        fig, ax = plt.subplots(3,1, figsize=(5,8))
        fig.suptitle('Lambda training loss', fontsize=20)
        lambda_loss = np.array(info_dict['lambda_training_loss'])
        n_steps = len(lambda_loss)
        ax[0].scatter(np.arange(step_now-n_steps, step_now), 
                lambda_loss, color='black', s=6, alpha=0.5)
        ax[0].set_title(f'ALL, {len(lambda_loss)} data points')
        
        loss_low = lambda_loss[np.where(lambda_loss<loss_low_treshold)[0]]
        steps_low = np.arange(step_now-n_steps, step_now)[np.where(lambda_loss<loss_low_treshold)[0]]
        ax[1].scatter(steps_low, 
                loss_low, color='black', s=6, alpha=0.5)
        ax[1].set_title(f'ONLY LOSS <{loss_low_treshold}, {len(loss_low)} data points')

        loss_high = lambda_loss[np.where(lambda_loss>loss_high_treshold)[0]]
        steps_high = np.arange(step_now-n_steps, step_now)[np.where(lambda_loss>loss_high_treshold)[0]]
        ax[2].scatter(steps_high, 
                loss_high, color='black', s=6, alpha=0.5)
        ax[2].set_title(f'ONLY LOSS >{loss_high_treshold}, {len(loss_high)} data points')

        fig.tight_layout()
        fig.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.4, hspace=0.4)
        return fig


def posterior_sample_variance(
    batch: Batch,
    sample_dict_all: dict,
    etas:Array,
    config: ConfigDict,
) -> None:

    var_phi = jnp.var(sample_dict_all['posterior_sample']['phi'], axis=1) # (n_etas, 13)
    var_theta = jnp.var(sample_dict_all['posterior_sample']['theta'], axis=1) # (n_etas, 2)


    fig_sampl, ax = plt.subplots(1,3, figsize=(15,5))
    fig_sampl.suptitle(f'Variance of {config.num_samples_elpd} posterior samples for a range of eta1, betahps fixed to 1', fontsize=18)

    for dim in range(var_phi.shape[1]):
        ax[0].plot(etas, var_phi[:,dim], label=f'Phi{dim}')
    ax[0].legend()
    ax[0].set_xlabel('eta range', fontsize=15)

    ax[1].plot(etas, var_theta[:,0], label=f'Theta0')
    ax[1].legend()
    ax[1].set_xlabel('eta range', fontsize=15)

    ax[2].plot(etas, var_theta[:,1], label=f'Theta1')
    ax[2].legend()
    ax[2].set_xlabel('eta range', fontsize=15)
    plt.tight_layout()

    fig_poisson, ax = plt.subplots(figsize=(10,10))
    fig_poisson.suptitle(f'Variance of Poisson mean of Y module for {config.num_samples_elpd} posterior samples', fontsize=18)
    for dim in range(var_phi.shape[1]):
        log_incidence = sample_dict_all['posterior_sample']['theta'][:,:,0] + sample_dict_all['posterior_sample']['theta'][:,:,1] * sample_dict_all['posterior_sample']['phi'][:,:,dim]
        mu = batch['T'][dim] * (1. / 1000) * jnp.exp(log_incidence)
        varmu = jnp.var(mu, axis=1)
        ax.plot(etas, varmu, label=f'Phi{dim}')
    ax.legend()
    ax.set_xlabel('eta range', fontsize=15)
    ax.set_ylabel('variance of mu', fontsize=15)
    fig_poisson.tight_layout()

    return fig_sampl, fig_poisson

#############################################################################################################################################
# VMP not trained on eta, eta fixed to some value

def plot_posterior_theta_hprange_single_eta(
      posterior_sample_dfs,
      eta,
      priorhps,
      priorhp_main: Optional[Mapping[str, Mapping]] = 
      {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
        'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
):

    n_samples = posterior_sample_dfs['eta_bayes']['priorhp_converged_bayes'].shape[0]

    pars = {'alpha': np.clip(100 / n_samples, 0., 1.),
            # 'colour': colour,
        }
    df_main = posterior_sample_dfs[eta[0]][priorhp_main['main'][eta[0]]] #0
    warnings.simplefilter(action='ignore', category=FutureWarning)
    grid = sns.JointGrid(
                        x='theta_1',
                        y='theta_2',
                        data=df_main,
                        hue='eta1',
                        xlim=[-3.3, -1], #[-3, 2],
                        ylim=[5, 35], #[-1, 35],
                        height=5)
    g = grid.plot_joint(sns.scatterplot, alpha=pars['alpha'],)
    # g.ax_joint.get_legend().set_title(r'$\eta$ values')
    g.ax_joint.set_xlabel(r"$\theta_1$")
    g.ax_joint.set_ylabel(r"$\theta_2$")

    sns.kdeplot(
        posterior_sample_dfs[eta[0]][priorhp_main['main'][eta[0]]]['theta_1'], #4
        ax=g.ax_marg_x,
        #legend=False,
        )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.kdeplot(
        posterior_sample_dfs[eta[0]][priorhp_main['main'][eta[0]]]['theta_2'], #4
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 1, $c_1$: {priorhps[priorhp_main['main'][eta[0]]][0]}, $c_2$: {priorhps[priorhp_main['main'][eta[0]]][1]} ",
        vertical=True,
        )
    
    sns.kdeplot(
        posterior_sample_dfs[eta[0]][priorhp_main['secondary'][eta[0]]]['theta_1'],
        ax=g.ax_marg_x,
        alpha=0.3,
        color='#1f77b4',
        #legend=False,
        )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.kdeplot(
        posterior_sample_dfs[eta[0]][priorhp_main['secondary'][eta[0]]]['theta_2'],
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 1, $c_1$: {priorhps[priorhp_main['secondary'][eta[0]]][0]}, $c_2$: {priorhps[priorhp_main['secondary'][eta[0]]][1]} ",
        vertical=True,
        alpha=0.3,
        color='#1f77b4',
        )


    # Add title
    g.fig.subplots_adjust(top=0.9)
    # g.fig.suptitle(r'Joint SMI posterior for $\theta_1$ and $\theta_2$', fontsize=13)

    plt.legend(loc='upper right')
    # g.ax_marg_y.legend(loc='lower left')
    sns.move_legend(g.ax_joint, "upper left")
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.5, hspace=0.5)

    return g.fig

def plot_posterior_theta_vsmcmc_single_eta(
      posterior_sample_df_main,
      mcmc_kde,
):

    n_samples = posterior_sample_df_main.shape[0]
    # pars = {'alpha': np.clip(100 / n_samples, 0., 1.),
    #     }

    df_main = posterior_sample_df_main #0
    mcmc1_df = pd.DataFrame(mcmc_kde, columns=['theta_1', 'theta_2'])
    vmp1 = df_main[df_main.eta1=='= 1'][['theta_1', 'theta_2']].values
    
    mcmc_samples_flat1 = np.array(mcmc_kde.reshape(-1, 2))
    VI_samples_flat1 = vmp1.reshape(-1, 2)
    a_mcmc_flat1 = np.ones((mcmc_samples_flat1.shape[0],)) / mcmc_samples_flat1.shape[0]
    b_VI_flat1 = np.ones((VI_samples_flat1.shape[0],)) / VI_samples_flat1.shape[0]
    wass = ot.emd2(a_mcmc_flat1, b_VI_flat1, ot.dist(mcmc_samples_flat1, VI_samples_flat1), numItermax=1000000)
    
    # wass = {}
    # for theta_ix in [0,1]:
    #     w1 = wasserstein_distance(mcmc_kde[:,theta_ix],vmp1[:,theta_ix])
    #     wass[f'theta{theta_ix+1}'] = w1

    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['legend.fontsize'] = 12
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.kdeplot(data=mcmc1_df, x='theta_1', y='theta_2', color='black', ax=ax, linewidth=0.3, alpha=0.7)
    ax.scatter(x=df_main['theta_1'], y=df_main['theta_2'], alpha=0.3, label='VMP', s=5)

    # Manually add a line to the legend for MCMC
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='black', lw=1.5, label='MCMC', alpha=0.7),
                    Line2D([0], [0], marker='o', color='w', label='VMP',
                            markerfacecolor='tab:blue', markersize=10)]
    ax.legend(handles=legend_elements)

    plt.xlim([-3.3, -1])
    plt.ylim([5, 40])
    ax.set_xlabel(r"$\theta_1$", size=13)
    ax.set_ylabel(r"$\theta_2$", size=13)
    # 
    # plt.legend(loc='upper right')
    # g.ax_marg_y.legend(loc='lower left')


    return fig, wass


def plot_posterior_theta_hprange_vsmcmc_SMI(
      posterior_sample_dfs,
      mcmc_df,
      smi_etas,
      priorhps,
      priorhp_main: Optional[Mapping[str, Mapping]] = 
      {'main': {'eta_bayes': 'priorhp_converged_bayes',
                'eta_cut': 'priorhp_converged_cut'},
        'secondary': {'eta_bayes': 'priorhp_alternative_bayes',
                'eta_cut': 'priorhp_alternative_cut'}}
):
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['legend.fontsize'] = 12
    n_samples = posterior_sample_dfs['eta_bayes']['priorhp_converged_bayes'].shape[0]

    pars = {'alpha': np.clip(100 / n_samples, 0., 1.),
            # 'colour': colour,
        }
    df_main = pd.concat([posterior_sample_dfs['eta_bayes'][priorhp_main['main']['eta_bayes']], 
                             posterior_sample_dfs['eta_cut'][priorhp_main['main']['eta_cut']]]) 
    df_second = pd.concat([posterior_sample_dfs['eta_bayes'][priorhp_main['secondary']['eta_bayes']], 
                             posterior_sample_dfs['eta_cut'][priorhp_main['secondary']['eta_cut']]]) 
    
    mcmc1 = mcmc_df[mcmc_df.eta1=='= 0.87'][['theta_1', 'theta_2']].values 
    mcmc0 = mcmc_df[mcmc_df.eta1=='= 0.02'][['theta_1', 'theta_2']].values 
    vmp1 = df_main[df_main.eta1=='= 0.87'][['theta_1', 'theta_2']].values
    vmp0 = df_main[df_main.eta1=='= 0.02'][['theta_1', 'theta_2']].values
    wass = {}
    # for theta_ix in [0,1]:
    #     w1 = wasserstein_distance(mcmc1[:,theta_ix],vmp1[:,theta_ix])
    #     w0 = wasserstein_distance(mcmc0[:,theta_ix],vmp0[:,theta_ix])
    #     wass[f'theta{theta_ix}_w1'] = w1
    #     wass[f'theta{theta_ix}_w0'] = w0


    mcmc_samples_flat1 = mcmc1.reshape(-1, 2)
    VI_samples_flat1 = vmp1.reshape(-1, 2)
    a_mcmc_flat1 = np.ones((mcmc_samples_flat1.shape[0],)) / mcmc_samples_flat1.shape[0]
    b_VI_flat1 = np.ones((VI_samples_flat1.shape[0],)) / VI_samples_flat1.shape[0]
    wass['w1'] = ot.emd2(a_mcmc_flat1, b_VI_flat1, ot.dist(mcmc_samples_flat1, VI_samples_flat1), numItermax=1000000)
    
    mcmc_samples_flat0 = mcmc0.reshape(-1, 2)
    VI_samples_flat0 = vmp0.reshape(-1, 2)
    a_mcmc_flat0 = np.ones((mcmc_samples_flat0.shape[0],)) / mcmc_samples_flat0.shape[0]
    b_VI_flat0 = np.ones((VI_samples_flat0.shape[0],)) / VI_samples_flat0.shape[0] 
    wass['w0'] = ot.emd2(a_mcmc_flat0, b_VI_flat0, ot.dist(mcmc_samples_flat0, VI_samples_flat0), numItermax=1000000)


         
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pars = {'alpha': np.clip(400 / n_samples, 0., 1.),
            # 'colour': colour,
        }
    grid = sns.JointGrid(
                        x='theta_1',
                        y='theta_2',
                        data=df_main,
                        hue='eta1',
                        xlim=[-3.3, -1], #[-3, 2],
                        ylim=[5, 35], #[-1, 35],
                        height=5)
    sns.kdeplot(data=mcmc_df, x='theta_1', y='theta_2', hue='eta1',
                palette=['black','black'], ax=grid.ax_joint, linewidth=0.3, alpha=0.4)
    g = grid.plot_joint(sns.scatterplot, alpha=0.3, s=10)
    # g.ax_joint.get_legend().set_title(r'$\eta$ values')
    g.ax_joint.set_xlabel(r"$\theta_1$")
    g.ax_joint.set_ylabel(r"$\theta_2$")
    
    sns.kdeplot(
        df_main[df_main.eta1=='= 0.87']['theta_1'], #4
        ax=g.ax_marg_x,
        #legend=False,
        )
    sns.kdeplot(
        df_main[df_main.eta1=='= 0.87']['theta_2'], #4
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 0.87, $c_1$: {priorhps[priorhp_main['main']['eta_bayes']][0]}, $c_2$: {priorhps[priorhp_main['main']['eta_bayes']][1]} ",
        vertical=True,
        )
    
    
    sns.kdeplot(
        df_main[df_main.eta1=='= 0.02']['theta_1'], #0
        ax=g.ax_marg_x,
        #legend=False,
        )
    sns.kdeplot(
        df_main[df_main.eta1=='= 0.02']['theta_2'], #0
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 0.02, $c_1$: {priorhps[priorhp_main['main']['eta_cut']][0]}, $c_2$: {priorhps[priorhp_main['main']['eta_cut']][1]} ",
        vertical=True,
        )
    
    sns.kdeplot(
        df_second[df_second.eta1=='= 0.87']['theta_1'],
        ax=g.ax_marg_x,
        alpha=0.3,
        color='#1f77b4',
        #legend=False,
        )
    sns.kdeplot(
        df_second[df_second.eta1=='= 0.87']['theta_2'],
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 0.87, $c_1$: {priorhps[priorhp_main['secondary']['eta_bayes']][0]}, $c_2$: {priorhps[priorhp_main['secondary']['eta_bayes']][1]} ",
        vertical=True,
        alpha=0.3,
        color='#1f77b4',
        )
            
    sns.kdeplot(
        df_second[df_second.eta1=='= 0.02']['theta_1'],
        ax=g.ax_marg_x,
        alpha=0.3,
        color='orange',
        #legend=False,
        )
    sns.kdeplot(
        df_second[df_second.eta1=='= 0.02']['theta_2'],
        ax=g.ax_marg_y,
        label=fr"$\eta_1$: 0.02, $c_1$: {priorhps[priorhp_main['secondary']['eta_cut']][0]}, $c_2$: {priorhps[priorhp_main['secondary']['eta_cut']][1]} ",
        vertical=True,
        alpha=0.3,
        color='orange',
        )
    
    # Add title
    g.fig.subplots_adjust(top=0.9)
    # g.fig.suptitle(r'Joint SMI posterior for $\theta_1$ and $\theta_2$', fontsize=13)
    hue_legend = g.ax_joint.get_legend()
    if hue_legend:
        hue_legend.set_title(r'$\eta_1$')
    plt.legend(loc='upper right', bbox_to_anchor=(0,-0.2), ncols=2, fontsize=9)

    # g.ax_marg_y.legend(loc='lower left')
    sns.move_legend(g.ax_joint, "upper right")
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.3, right=None, top=0.99, wspace=0.5, hspace=0.5)


    return g.fig, wass

def plot_optim_hparams_vs_true(path: str,
                        init_names: list,
                        optimiser_name: str,
                        loglik_types: list,
                        loss_type: str,
                        hp_names: list,
                            ):
    names_latex = {'eta': '$\eta$',
                'c1':'$c_1$',
                'c2':'$c_2$',}
    hp_names = np.array(hp_names)
    if 'eta' in hp_names:
        for loglik_type in loglik_types:
            eta_index = np.where(hp_names == 'eta')[0][0]
            indices = np.arange(len(hp_names))
            indices = np.delete(indices, eta_index)
            rolled_indices = np.insert(indices, 1, eta_index)


            n_plots = len(hp_names) + 1
            fig, ax = plt.subplots(int(n_plots/2)+int(n_plots%2>0), 2, 
                                figsize=(7,3.5*(int(n_plots/2)+int(n_plots%2>0))))
            # with open(workdir + f"/hp_info_{'eta' if config.estimate_smi else 'only'}c1c2_{config.tune_hparams}_{all_init_name}_{info_dict['likelihood']}_{optimiser_name}.sav", 'wb') as f:

            for init_type in init_names:
                with open(path + f'/hp_info_etac1c2_{loss_type}_{init_type}_{loglik_type}_{optimiser_name}.sav', 'rb') as fr:
                    res = pickle.load(fr)
                for a_ix, a in enumerate(ax.flatten()):
                    if a_ix==0:
                        a.plot(jnp.array(res['loss']), alpha=0.7)
                        a.set_xlabel('Iterations')
                        a.set_title('Training loss')
                    elif a_ix < (n_plots):  
                        a.plot(jnp.array(res['params'])[:,rolled_indices][:,a_ix-1])
                        hp_name = np.array(res['hp_names'])[rolled_indices][a_ix-1]
                        a.set_title('Trace plot for '+ names_latex[hp_name])
                        a.set_xlabel('Iterations')
            for a_ix, a in enumerate(ax.flatten()):
                if (a_ix -1)==1:
                    a.set_ylim(bottom=-0.1, top=1.1)
                elif a_ix!=0:
                    a.set_ylim(bottom=-0.1)

            plt.tight_layout()
            plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
            plt.savefig(path + f'/hp_tuning_all_hparams_{loss_type}_{loglik_type}_{optimiser_name}.png')
    else:
        for loglik_type in loglik_types:
            n_plots = len(hp_names)+2
            fig, ax = plt.subplots(int(n_plots/2)+int(n_plots%2>0), 2, 
                                figsize=(7,3.5*(int(n_plots/2)+int(n_plots%2>0))))

            for init_type in init_names:
                with open(path + f'/hp_info_onlyc1c2_{loss_type}_{init_type}_{loglik_type}_{optimiser_name}.sav', 'rb') as fr:
                    res = pickle.load(fr)
                for a_ix, a in enumerate(ax.flatten()):
                    if a_ix==0:
                        a.plot(jnp.array(res['loss']), alpha=0.7)
                        a.set_xlabel('Iterations')
                        a.set_title('Training loss')
                        # if loglik_type=='y':
                        #     # a.set_ylim(top=65)
                        # a.set_ylim(0.25, 0.26)
                    elif a_ix==2:
                        a.set_axis_off()
                    elif a_ix>2:
                        a.plot(jnp.array(res['params'])[:,a_ix-2])
                        hp_name = np.array(res['hp_names'])[a_ix-2]
                        a.set_title('Trace plot for '+ names_latex[hp_name])
                        a.set_xlabel('Iterations')

                    else:  
                        a.plot(jnp.array(res['params'])[:,a_ix-1])
                        hp_name = np.array(res['hp_names'])[a_ix-1]
                        a.set_title('Trace plot for '+ names_latex[hp_name])
                        a.set_xlabel('Iterations')
            for a_ix, a in enumerate(ax.flatten()):
                if ((a_ix!=0) and (a_ix!=2)):
                    a.set_ylim(bottom=-0.1)
            plt.tight_layout()
            plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
            plt.savefig(path + f'/hp_tuning_prior_hparams_{loss_type}_{loglik_type}_{optimiser_name}.png')


    