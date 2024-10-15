"""Probability functions for the Epidemiology model."""
#%%
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from jax.random import PRNGKey
import pandas as pd
from IPython.display import display, HTML
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

tfd = tfp.distributions

#%%
# draw counts


def gen_data(n_groups: int, 
             n_obs: int,
             SEED: int,
             data_filename: str = "synthetic_data.csv",
             true_params_filename: str = "true_params.csv",
             save_sim_data_filename: str = "sim_data_summary.png",
             ):
    #%%

    hparams = {'mu_prior_mean_m':0., 'mu_prior_scale_s':1, 'sigma_prior_concentration':1.5, 'sigma_prior_scale':0.5}
    # hparams = {'mu_prior_mean_m':0., 'mu_prior_scale_s':1, 'sigma_prior_concentration':3, 'sigma_prior_scale':1.5}
    keys = jax.random.split(PRNGKey(SEED), len(hparams))
    #%%

    mu_g = tfd.Normal(loc=hparams['mu_prior_mean_m'], 
                        scale=hparams['mu_prior_scale_s']).sample(
        sample_shape=(n_groups,), seed=keys[0])
    sigma_g = tfd.InverseGamma(concentration=hparams['sigma_prior_concentration'], 
                        scale=hparams['sigma_prior_scale']).sample(
        sample_shape=(n_groups,), seed=keys[0])

    Y_g_i = tfd.Normal(loc=mu_g, 
                        scale=sigma_g).sample(
        sample_shape=(n_obs,), seed=keys[1])
    

    #%%
    df = pd.DataFrame(Y_g_i)
    df.columns =  [f'Y_{g}' for g in range(n_groups)]
    df.to_csv(data_filename, index=False)
    hparams.update({'mu':mu_g, 'sigma':sigma_g})


    mpl.rcParams['font.size'] = 15
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['legend.fontsize'] = 15

    # summary plot
    if n_groups==10:
        figwidth = 0.7*n_groups
    elif n_groups==50:
        figwidth = 0.3*n_groups
# 
    fig, ax = plt.subplots(2,1,  figsize=(figwidth, 4), gridspec_kw={'height_ratios': [3, 1]})

    x_labs = jnp.arange(n_groups)+1

    ax[0].scatter(jnp.broadcast_to(x_labs,(n_obs, n_groups)),Y_g_i, alpha=0.3) #, label=r'$Y_{ij}$')
    ax[0].set_xticks(x_labs)
    ax[0].scatter(x_labs, mu_g, color='red', marker='+', label=r'$\mu_j$')
    ax[0].set_ylabel(r'$Y_{ij}$', fontsize=14)
    if n_groups==50:
        ax[0].set_xlim(0.4, 50.6)
    ax[0].legend(loc='lower left')

    tab = ax[1].table(cellText=[[ "{:0.1f}".format(v) for v in sigma_g]],loc='center')
    for key, cell in tab.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_alpha(0.3)  # Set the transparency (0.0 is fully transparent, 1.0 is fully opaque)

    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    # ax[1].axis('tight')
    # ax[1].axis('off')
    ax[1].set_ylabel(r'$\sigma_j$')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    # ax[1].set_xlabel('n. groups')

    plt.tight_layout()
    fig.savefig(save_sim_data_filename)

    with open(true_params_filename, 'wb') as f:
        pickle.dump(hparams, f)

    return fig

#%%
