#%%
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax.numpy as jnp
import numpy as np
import itertools
from matplotlib.ticker import FormatStrFormatter
from absl import flags
import sys
#%%
FLAGS = flags.FLAGS
flags.DEFINE_string('path_results', None, 'Path to hyperparameter optimisation results.')
flags.mark_flags_as_required(['path_results'])
FLAGS(sys.argv)

#########################################################################################################################################################
#%%
path = FLAGS.path_results
tune_hparams = 'elbo'
loglik_types = ['z','y'] 
optimisers = ['elbo_opt', 'plain_lr1', 'plain_lr2']
init_names = ['true', 'high', 'medium', 'small']
true_vals = [0., 1., 1.5, 0.5]
true_vals = {'mu_prior_mean_m': 0.,
                'mu_prior_scale_s': 1.,
                'sigma_prior_concentration': 1.5,
                'sigma_prior_scale': 0.5,}
names_latex = {'mu_prior_mean_m': '$m$',
                'mu_prior_scale_s':'$s$',
                'sigma_prior_concentration':'$g_1$',
                'sigma_prior_scale':'$g_2$',}

#########################################################################################################################################################
#%%

#%%
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['text.usetex'] = True
mpl.rcParams['legend.fontsize'] = 15

for optimiser in optimisers:
    print(f"optimiser {optimiser}")
    colors = ['orange', 'green', 'purple', 'blue']
    with open(path + f'/hp_info_allhps_{init_names[0]}_{optimiser}_{tune_hparams}.sav', 'rb') as fr:
        res = pickle.load(fr)
    hp_names = res['hp_names'].copy()
    hp_names = np.array(hp_names)

    n_plots = len(hp_names) + 1
    fig, ax = plt.subplots(int(n_plots/2)+int(n_plots%2>0), 2, 
                        figsize=(8,3.5*(int(n_plots/2)+int(n_plots%2>0))),
                        )
    last_losses = []
    for init_ix, init_type in enumerate(init_names):
        with open(path + f'/hp_info_allhps_{init_type}_{optimiser}_{tune_hparams}.sav', 'rb') as fr:
            res = pickle.load(fr)
        last_loss = jnp.array(res['loss'])[:1000][-20:].mean()
        last_losses.append(last_loss)
    best_init_ix = np.argmin(last_losses)
    for init_ix, init_type in enumerate(init_names):
        with open(path + f'/hp_info_allhps_{init_type}_{optimiser}_{tune_hparams}.sav', 'rb') as fr:
            res = pickle.load(fr)
        for a_ix, a in enumerate(ax.flatten()):
            color = colors[init_ix]
            if init_ix!=best_init_ix:
                alpha = 0.3
                linestyle = 'solid'
            else:
                alpha = 1.
                linestyle = 'dashed'
                color = 'black'
            a.grid(True, linestyle='--', alpha=0.7)
            if a_ix==0:
                a.plot(jnp.array(res['loss'][:1000]), alpha=alpha, color=color, 
                        label=f'Init {init_ix + 1}', linestyle=linestyle)
                a.set_xlabel('Iterations')
                a.set_title('Negative ELBO')
            elif a_ix==1:
                a.set_axis_off()
            elif a_ix <= (n_plots):  
                a.axhline(true_vals[hp_names[a_ix-2]], color='red', linestyle=':', label='True')
                a.plot(jnp.array(res['params'])[:1000,:][:,a_ix-2], alpha=alpha, 
                        color=color, label=f'Init {init_ix + 1}', linestyle=linestyle)
                hp_name = np.array(res['hp_names'])[a_ix-2]
                a.set_title('Trace for '+ names_latex[hp_name])
                a.set_xlabel('Iterations')
            a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # for a_ix, a in enumerate(ax.flatten()):
    #     if (a_ix -1)==1:
    #         a.set_ylim(bottom=-0.1, top=1.1)
    #     elif a_ix!=0:
    #         a.set_ylim(bottom=-0.1)
    handles, labels = ax[1,0].get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels.values():
            unique_labels[handle] = label
    handles, labels = zip(*unique_labels.items())
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=len(init_names)+1)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.93, wspace=0.2, hspace=0.4)
    plt.savefig(path + f'/hp_tuning_all_hparams_{tune_hparams}_{optimiser}_last.png')
    plt.show()

#%%
###################### converged points ############################################################################################################
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['text.usetex'] = True
mpl.rcParams['legend.fontsize'] = 15

colors = ['orange', 'green', 'purple', 'blue']
for optimiser in optimisers:
    print(f"optimiser {optimiser}")
    with open(path + f'/hp_info_allhps_{init_names[0]}_{optimiser}_{tune_hparams}.sav', 'rb') as fr:
        res = pickle.load(fr)
    hp_names = res['hp_names'].copy()
    hp_names = np.array(hp_names)

    n_plots = len(hp_names)
    fig, ax = plt.subplots(1, n_plots,
                        figsize=(3.5*n_plots,3.5), sharey=True)

    
    last_losses = []
    for init_ix, init_type in enumerate(init_names):
        with open(path + f'/hp_info_allhps_{init_type}_{optimiser}_{tune_hparams}.sav', 'rb') as fr:
            res = pickle.load(fr)
        last_loss = jnp.array(res['loss'])[-20:].mean()
        last_losses.append(last_loss)
    best_init_ix = np.argmin(last_losses)

    for init_ix, init_type in enumerate(init_names):
        with open(path + f'/hp_info_allhps_{init_type}_{optimiser}_{tune_hparams}.sav', 'rb') as fr:
            res = pickle.load(fr)
        loss = jnp.array(res['loss'])
        if init_ix==best_init_ix:
            print(f"20-last avg best {hp_name}: {jnp.array(res['params'])[-20:,:].mean(0)}")

        for a_ix, a in enumerate(ax.flatten()):
            hp_name = np.array(res['hp_names'])[a_ix]
            a.axhline(true_vals[hp_names[a_ix]], color='red', linestyle=':', label='True')
            a.grid(True, linestyle='--', alpha=0.7)
            color = colors[init_ix]
            if init_ix!=best_init_ix:
                alpha = 0.3
                s = 20.
            else:
                alpha = 1
                s = 60
                color = 'black'


            # if ((hp_name == 'eta')&(loglik_type=='y')):
            #     a.set_ylim(0.85, 0.9)

            # if ((hp_name == 'c1')&(loglik_type=='z')&(path_name=='prior')):
            #     a.set_ylim(1.7, 1.73)
                
            # if ((loglik_type=='z')&(path_name=='prior')):
            #     a.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            # else:
            #     a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # if ((hp_name == 'c2')&((loglik_type=='z')|(loglik_type=='y'))&(path_name=='prior'))|((hp_name == 'c2')&(loglik_type=='y')&(path_name=='all')):
            #     a.set_ylim(14.8, 15.2)
            # elif (hp_name == 'c2'):
            #     a.set_ylim(13.5, 15.2)


            a.set_title(names_latex[hp_name])
            a.scatter(loss[-20:].mean(), jnp.array(res['params'])[:,a_ix][-20:].mean(), 
                        label=f'Init {init_ix + 1}', alpha=alpha, s=s, color=color)
            a.set_xlabel('Negative ELBO')
            a.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    handles, labels = ax[0].get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels.values():
            unique_labels[handle] = label            
    handles, labels = zip(*unique_labels.items())
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=len(init_names)+1)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.3, right=None, top=0.93, wspace=0.3, hspace=0.4)
    plt.savefig(path + f'/hp_converged_hparams_{tune_hparams}_{optimiser}.png')
    plt.show()


#%%

