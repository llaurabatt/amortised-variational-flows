#%%
import pickle
import matplotlib.pyplot as plt
# import jax.numpy as jnp
import numpy as np
import itertools
import matplotlib as mpl
from absl import flags
import sys
#%%

FLAGS = flags.FLAGS
flags.DEFINE_string('path', None, 'Path to hyperparameter optimisation results.')
flags.mark_flags_as_required(['path'])
FLAGS(sys.argv)
#%%
# path = '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow'

# eta_vals = [0.000001, 0.001000, 0.300000, 0.500000, 0.700000, 1.000000]


#########################################################################################################################################################
#%%
path = FLAGS.path
init_names = ['default', 'mixed','low', 'high']
optimisers = [ 'elbo_opt', 'plain_lr1', 'plain_lr2']
init_eta_vals = [1.00, 0.50, 0.00]
#%%
with open(path + f'/hp_info_etapriorhps_{init_names[0]}_{optimisers[0]}_new.sav', 'rb') as fr:
    res = pickle.load(fr)
hp_names = res['hp_names'].copy()

#########################################################################################################################################################
#%%|
# chosen optimum: default sigma_a=5.5 sigma_w=11 sigma_k=0.4 ell_k=0.2
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['text.usetex'] = True

names_latex = {'eta': '$\eta$',
               'w_prior_scale':'$\sigma_w$',
               'a_prior_scale':'$\sigma_a$',
               'kernel_amplitude':'$\sigma_k$',
               'kernel_length_scale':'$\ell_k$'}
hp_names = np.array(hp_names)
eta_index = np.where(hp_names == 'eta')[0][0]
indices = np.arange(len(hp_names))
indices = np.delete(indices, eta_index)
rolled_indices = np.insert(indices, 2, eta_index)
colors = ['purple', 'orange', 'green', 'red', 'blue']

n_plots = len(hp_names) + 1
for optimiser_name in optimisers:
    fig, ax = plt.subplots(int(n_plots/3)+int(n_plots%3>0), 3, figsize=(10,3.5*(int(n_plots/3)+int(n_plots%3>0))))
    last_losses = []
    for init_ix, init_type in enumerate(init_names):
        with open(path + f'/hp_info_etapriorhps_{init_type}_{optimiser_name}_new.sav', 'rb') as fr:
            res = pickle.load(fr)
        last_loss = np.array(res['loss'])[-20:].mean()
        last_losses.append(last_loss)
    best_init_ix = np.argmin(last_losses) 
    for init_ix, init_type in enumerate(init_names):
        with open(path + f'/hp_info_etapriorhps_{init_type}_{optimiser_name}_new.sav', 'rb') as fr:
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
                a.plot(np.array(res['loss'])[:4001], alpha=alpha, color=color, 
                           label=f'Init {init_ix + 1}', linestyle=linestyle)
                a.set_xlabel('Iterations')
                a.set_title('Posterior Mean Squared Error')
            elif a_ix < (n_plots):  
                a.plot(np.array(res['params'])[:4001,rolled_indices][:,a_ix-1], alpha=alpha, 
                           color=color, label=f'Init {init_ix + 1}', linestyle=linestyle)
                hp_name = np.array(res['hp_names'])[rolled_indices][a_ix-1]
                a.set_title('Trace for '+ names_latex[hp_name])
                a.set_xlabel('Iterations')
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=len(init_names))

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
    plt.savefig(path + f'/hp_tuning_all_hparams_{optimiser_name}_4000.png')
    plt.show()

# #%%
# #########################################################################################################################################################
# # chosen optimum: default sigma_a=0.2 or 1 sigma_w=0.2 or 3 sigma_k=0.1 ell_k=0.5

# path = '/home/llaurabat/spatial-smi-output-integrated-allhps-NOETA-40val-smallcondval/all_items/nsf/vmp_flow'

# init_names = ['default', 'low', 'high']

# with open(path + f'/hp_info_{init_names[0]}.sav', 'rb') as fr:
#     res = pickle.load(fr)
# hp_names = res['hp_names'].copy()
# names_latex = {'eta': '$\eta$',
#                'w_prior_scale':'$\sigma_w$',
#                'a_prior_scale':'$\sigma_a$',
#                'kernel_amplitude':'$\sigma_k$',
#                'kernel_length_scale':'$\ell_k$'}
# hp_names = np.array(hp_names)
# n_plots = len(hp_names)+2
# fig, ax = plt.subplots(int(n_plots/3)+int(n_plots%3>0), 3, figsize=(10,3.5*(int(n_plots/3)+int(n_plots%3>0))))

# for init_type in init_names:
#     with open(path + f'/hp_info_{init_type}.sav', 'rb') as fr:
#         res = pickle.load(fr)
#     for a_ix, a in enumerate(ax.flatten()):
#         if a_ix==0:
#             a.plot(jnp.array(res['loss']), alpha=0.7)
#             a.set_xlabel('Iterations')
#             a.set_title('Training loss')
#             a.set_ylim(0.25, 0.26)
#         elif a_ix==3:
#             a.set_axis_off()
#         elif a_ix>3:
#             a.plot(jnp.array(res['params'])[:,a_ix-2])
#             hp_name = np.array(res['hp_names'])[a_ix-2]
#             a.set_title('Trace plot for '+ names_latex[hp_name])
#             a.set_xlabel('Iterations')
#         else:  
#             a.plot(jnp.array(res['params'])[:,a_ix-1])
#             hp_name = np.array(res['hp_names'])[a_ix-1]
#             a.set_title('Trace plot for '+ names_latex[hp_name])
#             a.set_xlabel('Iterations')
# plt.tight_layout()
# plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
# plt.savefig(path + '/hp_tuning_prior_hparams__.png')
# plt.show()

#%%
