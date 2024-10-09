#%%
import pickle
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import itertools
#%%
path = '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow'

# eta_vals = [0.000001, 0.001000, 0.300000, 0.500000, 0.700000, 1.000000]


#########################################################################################################################################################
#%%
init_names = ['default', 'low', 'high']
init_eta_vals = [1.00, 0.50, 0.00]
#%%

with open(path + f'/hp_info_{init_names[0]}.sav', 'rb') as fr:
    res = pickle.load(fr)
hp_names = res['hp_names'].copy()

#########################################################################################################################################################
#%%
# chosen optimum: default sigma_a=5.5 sigma_w=11 sigma_k=0.4 ell_k=0.2
path = '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow'

init_names = ['default', 'low', 'high']
with open(path + f'/hp_info_{init_names[0]}.sav', 'rb') as fr:
    res = pickle.load(fr)
hp_names = res['hp_names'].copy()
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


n_plots = len(hp_names) + 1
fig, ax = plt.subplots(int(n_plots/3)+int(n_plots%3>0), 3, figsize=(10,3.5*(int(n_plots/3)+int(n_plots%3>0))))

for init_type in init_names:
    with open(path + f'/hp_info_{init_type}.sav', 'rb') as fr:
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
plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
plt.savefig(path + '/hp_tuning_all_hparams_new.png')
plt.show()

#%%
#########################################################################################################################################################
# chosen optimum: default sigma_a=0.2 or 1 sigma_w=0.2 or 3 sigma_k=0.1 ell_k=0.5

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
# plt.savefig(path + '/hp_tuning_prior_hparams_new.png')
# plt.show()

#%%
