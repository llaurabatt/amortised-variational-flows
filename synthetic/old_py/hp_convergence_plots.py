#%%
import pickle
import matplotlib.pyplot as plt
import jax.numpy as jnp
import itertools
#%%
path = '/home/llaurabat/modularbayes-output/epidemiology_new/nsf/vmp_flow_all'
eta_init = 1.000000
with open(path + f'/hp_info_eta{eta_init:.6f}_z.sav', 'rb') as fr:
    res_z = pickle.load(fr)
res_z_dict = {'training loss':jnp.array(res_z['loss']), 'eta':jnp.array(res_z['params'])[:,1], 
'c1':jnp.array(res_z['params'])[:,2], 'c2':jnp.array(res_z['params'])[:,3]}
#%%
fig, ax = plt.subplots(2, 2, figsize=(10,10))
fig.suptitle('Hyperparameter tuning: ELPD optimisation for the Z module', fontsize=20)
ax_flattened = ax.flatten()
for ix, (k, v) in enumerate(res_z_dict.items()):
    ax_flattened[ix].plot(v)
    ax_flattened[ix].set_title(f'{k}', fontsize=12)
    if ix!=0:
        ax_flattened[ix].set_ylim(bottom=-0.1, top=1.25)
    ax_flattened[ix].set_xlabel('iterations')

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.2, hspace=0.3)
plt.savefig(path + f'/hp_tuning_z_eta{eta_init}.png')
plt.show()
#%%
with open(path + f'/hp_info_eta{eta_init:.6f}_y.sav', 'rb') as fr:
    res_y = pickle.load(fr)
res_y_dict = {'training loss':jnp.array(res_y['loss']), 'eta':jnp.array(res_y['params'])[:,1], 
'c1':jnp.array(res_y['params'])[:,2], 'c2':jnp.array(res_y['params'])[:,3]}
#%%
fig, ax = plt.subplots(2, 2, figsize=(10,10))
fig.suptitle('Hyperparameter tuning: ELPD optimisation for the Y module', fontsize=20)
ax_flattened = ax.flatten()
for ix, (k, v) in enumerate(res_y_dict.items()):
    ax_flattened[ix].plot(v)
    ax_flattened[ix].set_title(f'{k}', fontsize=12)
    if ix!=0:
        ax_flattened[ix].set_ylim(bottom=-0.1, top=2)
    ax_flattened[ix].set_xlabel('iterations')

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.2, hspace=0.3)
plt.savefig(path + f'/hp_tuning_y_eta{eta_init}.png')
plt.show()
#%%

