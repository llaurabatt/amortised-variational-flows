import pickle
import matplotlib.pyplot as plt
import jax.numpy as jnp
import itertools

path = '/home/llaurabat/modularbayes-output/epidemiology_new/nsf/vmp_flow_all'
lik_types = ['full', 'y', 'z']
eta_vals = [0.000001, 0.001000, 0.300000, 0.500000, 0.700000, 1.000000]


fig, ax = plt.subplots(6,3, figsize=(10,15))
fig.suptitle('Training loss', fontsize=20)
for ll, eta in itertools.product(lik_types, eta_vals):
    with open(path + f'/hp_info_eta{eta:.6f}_{ll}.sav', 'rb') as fr:
        res = pickle.load(fr)
    ax[eta_vals.index(eta), lik_types.index(ll)].plot(jnp.array(res['loss']))
    ax[eta_vals.index(eta), lik_types.index(ll)].set_title(f'Likelihood {ll}; eta init {eta}')

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.93, wspace=0.1, hspace=0.4)
plt.savefig(path + '/hp_tuning_loss.png')
plt.show()

fig, ax = plt.subplots(6,3, figsize=(10,15))
fig.suptitle('Eta 1', fontsize=20)
for ll, eta in itertools.product(lik_types, eta_vals):
    with open(path + f'/hp_info_eta{eta:.6f}_{ll}.sav', 'rb') as fr:
        res = pickle.load(fr)
    ax[eta_vals.index(eta), lik_types.index(ll)].plot(jnp.array(res['params'])[:,1])
    ax[eta_vals.index(eta), lik_types.index(ll)].set_title(f"Likelihood {ll}; eta init {eta}")

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.93, wspace=0.1, hspace=0.4)
plt.savefig(path + '/hp_tuning_eta1.png')
plt.show()

fig, ax = plt.subplots(6,3, figsize=(10,15))
fig.suptitle('Conc 1', fontsize=20)
for ll, eta in itertools.product(lik_types, eta_vals):
    with open(path + f'/hp_info_eta{eta:.6f}_{ll}.sav', 'rb') as fr:
        res = pickle.load(fr)
    ax[eta_vals.index(eta), lik_types.index(ll)].plot(jnp.array(res['params'])[:,2])
    ax[eta_vals.index(eta), lik_types.index(ll)].set_title(f'Likelihood {ll}; eta init {eta}')

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.93, wspace=0.1, hspace=0.4)
plt.savefig(path + '/hp_tuning_conc1.png')
plt.show()

fig, ax = plt.subplots(6,3, figsize=(10,15))
fig.suptitle('Conc 2', fontsize=20)
for ll, eta in itertools.product(lik_types, eta_vals):
    with open(path + f'/hp_info_eta{eta:.6f}_{ll}.sav', 'rb') as fr:
        res = pickle.load(fr)
    ax[eta_vals.index(eta), lik_types.index(ll)].plot(jnp.array(res['params'])[:,3])
    ax[eta_vals.index(eta), lik_types.index(ll)].set_title(f'Likelihood {ll}; eta init {eta}')

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.93, wspace=0.1, hspace=0.4)
plt.savefig(path + '/hp_tuning_conc2.png')
plt.show()