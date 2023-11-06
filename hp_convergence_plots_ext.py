import pickle
import matplotlib.pyplot as plt
import jax.numpy as jnp
import itertools

path = '/home/llaurabat/spatial-smi-output-integrated-allhps-mse-randomanchors/all_items/nsf/vmp_flow'
lik_types = ['full', 'y', 'z']
eta_vals = [0.000001, 0.001000, 0.300000, 0.500000, 0.700000, 1.000000]


fig, ax = plt.subplots(figsize=(10,15))
fig.suptitle('Training loss', fontsize=20)

with open(path + f"/hp_info_priordefaults_eta1.000000.sav", 'rb') as fr:
    res = pickle.load(fr)
ax.plot(jnp.array(res['loss']))
plt.savefig(path + '/hp_tuning_loss.png')
plt.show()


fig, ax = plt.subplots(figsize=(10,15))
fig.suptitle('Eta', fontsize=20)
ax.plot(jnp.array(res['params'])[:,-1])
plt.savefig(path + '/hp_tuning_eta1.png')
plt.show()

fig, ax = plt.subplots( figsize=(10,15))
fig.suptitle('Kernel lengthscale', fontsize=20)
ax.plot(jnp.array(res['params'])[:,-2])
plt.savefig(path + '/hp_tuning_kernel_ls.png')
plt.show()

fig, ax = plt.subplots( figsize=(10,15))
fig.suptitle('Kernel amplitude', fontsize=20)
ax.plot(jnp.array(res['params'])[:,-3])
plt.savefig(path + '/hp_tuning_kernel_ampl.png')
plt.show()