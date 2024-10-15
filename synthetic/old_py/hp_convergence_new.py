#%%
import pickle
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import itertools
#%%


#########################################################################################################################################################
#%%
loglik_types = ['z', 'y']
init_names = ['default','medium','small', 'high']
init_eta_vals = [1.00, 0.50, 0.00]
names_latex = {'eta': '$\eta$',
                'c1':'$c_1$',
                'c2':'$c_2$',}

true_vals = {'eta': 1,
                'c1':1,
                'c2':1}

#########################################################################################################################################################
#%%
path='/home/llaurabat/c1c2_1-syntheticdata-30-modularbayes-output-integrated-etac1c2-smallcondval-gamma_a2_scale0.5/synthetic/nsf/vmp_flow'
# Chosen opt: for y: medium at pos. 10000 init 3: eta[1.0], [1.6, 4.5] 2,3
# Chosen opt: for z: medium at pos. 10000 init 3: eta[0.0], [0.1, 4.5] 0,3

for loglik_type in loglik_types:
    print(f"loglik type: {loglik_type}")
    with open(path + f'/hp_info_etac1c2_{init_names[0]}_{loglik_type}.sav', 'rb') as fr:
        res = pickle.load(fr)
    hp_names = res['hp_names'].copy()
    hp_names = np.array(hp_names)
    eta_index = np.where(hp_names == 'eta')[0][0]
    indices = np.arange(len(hp_names))
    indices = np.delete(indices, eta_index)
    rolled_indices = np.insert(indices, 1, eta_index)


    n_plots = len(hp_names) + 1
    fig, ax = plt.subplots(int(n_plots/2)+int(n_plots%2>0), 2, 
                           figsize=(7,3.5*(int(n_plots/2)+int(n_plots%2>0))))

    for init_type in init_names:
        with open(path + f'/hp_info_etac1c2_{init_type}_{loglik_type}.sav', 'rb') as fr:
            res = pickle.load(fr)
        for a_ix, a in enumerate(ax.flatten()):
            if a_ix==0:
                a.plot(jnp.array(res['loss']), alpha=0.7)
                a.set_xlabel('Iterations')
                a.set_title('Training loss')
                # if loglik_type=='y':
                #     # a.set_ylim(top=175)
            elif a_ix < (n_plots):  
                a.plot(jnp.array(res['params'])[:,rolled_indices][:,a_ix-1])
                hp_name = np.array(res['hp_names'])[rolled_indices][a_ix-1]
                a.set_title('Trace plot for '+ names_latex[hp_name])
                a.set_xlabel('Iterations')
                if (a_ix-1)==1:
                    a.set_ylim(bottom=0.0, top=1.1)
                a.axhline(true_vals[hp_name], color='black')
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
    plt.savefig(path + f'/synth_hp_tuning_all_hparams_{loglik_type}.png')
    plt.show()

#%%
#########################################################################################################################################################
path = '/home/llaurabat/c1c2_1-syntheticdata-30-modularbayes-output-integrated-onlyc1c2-smallcondval-gamma_a2_scale0.5/synthetic/nsf/vmp_flow'

# Chosen opt: for y (that will want bayes): medium at pos. 10000 init 3: [0.1, 0.1] (old: default at pos. 8000: [0., 1.1])
# Chosen opt: for z (that will want cut): medium at pos. 10000 init 3: [2.8, 1.2](old: default at pos. 8000: [2.8, 3.2]

for loglik_type in loglik_types:
    print(f"loglik type: {loglik_type}")
    with open(path + f'/hp_info_etac1c2_{init_names[0]}_{loglik_type}.sav', 'rb') as fr:
        res = pickle.load(fr)
    hp_names = res['hp_names'].copy()
    hp_names = np.array(hp_names)
    n_plots = len(hp_names)+2
    fig, ax = plt.subplots(int(n_plots/2)+int(n_plots%2>0), 2, 
                           figsize=(7,3.5*(int(n_plots/2)+int(n_plots%2>0))))

    for init_type in init_names:
        with open(path + f'/hp_info_etac1c2_{init_type}_{loglik_type}.sav', 'rb') as fr:
            res = pickle.load(fr)
        for a_ix, a in enumerate(ax.flatten()):
            if a_ix==0:
                a.plot(jnp.array(res['loss']), alpha=0.7)
                a.set_xlabel('Iterations')
                a.set_title('Training loss')
                # if loglik_type=='z':
                #     a.set_ylim(top=85)
                # a.set_ylim(0.25, 0.26)
            elif a_ix==2:
                a.set_axis_off()
            elif a_ix>2:
                a.plot(jnp.array(res['params'])[:,a_ix-2])
                hp_name = np.array(res['hp_names'])[a_ix-2]
                a.set_title('Trace plot for '+ names_latex[hp_name])
                a.set_xlabel('Iterations')
                a.axhline(true_vals[hp_name], color='black')
            else:  
                a.plot(jnp.array(res['params'])[:,a_ix-1])
                hp_name = np.array(res['hp_names'])[a_ix-1]
                a.set_title('Trace plot for '+ names_latex[hp_name])
                a.set_xlabel('Iterations')
                a.axhline(true_vals[hp_name], color='black')
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
    plt.savefig(path + f'/synth_hp_tuning_prior_hparams_{loglik_type}.png')

#%%
