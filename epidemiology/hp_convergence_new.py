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
flags.DEFINE_string('path_bayes', None, 'Path to Bayes hyperparameter optimisation results.')
flags.DEFINE_string('path_smi', None, 'Path to SMI hyperparameter optimisation results.')
flags.mark_flags_as_required(['path_bayes', 'path_smi'])
FLAGS(sys.argv)

#########################################################################################################################################################
#%%
path_bayes = FLAGS.path_bayes
path_smi = FLAGS.path_smi
tune_hparams = 'elpd_loocv'
loglik_types = ['z','y'] 
optimisers = ['elbo_opt'] #, 'plain_lr1', 'plain_lr2']
init_names = ['default','medium', 'high', 'smallmedium', 'small']
init_eta_vals = [1.00, 0.50, 0.00]
names_latex = {'eta': '$\eta$',
                'c1':'$c_1$',
                'c2':'$c_2$',}

#########################################################################################################################################################
#%%
# path='/home/llaurabat/modularbayes-output-integrated-etac1c2-smallcondval-gamma_a0.6_scale2/epidemiology_new/nsf/vmp_flow'
# path='/home/llaurabat/NEW-modularbayes-output-integrated-etac1c2-smallcondval-uniform_a0_b15/epidemiology_new/nsf/vmp_flow'

# OLD Chosen opt: for y: medium at pos. 10000 init 3: eta[1.0], [1.6, 4.5] 2,3
# OLD Chosen opt: for z: medium at pos. 10000 init 3: eta[0.0], [0.1, 4.5] 0,3
# Chosen opt Gamma(0.6, scale=2): for y (bayes): default at pos. 10000 init 3: eta[1.0], [4.5, 4.5] 2,3
# Chosen opt Gamma(0.6, scale=2): for z (cut): default at pos. 10000 init 3: eta[0.0], [0.2, 1.4] 0,3
# Chosen opt Gamma(0.6, scale=4): for y (bayes): default at pos. 10000 init small or default: eta[1.0], [5, 0.1] 2,3
# Chosen opt Gamma(0.6, scale=4): for z (cut): default at pos. 10000 init default: eta[0.0], [0.1, 1.] 0,3
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['text.usetex'] = True

for optimiser in optimisers:
    for loglik_type in loglik_types:
        print(f"loglik type: {loglik_type}, optimiser {optimiser}")
        colors = ['purple', 'orange', 'green', 'red', 'blue']
        with open(path_smi + f'/hp_info_etac1c2_{tune_hparams}_{init_names[0]}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
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
        last_losses = []
        for init_ix, init_type in enumerate(init_names):
            with open(path_smi + f'/hp_info_etac1c2_{tune_hparams}_{init_type}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
                res = pickle.load(fr)
            last_loss = jnp.array(res['loss'])[-20:].mean()
            last_losses.append(last_loss)
        best_init_ix = np.argmin(last_losses)
        for init_ix, init_type in enumerate(init_names):
            with open(path_smi + f'/hp_info_etac1c2_{tune_hparams}_{init_type}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
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
                    a.plot(jnp.array(res['loss']), alpha=alpha, color=color, 
                           label=f'Init {init_ix + 1}', linestyle=linestyle)
                    a.set_xlabel('Iterations')
                    a.set_title('Negative LOOCV Log-Likelihood')
                elif a_ix < (n_plots):  
                    a.plot(jnp.array(res['params'])[:,rolled_indices][:,a_ix-1], alpha=alpha, 
                           color=color, label=f'Init {init_ix + 1}', linestyle=linestyle)
                    hp_name = np.array(res['hp_names'])[rolled_indices][a_ix-1]
                    a.set_title('Trace for '+ names_latex[hp_name])
                    a.set_xlabel('Iterations')
                a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        for a_ix, a in enumerate(ax.flatten()):
            if (a_ix -1)==1:
                a.set_ylim(bottom=-0.1, top=1.1)
            elif a_ix!=0:
                a.set_ylim(bottom=-0.1)
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=len(init_names))

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
        plt.savefig(path_smi + f'/hp_tuning_all_hparams_{tune_hparams}_{loglik_type}_{optimiser}.png')
        plt.show()



#%%
#########################################################################################################################################################
# path = '/home/llaurabat/NEW-modularbayes-output-integrated-onlyc1c2-smallcondval-uniform_a0_b15/epidemiology_new/nsf/vmp_flow'

mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['text.usetex'] = True
# OLD Chosen opt: for y (that will want bayes): medium at pos. 10000 init 3: [0.1, 0.1] (old: default at pos. 8000: [0., 1.1])
# OLD Chosen opt: for z (that will want cut): medium at pos. 10000 init 3: [2.8, 1.2](old: default at pos. 8000: [2.8, 3.2]
# Chosen opt Gamma(0.6,scale=2): for y (that will want bayes): default at pos. 10000 init 3: [6.3, 1.2] (old: default at pos. 8000: [0., 1.1])
# Chosen opt Gamma(0.6,scale=2): for z (that will want cut): default at pos. 10000 init 3: [0.1, 0.1](old: default at pos. 8000: [2.8, 3.2]
# Chosen opt Gamma(0.6,scale=4): for y (that will want bayes): default at pos. 10000 init default: [6, 0.1]
# Chosen opt Gamma(0.6,scale=4): for z (that will want cut): default at pos. 10000 init default: [0.1, 3]

for optimiser in optimisers:
    for loglik_type in loglik_types:
        print(f"loglik type: {loglik_type}, optimiser {optimiser}")
        colors = ['purple', 'orange', 'green', 'red', 'blue']
        with open(path_bayes + f'/hp_info_onlyc1c2_{tune_hparams}_{init_names[0]}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
            res = pickle.load(fr)
        hp_names = res['hp_names'].copy()
        hp_names = np.array(hp_names)
        n_plots = len(hp_names)+2
        fig, ax = plt.subplots(int(n_plots/2)+int(n_plots%2>0), 2, 
                            figsize=(7,3.5*(int(n_plots/2)+int(n_plots%2>0))))
        last_losses = []
        for init_ix, init_type in enumerate(init_names):
            with open(path_bayes + f'/hp_info_onlyc1c2_{tune_hparams}_{init_type}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
                res = pickle.load(fr)
            last_loss = jnp.array(res['loss'])[-20:].mean()
            last_losses.append(last_loss)
        best_init_ix = np.argmin(last_losses)
        for init_ix, init_type in enumerate(init_names):
            with open(path_bayes + f'/hp_info_onlyc1c2_{tune_hparams}_{init_type}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
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
                    a.plot(jnp.array(res['loss']), alpha=alpha, color=color, label=f'Init {init_ix + 1}', linestyle=linestyle)
                    a.set_xlabel('Iterations')
                    a.set_title('Negative LOOCV Log-Likelihood')
                    # if loglik_type=='y':
                    #     # a.set_ylim(top=65)
                    # a.set_ylim(0.25, 0.26)
                elif a_ix==2:
                    a.set_axis_off()
                elif a_ix>2:
                    a.plot(jnp.array(res['params'])[:,a_ix-2], alpha=alpha, color=color, label=f'Init {init_ix + 1}', linestyle=linestyle)
                    hp_name = np.array(res['hp_names'])[a_ix-2]
                    a.set_title('Trace for '+ names_latex[hp_name])
                    a.set_xlabel('Iterations')

                else:  
                    a.plot(jnp.array(res['params'])[:,a_ix-1], alpha=alpha, color=color, label=f'Init {init_ix + 1}', linestyle=linestyle)
                    hp_name = np.array(res['hp_names'])[a_ix-1]
                    a.set_title('Trace for '+ names_latex[hp_name])
                    a.set_xlabel('Iterations')
                a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        for a_ix, a in enumerate(ax.flatten()):
            if ((a_ix!=0) and (a_ix!=2)):
                a.set_ylim(bottom=-0.1)
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=len(init_names))

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
        plt.savefig(path_bayes + f'/hp_tuning_prior_hparams_{tune_hparams}_{loglik_type}_{optimiser}.png')
        plt.show()

#%%
###################### converged points ############################################################################################################
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['text.usetex'] = True
# path_smi = '/home/llaurabat/NEW-modularbayes-output-integrated-etac1c2-smallcondval-uniform_a0_b15/epidemiology_new/nsf/vmp_flow'
# path_bayes = '/home/llaurabat/NEW-modularbayes-output-integrated-onlyc1c2-smallcondval-uniform_a0_b15/epidemiology_new/nsf/vmp_flow'
paths = { 'prior': path_bayes, 'all': path_smi,}
colors = ['purple', 'orange', 'green', 'blue', 'red']
for path_name, path in paths.items():
    for optimiser in optimisers:
        for loglik_type in loglik_types:
            path_type = 'only' if path_name=='prior' else 'eta'
            print(f"path name: {path_name}, loglik type: {loglik_type}, optimiser {optimiser}")
            with open(path + f'/hp_info_{path_type}c1c2_{tune_hparams}_{init_names[0]}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
                res = pickle.load(fr)
            hp_names = res['hp_names'].copy()
            hp_names = np.array(hp_names)

            n_plots = len(hp_names)
            fig, ax = plt.subplots(1, n_plots,
                                figsize=(3.5*n_plots,3.5))

            
            last_losses = []
            for init_ix, init_type in enumerate(init_names):
                with open(path + f'/hp_info_{path_type}c1c2_{tune_hparams}_{init_type}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
                    res = pickle.load(fr)
                last_loss = jnp.array(res['loss'])[-20:].mean()
                last_losses.append(last_loss)
            best_init_ix = np.argmin(last_losses)

            for init_ix, init_type in enumerate(init_names):
                with open(path + f'/hp_info_{path_type}c1c2_{tune_hparams}_{init_type}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
                    res = pickle.load(fr)
                loss = jnp.array(res['loss'])
                if init_ix==best_init_ix:
                    print(f"20-last avg best {hp_name}: {jnp.array(res['params'])[-20:,:].mean(0)}")

                for a_ix, a in enumerate(ax.flatten()):
                    hp_name = np.array(res['hp_names'])[a_ix]
                    a.grid(True, linestyle='--', alpha=0.7)
                    color = colors[init_ix]
                    if init_ix!=best_init_ix:
                        alpha = 0.3
                        s = 20.
                    else:
                        alpha = 1
                        s = 60
                        color = 'black'


                    if ((hp_name == 'eta')&(loglik_type=='y')):
                        a.set_ylim(0.85, 0.9)

                    if ((hp_name == 'c1')&(loglik_type=='z')&(path_name=='prior')):
                        a.set_ylim(1.7, 1.73)
                        
                    if ((loglik_type=='z')&(path_name=='prior')):
                        a.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                    else:
                        a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                    if ((hp_name == 'c2')&((loglik_type=='z')|(loglik_type=='y'))&(path_name=='prior'))|((hp_name == 'c2')&(loglik_type=='y')&(path_name=='all')):
                        a.set_ylim(14.8, 15.2)
                    elif (hp_name == 'c2'):
                        a.set_ylim(13.5, 15.2)


                    a.set_title(names_latex[hp_name])
                    a.scatter(loss[-20:].mean(), jnp.array(res['params'])[:,a_ix][-20:].mean(), 
                              label=f'Init {init_ix + 1}', alpha=alpha, s=s, color=color)
                    a.set_xlabel('Negative LOOCV Log-Likelihood')
                    a.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    
            handles, labels = ax[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=len(init_names))

            plt.tight_layout()
            plt.subplots_adjust(left=None, bottom=0.3, right=None, top=0.93, wspace=0.3, hspace=0.4)
            plt.savefig(path + f'/hp_converged_{path_name}_hparams_{tune_hparams}_{loglik_type}_{optimiser}.png')
            plt.show()




#%%
###################### same but with CIs on SMI ############################################################################################################
for optimiser in optimisers:
    for loglik_type in loglik_types:
        print(f"loglik type: {loglik_type}, optimiser {optimiser}")
        losses = []
        params = []

        # Load data
        for init_type in init_names:
            with open(path_smi + f'/hp_info_etac1c2_{tune_hparams}_{init_type}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
                res = pickle.load(fr)
            losses.append(res['loss'])
            params.append(res['params'])

        # Convert list to numpy array for easier manipulation
        params = np.array(params)  # Shape: (num_init_types, num_iterations, num_params)

        # Compute means and standard deviations for params
        mean_params = np.mean(params, axis=0)
        std_params = np.std(params, axis=0)
        ci_upper = mean_params + 2 * std_params #/ np.sqrt(len(init_names))
        ci_lower = mean_params - 2 * std_params #/ np.sqrt(len(init_names))

        # Loss
        mean_losses = np.mean(losses, axis=0)
        std_losses = np.std(losses, axis=0)
        losses_upper = mean_losses + 2 * std_losses #/ np.sqrt(len(init_names))
        losses_lower = mean_losses - 2 * std_losses #/ np.sqrt(len(init_names))

        # Get hyperparameter names and set up rolled indices
        with open(path_smi + f'/hp_info_etac1c2_{tune_hparams}_{init_names[0]}_{loglik_type}_{optimiser}.sav', 'rb') as fr:
            res = pickle.load(fr)
        hp_names = res['hp_names'].copy()
        hp_names = np.array(hp_names)
        eta_index = np.where(hp_names == 'eta')[0][0]
        indices = np.arange(len(hp_names))
        indices = np.delete(indices, eta_index)
        rolled_indices = np.insert(indices, 1, eta_index)

        # Plotting setup
        n_plots = len(hp_names)
        fig, ax = plt.subplots(int(n_plots/2) + int(n_plots % 2 > 0), 2, figsize=(7, 3.5 * (int(n_plots/2) + int(n_plots % 2 > 0))))

        # Plot for losses
        a = ax.flatten()[0]
        a.grid(True, linestyle='--', alpha=0.7)
        a.plot(mean_losses, label='Mean Loss')
        a.fill_between(range(len(mean_losses)), losses_lower, losses_upper, alpha=0.2, label='95% CI')
        a.set_title('Negative LOOCV Log-Likelihood')
        a.set_xlabel('Iterations')

        # Plots for other hyperparameters
        for i, idx in enumerate(rolled_indices):
            a = ax.flatten()[i + 1]
            a.grid(True, linestyle='--', alpha=0.7)
            hp_name = hp_names[idx]
            a.plot(mean_params[:, idx], label='Mean')
            a.fill_between(range(mean_params.shape[0]), ci_lower[:, idx], ci_upper[:, idx],  alpha=0.2, label='95% CI')
            a.set_title('Trace for ' + names_latex.get(hp_name, hp_name))
            a.set_xlabel('Iterations')
            if hp_name == 'eta':
                a.set_ylim(bottom=-0.1, top=1.1)

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
        plt.savefig(path_smi + f'/hp_tuning_all_hparams_{tune_hparams}_{loglik_type}_{optimiser}_CI.png')
        plt.show()
#%%

