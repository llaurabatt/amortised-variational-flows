#%%
import os
# env TeX Live is incomplete (no latex.fmt); use the working SYSTEM TeX at /usr/bin
os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')
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
flags.DEFINE_string('loss', 'mean_dist', 'Loss-metric subfolder (mean_dist | mean_sq_dist).')
flags.DEFINE_string('tag', 'w_a_k_lk_eta',
                    'Tuned-hparam tag = tune_<tag>/ subfolder and hp_info_<tag>_* files '
                    '(e.g. w_a_lk_eta for the sigma_k=1 runs).')
flags.DEFINE_string('tag_suffix', '',
                    'Appended to the tune_<tag> FOLDER name only, not the file names '
                    '(e.g. _extended for warmstart probe runs).')
flags.DEFINE_list('inits', ['default', 'mixed', 'low', 'high'],
                  'Init names to plot (e.g. warmstart for probe runs).')
flags.DEFINE_integer('max_iters', 4001,
                     'Iteration window to plot (probe runs go to 15k).')
flags.mark_flags_as_required(['path'])
FLAGS(sys.argv)
#%%
# path = '/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow'

# eta_vals = [0.000001, 0.001000, 0.300000, 0.500000, 0.700000, 1.000000]


#########################################################################################################################################################
#%%
path = FLAGS.path + f'/tune_{FLAGS.tag}{FLAGS.tag_suffix}/{FLAGS.loss}'  # hp_info_*.sav and hp_tuning_*.png live here
init_names = list(FLAGS.inits)
optimisers = [ 'elbo_opt', 'plain_lr1', 'plain_lr2']
init_eta_vals = [1.00, 0.50, 0.00]
#%%
with open(path + f'/hp_info_{FLAGS.tag}_{init_names[0]}_{optimisers[0]}_new.sav', 'rb') as fr:
    res = pickle.load(fr)
hp_names = res['hp_names'].copy()

# Self-describing loss-panel title: the tuning objective is stamped into each
# .sav as 'loss_metric' (see train_vmp_flow_allhp_smallcondval.py). Map it to a
# title + display transform. 'mean_sq_dist' stores the mean of SQUARED distances,
# so plot its sqrt (RPMSE) -> distance units. Default 'mean_dist' covers pre-stamp
# .sav files (all of which used the MD objective).
METRIC_DISPLAY = {
    'mean_dist':    {'title': 'Mean posterior distance to held-out anchors',
                     'transform': lambda x: x},
    'mean_sq_dist': {'title': 'Root posterior mean squared error (RPMSE)',
                     'transform': np.sqrt},
}
disp = METRIC_DISPLAY[res.get('loss_metric', 'mean_dist')]

#########################################################################################################################################################
#%%|
# chosen optimum: default sigma_a=5.5 sigma_w=11 sigma_k=0.4 ell_k=0.2
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['text.usetex'] = True  # uses SYSTEM /usr/bin latex (PATH set at top)

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
        with open(path + f'/hp_info_{FLAGS.tag}_{init_type}_{optimiser_name}_new.sav', 'rb') as fr:
            res = pickle.load(fr)
        last_loss = np.array(res['loss'])[-20:].mean()
        last_losses.append(last_loss)
    best_init_ix = np.argmin(last_losses) 
    for init_ix, init_type in enumerate(init_names):
        with open(path + f'/hp_info_{FLAGS.tag}_{init_type}_{optimiser_name}_new.sav', 'rb') as fr:
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
                a.plot(disp['transform'](np.array(res['loss'])[:FLAGS.max_iters]), alpha=alpha, color=color,
                           label=f'Init {init_ix + 1}', linestyle=linestyle)
                a.set_xlabel('Iterations')
                a.set_title(disp['title'])
            elif a_ix < (n_plots):  
                a.plot(np.array(res['params'])[:FLAGS.max_iters,rolled_indices][:,a_ix-1], alpha=alpha, 
                           color=color, label=f'Init {init_ix + 1}', linestyle=linestyle)
                hp_name = np.array(res['hp_names'])[rolled_indices][a_ix-1]
                a.set_title('Trace for '+ names_latex[hp_name])
                a.set_xlabel('Iterations')
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=len(init_names))

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.93, wspace=0.2, hspace=0.4)
    # metric label in filename: rpmse for the mean_sq_dist objective (plotted as sqrt)
    metric_label = 'rpmse' if res.get('loss_metric') == 'mean_sq_dist' else 'mean_dist'
    plt.savefig(path + f'/hp_tuning_{FLAGS.tag}{FLAGS.tag_suffix}_{metric_label}_{optimiser_name}.png')
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
