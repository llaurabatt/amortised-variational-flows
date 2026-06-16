#%%
import os
# env TeX Live is incomplete (no latex.fmt); use the working SYSTEM TeX at /usr/bin
os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from absl import flags
import sys
#%%

FLAGS = flags.FLAGS
flags.DEFINE_string('path', None, 'Workdir of the VMP run (contains tune_eta/).')
flags.DEFINE_string('loss', 'mean_dist', 'Loss-metric subfolder (mean_dist | mean_sq_dist).')
flags.mark_flags_as_required(['path'])
FLAGS(sys.argv)
#%%
# Eta-only SGD traces produced by tune_vmp_hparams(['eta']); see launch_eta_only.sh.
# .sav files live under tune_eta/<loss>/hp_info_eta_{init}_{optimiser}_new.sav.
path = FLAGS.path + f'/tune_eta/{FLAGS.loss}'
init_names = ['default', 'mixed', 'low', 'high']
optimisers = ['elbo_opt', 'plain_lr1', 'plain_lr2']
colors = ['purple', 'orange', 'green', 'red']

mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['text.usetex'] = True  # uses SYSTEM /usr/bin latex (PATH set at top)

# Self-describing titles: the tuning objective is stamped into each .sav as
# 'loss_metric' (see train_vmp_flow_allhp_smallcondval.py). Map it to a panel
# title and a display transform. 'mean_sq_dist' stores the mean of SQUARED
# distances, so we plot its sqrt (RPMSE) -> distance units. Default 'mean_dist'
# covers pre-stamp .sav files (all of which used the MD objective).
METRIC_DISPLAY = {
    'mean_dist':    {'title': 'Mean posterior distance to held-out anchors',
                     'transform': lambda x: x},
    'mean_sq_dist': {'title': 'Root posterior mean squared error (RPMSE)',
                     'transform': np.sqrt},
}


def _load(init_type, optimiser_name):
  fname = path + f'/hp_info_eta_{init_type}_{optimiser_name}_new.sav'
  with open(fname, 'rb') as fr:
    return pickle.load(fr)


#%%
print('Eta-only tuning results (prior scales held at PriorHparams defaults)')
for optimiser_name in optimisers:
  # pick the best init by mean loss over the last 20 steps
  last_losses = []
  for init_type in init_names:
    res = _load(init_type, optimiser_name)
    last_losses.append(np.array(res['loss'])[-20:].mean())
  best_init_ix = int(np.argmin(last_losses))

  fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
  eta_hat = None
  disp = METRIC_DISPLAY[_load(init_names[0], optimiser_name).get('loss_metric', 'mean_dist')]
  for init_ix, init_type in enumerate(init_names):
    res = _load(init_type, optimiser_name)
    eta_idx = list(np.array(res['hp_names'])).index('eta')
    loss = disp['transform'](np.array(res['loss']))
    eta_trace = np.array(res['params'])[:, eta_idx]

    if init_ix == best_init_ix:
      alpha, linestyle, color = 1.0, 'dashed', 'black'
      eta_hat = float(eta_trace[-20:].mean())
    else:
      alpha, linestyle, color = 0.3, 'solid', colors[init_ix]

    ax[0].plot(loss, alpha=alpha, color=color, linestyle=linestyle,
               label=f'Init {init_ix + 1} ({init_type})')
    ax[1].plot(eta_trace, alpha=alpha, color=color, linestyle=linestyle,
               label=f'Init {init_ix + 1} ({init_type})')

  for a in ax:
    a.grid(True, linestyle='--', alpha=0.7)
    a.set_xlabel('Iterations')
  ax[0].set_title(disp['title'])
  ax[1].set_title(r'Trace for $\eta$')
  ax[1].axhline(eta_hat, color='black', linestyle=':', alpha=0.5)

  handles, labels = ax[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04),
             ncol=len(init_names))
  plt.tight_layout()
  plt.subplots_adjust(bottom=0.28, top=0.9, wspace=0.25)
  out = path + f'/eta_tune_{optimiser_name}.png'
  plt.savefig(out)
  plt.close(fig)
  print(f'  {optimiser_name:10s}: eta_hat = {eta_hat:.4f}  (best init: {init_names[best_init_ix]})  -> {out}')
