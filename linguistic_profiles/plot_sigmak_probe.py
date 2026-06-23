import os
os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from absl import flags
import sys

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'Workdir containing tune_* subdirs.')
flags.mark_flags_as_required(['workdir'])
FLAGS(sys.argv)

workdir = FLAGS.workdir
ph0_dir = f'{workdir}/tune_w_a_k_lk_eta/mean_dist/'
ph1_dir = f'{workdir}/tune_w_a_lk_eta/mean_dist/'
ph2_dir = f'{workdir}/tune_w_a_k_lk_eta_probe/mean_dist/'

OPTIMISERS  = ['elbo_opt', 'plain_lr1', 'plain_lr2']
SIGMA_K_FIXED = 0.4
N = 5000  # steps per phase

mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['text.usetex'] = True

names_latex = {
    'eta': r'$\eta$',
    'w_prior_scale': r'$\sigma_w$',
    'a_prior_scale': r'$\sigma_a$',
    'kernel_amplitude': r'$\sigma_k$',
    'kernel_length_scale': r'$\ell_k$',
}
colors = ['purple', 'orange', 'green']

# Panel order: σ_w, σ_a, η, σ_k, ℓ_k  (matches existing plots)
HP_ORDER = ['w_prior_scale', 'a_prior_scale', 'eta', 'kernel_amplitude', 'kernel_length_scale']

# Phase 0: best run from original mean_dist full tune (plain_lr2 high — used as warmstart)
with open(f'{ph0_dir}hp_info_w_a_k_lk_eta_high_plain_lr2_new.sav', 'rb') as f:
    d0 = pickle.load(f)
ph0_names  = list(d0['hp_names'])
ph0_params = np.array(d0['params'])
ph0_loss   = np.array(d0['loss'])

steps_ph0 = np.arange(1, N + 1)
steps_ph1 = np.arange(N + 1, 2*N + 1)
steps_ph2 = np.arange(2*N + 1, 3*N + 1)

# Find best Phase-1/2 optimiser by Phase 2 final loss
last_losses = []
for opt in OPTIMISERS:
    with open(f'{ph2_dir}hp_info_w_a_k_lk_eta_warmstart_{opt}_new.sav', 'rb') as f:
        d = pickle.load(f)
    last_losses.append(np.array(d['loss'])[-20:].mean())
best_ix = int(np.argmin(last_losses))

fig, ax = plt.subplots(2, 3, figsize=(10, 7))

for opt_ix, opt in enumerate(OPTIMISERS):
    with open(f'{ph1_dir}hp_info_w_a_lk_eta_warmstart_{opt}_new.sav', 'rb') as f:
        d1 = pickle.load(f)
    with open(f'{ph2_dir}hp_info_w_a_k_lk_eta_warmstart_{opt}_new.sav', 'rb') as f:
        d2 = pickle.load(f)

    ph1_names  = list(d1['hp_names'])
    ph2_names  = list(d2['hp_names'])
    ph1_params = np.array(d1['params'])
    ph2_params = np.array(d2['params'])
    ph1_loss   = np.array(d1['loss'])
    ph2_loss   = np.array(d2['loss'])

    # Phase 1 final values as stitching point into Phase 2
    ph1_final = {n: ph1_params[-1, i] for i, n in enumerate(ph1_names)}
    ph1_final['kernel_amplitude'] = SIGMA_K_FIXED

    color     = 'black'  if opt_ix == best_ix else colors[opt_ix]
    alpha     = 1.0      if opt_ix == best_ix else 0.3
    linestyle = 'dashed' if opt_ix == best_ix else 'solid'
    label     = f'Opt {opt_ix + 1}'

    # Stitch steps: ph0 | ph1 connect | ph2
    steps_s = np.concatenate([steps_ph0, steps_ph1, [2*N], steps_ph2])

    for a_ix, a in enumerate(ax.flatten()):
        a.grid(True, linestyle='--', alpha=0.7)
        if a_ix == 0:
            loss_s = np.concatenate([ph0_loss, ph1_loss, [ph1_loss[-1]], ph2_loss])
            a.plot(steps_s, loss_s, alpha=alpha, color=color,
                   linestyle=linestyle, label=label)
            a.set_xlabel('Iterations')
            a.set_title('Mean posterior distance to held-out anchors')
        elif a_ix <= len(HP_ORDER):
            hp = HP_ORDER[a_ix - 1]
            trace_ph0 = ph0_params[:, ph0_names.index(hp)]
            if hp == 'kernel_amplitude':
                trace_ph1 = np.full(N, SIGMA_K_FIXED)
            else:
                trace_ph1 = ph1_params[:, ph1_names.index(hp)]
            trace_ph2 = ph2_params[:, ph2_names.index(hp)]
            trace_s = np.concatenate([trace_ph0, trace_ph1, [ph1_final[hp]], trace_ph2])
            a.plot(steps_s, trace_s, alpha=alpha, color=color,
                   linestyle=linestyle, label=label)
            a.set_title('Trace for ' + names_latex[hp])
            a.set_xlabel('Iterations')

# Phase dividers
for a in ax.flatten():
    a.axvline(N,   color='k', linestyle=':', lw=1.5, zorder=3)
    a.axvline(2*N, color='k', linestyle=':', lw=1.5, zorder=3)
    a.set_xlim(1, 3*N)

# Phase labels on loss panel
for x, txt in [(N/2, r'Phase 0'), (N + N/2, r'Phase 1'), (2*N + N/2, r'Phase 2')]:
    ax[0,0].text(x / (3*N), 0.97, txt, ha='center', va='top',
                 transform=ax[0,0].transAxes, fontsize=9)

# Subtitles for phases
ax[0,0].text(0.5/3, 0.88, r'($\sigma_k$ free)', ha='center', va='top',
             transform=ax[0,0].transAxes, fontsize=8, color='#555555')
ax[0,0].text(1.5/3, 0.88, r'($\sigma_k$ fixed=0.4)', ha='center', va='top',
             transform=ax[0,0].transAxes, fontsize=8, color='#555555')
ax[0,0].text(2.5/3, 0.88, r'($\sigma_k$ free)', ha='center', va='top',
             transform=ax[0,0].transAxes, fontsize=8, color='#555555')

handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.08),
           ncol=len(OPTIMISERS))

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.15, right=None, top=0.93, wspace=0.2, hspace=0.4)
fig.suptitle(r'$\sigma_k$ stability probe: free $\to$ fixed 0.4 $\to$ free', fontsize=13)

out = f'{workdir}/sigmak_probe_traces.png'
plt.savefig(out)
print(f'Saved {out}')
