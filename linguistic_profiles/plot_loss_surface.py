#%%
"""Plot the loss-surface scan written by train_vmp_flow_allhp_smallcondval's
scan_loss_surface() (workdir/loss_surface_scan.sav).

For each held-out error metric (mean distance and root-PMSE, both in km) it
produces:
  - hp_loss_surface_1d_<metric>.png : metric vs each hyperparameter (1-D sweeps),
    rho training-support shaded and the reference point marked. Flat curve =>
    weakly identified; clear U with an interior minimum => identified.
  - hp_loss_surface_2d_<x>_<y>_<metric>.png : metric over a 2-D (x, y) sheet, for
    every axis pair in the scan. For (sigma_k, sigma_w) the product contours
    sigma_k*sigma_w = const are overlaid (flat ALONG a contour => the amplitude
    product is identified but the ratio is not).

Reads both the generalised `grids` dict (keyed '<x>__<y>', holding meandist_km +
rootpmse_km) and the legacy `grid_sk_sw` key (rootpmse_km only).
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # .sav holds jax arrays; avoid TPU
# The env's bundled TeX Live is incomplete (no latex.fmt). Put /usr/bin first so
# matplotlib's latex/dvipng subprocesses use the working SYSTEM TeX instead.
os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')
import pickle
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from absl import flags

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'

FLAGS = flags.FLAGS
flags.DEFINE_string('path', None, 'Workdir containing loss_surface_scan.sav.')
flags.mark_flags_as_required(['path'])
FLAGS(sys.argv)
path = FLAGS.path

with open(path + '/loss_surface_scan.sav', 'rb') as f:
    R = pickle.load(f)

KM = float(R.get('km_per_unit', 200.0))
LATEX = {'eta': r'$\eta$', 'w_prior_scale': r'$\sigma_w$', 'a_prior_scale': r'$\sigma_a$',
         'kernel_amplitude': r'$\sigma_k$', 'kernel_length_scale': r'$\ell_k$'}
# _HP_CODES (project convention) for filenames: e.g. (sigma_w, sigma_k) -> 'w_k'.
HP_CODES = {'w_prior_scale': 'w', 'a_prior_scale': 'a', 'kernel_amplitude': 'k',
            'kernel_length_scale': 'lk', 'eta': 'eta'}
ref = R['ref']

# Held-out error metrics, both expressed in km. Each maps a sweep-row dict to a
# plotted series + an error band. root-PMSE = sqrt(E||X-x||^2); its SD is taken
# by the delta method, sd(sqrt(m)) ~= sd(m) / (2 sqrt(m)).
METRICS = {
    'meandist': dict(
        name='Mean posterior distance',
        ylabel='held-out mean distance (km)',
        y=lambda s: KM * np.asarray(s['mean_dist']),
        sd=lambda s: KM * np.asarray(s['mean_dist_sd']),
        grid_key='meandist_km'),
    'rootpmse': dict(
        name='Root posterior mean squared error',
        ylabel='held-out root-PMSE (km)',
        y=lambda s: KM * np.sqrt(np.asarray(s['mean_sq'])),
        sd=lambda s: KM * np.asarray(s['mean_sq_sd']) /
                     (2.0 * np.sqrt(np.maximum(np.asarray(s['mean_sq']), 1e-12))),
        grid_key='rootpmse_km'),
}

RHO = {'w_prior_scale': ('gamma', 5., 1.), 'a_prior_scale': ('gamma', 10., 1.),
       'kernel_amplitude': ('uniform', 0.1, 0.4), 'kernel_length_scale': ('uniform', 0.2, 0.5),
       'eta': ('uniform', 0., 1.)}

# Hard box the SGD-optimised hparams are clipped to each step (jnp.clip in
# tune_vmp_hparams) -- distinct from the rho training distribution above. Drawn
# as red lines; for sigma_k/ell_k it coincides with the uniform rho range.
CLIP = {'w_prior_scale': (0., 10.), 'a_prior_scale': (3., 19.),
        'kernel_amplitude': (0.1, 0.4), 'kernel_length_scale': (0.2, 0.5),
        'eta': (0., 1.)}

def rho_band(name):
    spec = RHO.get(name)
    if spec is None:
        return None
    kind, a, b = spec
    if kind == 'uniform':
        return a, b
    try:
        from scipy.stats import gamma
        return float(gamma.ppf(0.1, a, scale=1. / b)), float(gamma.ppf(0.9, a, scale=1. / b))
    except Exception:
        m, sd = a / b, (a ** 0.5) / b
        return m - 1.2816 * sd, m + 1.2816 * sd

# ---- 1-D sweeps (one figure per metric) ----------------------------------
names = [n for n in R['cond_names'] if n in R['sweeps']]
ncol = 3
nrow = int(np.ceil(len(names) / ncol))
for mkey, M in METRICS.items():
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, name in enumerate(names):
        ax = axes.flat[k]; ax.set_visible(True)
        s = R['sweeps'][name]
        x = np.asarray(s['x']); y = M['y'](s); sd = M['sd'](s)
        ax.fill_between(x, y - sd, y + sd, color='tab:blue', alpha=0.2)
        ax.plot(x, y, color='tab:blue', marker='o', ms=3)
        band = rho_band(name)
        if band is not None:
            ax.axvspan(band[0], band[1], color='tab:green', alpha=0.10,
                       label=r'$\rho$ training range')
        if name in ref:
            ax.axvline(ref[name], color='grey', ls=':', lw=1.2,
                       label='value held in other panels')
        clip = CLIP.get(name)
        if clip is not None:
            for bi, b in enumerate(clip):
                ax.axvline(b, color='red', lw=1.0, alpha=0.8,
                           label='clip bounds' if bi == 0 else None)
        ax.set_title(M['name'] + ' vs ' + LATEX.get(name, name))
        ax.set_xlabel(LATEX.get(name, name)); ax.set_ylabel(M['ylabel'])
        ax.grid(True, ls='--', alpha=0.6)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(M['name'] + ' sensitivity to each hyperparameter '
                 '(flat = unidentified, U-shape = identified)', y=1.02)
    fig.tight_layout()
    out1 = path + f'/hp_loss_surface_1d_{mkey}.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    print('wrote', out1)

# ---- 2-D sheets ----------------------------------------------------------
# Grids are keyed by their _HP_CODES tag (e.g. 'w_k', 'k_eta'); x_name is the
# canonical-first hparam (horizontal axis), y_name canonical-second (vertical).
def plot_surface(code, xname, yname, xvals, yvals, Z, mkey, M):
    xv = np.asarray(xvals); yv = np.asarray(yvals); Z = np.asarray(Z)  # Z[i_x, j_y]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    pcm = ax.pcolormesh(xv, yv, Z.T, shading='auto', cmap='viridis')  # x horizontal, y vertical
    fig.colorbar(pcm, ax=ax, label=M['ylabel'])
    title = M['name'] + r' over $(%s,%s)$' % (
        LATEX.get(xname, xname).strip('$'), LATEX.get(yname, yname).strip('$'))
    # product contours only meaningful for the (sigma_w, sigma_k) amplitude pair
    if {xname, yname} == {'kernel_amplitude', 'w_prior_scale'}:
        XV, YV = np.meshgrid(xv, yv)  # shape (len(yv), len(xv)) matching Z.T
        PROD = XV * YV
        levels = np.round(np.quantile(PROD, [0.2, 0.4, 0.6, 0.8]), 2)
        cs = ax.contour(xv, yv, PROD, levels=levels, colors='white', linewidths=1, alpha=0.8)
        ax.clabel(cs, fmt=r'$\sigma_w\sigma_k$=%.2f', fontsize=8)
        title += '\n(loss flat along white product-contours: ratio unidentified)'
    # clip box (red): the hard SGD-optimiser bounds on each axis (data NOT clipped)
    cx, cy = CLIP.get(xname), CLIP.get(yname)
    if cx is not None:
        for bi, b in enumerate(cx):
            ax.axvline(b, color='red', lw=1.0, alpha=0.8, label='clip bounds' if bi == 0 else None)
    if cy is not None:
        for b in cy:
            ax.axhline(b, color='red', lw=1.0, alpha=0.8)
    if cx is not None or cy is not None:
        ax.legend(fontsize=8, loc='upper right')
    ax.set_xlabel(LATEX.get(xname, xname)); ax.set_ylabel(LATEX.get(yname, yname))
    ax.set_title(title)
    fig.tight_layout()
    out = path + f'/hp_loss_surface_2d_{code}_{mkey}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out)
    plt.close(fig)

made_any = False
for _key, g in R.get('grids', {}).items():
    xn, yn = g['x_name'], g['y_name']
    code = f"{HP_CODES.get(xn, xn)}_{HP_CODES.get(yn, yn)}"  # derived from names, not the dict key
    for mkey, M in METRICS.items():
        gk = M['grid_key']
        if gk in g:
            plot_surface(code, xn, yn, g['x'], g['y'], g[gk], mkey, M)
            made_any = True
if not made_any:
    print('no 2-D grid in scan (need both axes of a pair in cond_hparams)')
