#%%
"""Plot the loss-surface scan written by train_vmp_flow_allhp_smallcondval's
scan_loss_surface() (workdir/loss_surface_scan.sav).

Produces:
  - hp_loss_surface_1d.png : PMSE vs each hyperparameter (1-D sweeps), with the
    rho training-support range shaded and the reference point marked. Flat curve
    => weakly identified; clear U with an interior minimum => identified.
  - hp_loss_surface_2d_sk_sw.png : PMSE over (sigma_k, sigma_w) with product
    contours sigma_k*sigma_w = const overlaid. Loss constant ALONG a contour
    => the field-amplitude product is identified but the ratio is not.
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

# usetex / serif / cm fonts come from linguistic_profiles/matplotlibrc when run
# from that dir; set the essentials explicitly so it also works elsewhere.
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

LATEX = {'eta': r'$\eta$', 'w_prior_scale': r'$\sigma_w$', 'a_prior_scale': r'$\sigma_a$',
         'kernel_amplitude': r'$\sigma_k$', 'kernel_length_scale': r'$\ell_k$'}
ref = R['ref']

# rho training distributions (config Eqn 12). For Uniform we shade the support;
# for Gamma (no hard bound) we shade the 10-90% central band.
RHO = {'w_prior_scale': ('gamma', 5., 1.), 'a_prior_scale': ('gamma', 10., 1.),
       'kernel_amplitude': ('uniform', 0.1, 0.4), 'kernel_length_scale': ('uniform', 0.2, 0.5),
       'eta': ('uniform', 0., 1.)}  # Beta(0.5,0.5) support is [0,1]

def rho_band(name):
    spec = RHO.get(name)
    if spec is None:
        return None
    kind, a, b = spec
    if kind == 'uniform':
        return a, b
    # gamma(shape=a, rate=b): 10-90% quantiles
    try:
        from scipy.stats import gamma
        return float(gamma.ppf(0.1, a, scale=1. / b)), float(gamma.ppf(0.9, a, scale=1. / b))
    except Exception:
        m, sd = a / b, (a ** 0.5) / b  # normal approx, z_0.9 = 1.2816
        return m - 1.2816 * sd, m + 1.2816 * sd

# ---- 1-D sweeps ----------------------------------------------------------
names = [n for n in R['cond_names'] if n in R['sweeps']]
ncol = 3
nrow = int(np.ceil(len(names) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
for ax in axes.flat:
    ax.set_visible(False)
for k, name in enumerate(names):
    ax = axes.flat[k]; ax.set_visible(True)
    s = R['sweeps'][name]
    x = np.asarray(s['x']); y = np.asarray(s['pmse']); sd = np.asarray(s['pmse_sd'])
    ax.fill_between(x, y - sd, y + sd, color='tab:blue', alpha=0.2)
    ax.plot(x, y, color='tab:blue', marker='o', ms=3)
    band = rho_band(name)
    if band is not None:
        ax.axvspan(band[0], band[1], color='tab:green', alpha=0.10,
                   label=r'$\rho$ training range')
    if name in ref:
        ax.axvline(ref[name], color='grey', ls=':', lw=1.2,
                   label='value held in other panels')
    ax.set_title('PMSE vs ' + LATEX.get(name, name))
    ax.set_xlabel(LATEX.get(name, name)); ax.set_ylabel('held-out anchor PMSE')
    ax.grid(True, ls='--', alpha=0.6)
    if k == 0:
        ax.legend(fontsize=8)
fig.suptitle('Held-out PMSE sensitivity to each hyperparameter '
             '(flat = unidentified, U-shape = identified)', y=1.02)
fig.tight_layout()
out1 = path + '/hp_loss_surface_1d.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
print('wrote', out1)

# ---- 2-D (sigma_k, sigma_w) sheet ---------------------------------------
if 'grid_sk_sw' in R:
    g = R['grid_sk_sw']
    sk = np.asarray(g['sigma_k']); sw = np.asarray(g['sigma_w'])
    Z = np.asarray(g['pmse'])  # shape (len(sk), len(sw))
    fig2, ax = plt.subplots(figsize=(6.5, 5))
    SW, SK = np.meshgrid(sw, sk)
    pcm = ax.pcolormesh(SW, SK, Z, shading='auto', cmap='viridis')
    fig2.colorbar(pcm, ax=ax, label='held-out anchor PMSE')
    # product contours sigma_k*sigma_w = const
    PROD = SK * SW
    levels = np.round(np.quantile(PROD, [0.2, 0.4, 0.6, 0.8]), 2)
    cs = ax.contour(SW, SK, PROD, levels=levels, colors='white', linewidths=1, alpha=0.8)
    ax.clabel(cs, fmt=r'$\sigma_k\sigma_w$=%.2f', fontsize=8)
    ax.set_xlabel(r'$\sigma_w$ (w_prior_scale)')
    ax.set_ylabel(r'$\sigma_k$ (kernel_amplitude)')
    ax.set_title('PMSE over $(\\sigma_k,\\sigma_w)$\n'
                 'flat ALONG white product-contours => ratio unidentified')
    fig2.tight_layout()
    out2 = path + '/hp_loss_surface_2d_sk_sw.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print('wrote', out2)
else:
    print('no 2-D grid in scan (need both kernel_amplitude and w_prior_scale in cond_hparams)')
