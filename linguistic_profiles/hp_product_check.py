#%%
"""Diagnostic: test the sigma_k / sigma_w non-identifiability hypothesis.

In the LMC, the field variation enters as gamma_b(x) * w, scaling as
sigma_k * sigma_w, so only the PRODUCT is identified while the RATIO is free.
If that is what drives the boundary-pinning, then across the 4 inits x 3
optimisers the individual sigma_k, sigma_w should scatter to different box
corners while their PRODUCT stays roughly constant.

This reads the hp_info_*_new.sav files written by tune_vmp_hparams and prints,
per file, the converged (last-20-iteration mean) hyperparameters and the
product sigma_k * sigma_w, then summarises the spread (coefficient of
variation) of sigma_k, sigma_w and the product. CV(product) << CV(factors)
confirms the degeneracy.
"""
import os
# Force CPU: the .sav files hold JAX arrays and unpickling does a device_put;
# the TPU is held by the running tune job, so we must not touch it.
os.environ['JAX_PLATFORMS'] = 'cpu'
import pickle
import numpy as np
from absl import flags
import sys

FLAGS = flags.FLAGS
flags.DEFINE_string('path', None, 'Path to hyperparameter optimisation results.')
flags.DEFINE_integer('tail', 20, 'Number of final iterations to average over.')
flags.mark_flags_as_required(['path'])
FLAGS(sys.argv)

path = FLAGS.path
init_names = ['default', 'mixed', 'low', 'high']
optimisers = ['elbo_opt', 'plain_lr1', 'plain_lr2']

rows = []  # (init, opt, sigma_w, sigma_a, sigma_k, ell_k, eta, product)
for opt in optimisers:
    for init in init_names:
        fp = f'{path}/hp_info_etapriorhps_{init}_{opt}_new.sav'
        if not os.path.exists(fp):
            print(f'  [skip] missing: {os.path.basename(fp)}')
            continue
        with open(fp, 'rb') as fr:
            r = pickle.load(fr)
        names = list(r['hp_names'])
        p = np.asarray(r['params'])[-FLAGS.tail:].mean(0)
        d = {n: float(v) for n, v in zip(names, p)}
        sk, sw = d['kernel_amplitude'], d['w_prior_scale']
        rows.append((init, opt, sw, d['a_prior_scale'], sk,
                     d['kernel_length_scale'], d['eta'], sk * sw))

if not rows:
    print('No .sav files found yet — run after tune_vmp_hparams has written them.')
    sys.exit(0)

hdr = f"{'init':<8} {'optim':<10} {'sig_w':>8} {'sig_a':>8} {'sig_k':>8} {'ell_k':>8} {'eta':>8} {'sk*sw':>8}"
print('\n' + hdr)
print('-' * len(hdr))
for init, opt, sw, sa, sk, lk, eta, prod in rows:
    print(f"{init:<8} {opt:<10} {sw:>8.3f} {sa:>8.3f} {sk:>8.4f} {lk:>8.4f} {eta:>8.4f} {prod:>8.4f}")


def cv(x):
    x = np.asarray(x, float)
    m = x.mean()
    return float(x.std() / m) if m != 0 else float('nan')

sk_all = [r[4] for r in rows]
sw_all = [r[2] for r in rows]
pr_all = [r[7] for r in rows]
eta_all = [r[6] for r in rows]
print('\nSpread across all runs (coefficient of variation = std/mean):')
print(f"  sigma_k : CV={cv(sk_all):.3f}  (values {np.round(sk_all,4)})")
print(f"  sigma_w : CV={cv(sw_all):.3f}  (values {np.round(sw_all,3)})")
print(f"  product : CV={cv(pr_all):.3f}  (values {np.round(pr_all,4)})")
print(f"  eta     : CV={cv(eta_all):.3f}  (values {np.round(eta_all,4)})")
print('\nInterpretation: if CV(product) << CV(sigma_k), CV(sigma_w), the field '
      'amplitude sigma_k*sigma_w is identified while the ratio is not '
      '-> degeneracy confirmed; drop one of sigma_k / sigma_w from the tuning.')
