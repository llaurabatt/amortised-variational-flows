"""95% HPD coverage of held-out validation anchors (reviewer point: calibration).

For each val anchor (true location known, treated as floating during inference):
fit a 2D KDE to its posterior location samples, find the density threshold that
encloses `hdi_prob` of the samples (sample-based HPD), and check whether the
density AT THE TRUE LOCATION clears it. Coverage = fraction of anchors covered;
well-calibrated posteriors give coverage ~= hdi_prob. Also reports the HPD
region area (km^2) per anchor, reusable for the floating-profile summaries.

Needs the posterior samples netcdf (InferenceData with posterior['loc_floating']
and coord LP_floating including the val-anchor LPs) produced by the forward pass
of a trained flow. Run on CPU: export JAX_PLATFORMS=cpu

Example:
  python hpd_coverage.py --selftest
  python hpd_coverage.py \
    --config=configs/all_items_flow_nsf_vmp_flow_3ELBO_40val_b10_w_a_lk_eta.py \
    --az_path=<workdir>/samples/lalme_az.nc --output_dir=<workdir>
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import sys
import pathlib

import numpy as np
import scipy.stats

from absl import flags
from ml_collections import config_flags

# Box scaling: the unit box spans 200 km (region 200x180 km), so 1 unit^2 = 200^2 km^2.
KM_PER_UNIT = 200.0

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', None, 'Experiment config (defines the anchor split).')
flags.DEFINE_string('az_path', '', 'netcdf with posterior samples (InferenceData).')
flags.DEFINE_string('output_dir', '', 'Where to write the per-anchor CSV.')
flags.DEFINE_float('hdi_prob', 0.95, 'HPD probability level.')
flags.DEFINE_integer('grid_n', 200, 'Grid resolution per axis for HPD area integration.')
flags.DEFINE_bool('selftest', False, 'Run synthetic checks instead of real data.')
FLAGS(sys.argv)


def hpd_membership(samples: np.ndarray,
                   truth: np.ndarray,
                   hdi_prob: float = 0.95,
                   grid_n: int = 200):
  """Sample-based HPD test for one profile.

  samples: (N, 2) posterior location samples; truth: (2,) known location.
  Returns dict with covered (bool), density at truth, HPD density threshold,
  and the HPD region area in scaled units^2.
  """
  kde = scipy.stats.gaussian_kde(samples.T)
  # Density threshold enclosing hdi_prob of the posterior mass: the density
  # value such that (1 - hdi_prob) of the samples sit below it.
  sample_densities = kde(samples.T)
  threshold = np.quantile(sample_densities, 1.0 - hdi_prob)
  density_truth = float(kde(truth.reshape(2, 1))[0])

  # HPD area by grid integration over the unit box (y range covers the box's
  # used part; integrate over the samples' bounding box padded by 3 bandwidths).
  pad = 3.0 * np.sqrt(np.diag(kde.covariance)).max()
  x_lo, y_lo = samples.min(axis=0) - pad
  x_hi, y_hi = samples.max(axis=0) + pad
  xs = np.linspace(x_lo, x_hi, grid_n)
  ys = np.linspace(y_lo, y_hi, grid_n)
  xx, yy = np.meshgrid(xs, ys)
  dens = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(grid_n, grid_n)
  cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
  area = float((dens >= threshold).sum() * cell_area)

  return {
      'covered': bool(density_truth >= threshold),
      'density_truth': density_truth,
      'threshold': float(threshold),
      'hpd_area': area,
      'hpd_area_km2': area * KM_PER_UNIT**2,
  }


def _selftest():
  rng = np.random.default_rng(0)
  n = 2000
  # 1) truth at the mode of a tight gaussian -> covered
  s = rng.normal([0.5, 0.5], 0.02, size=(n, 2))
  r = hpd_membership(s, np.array([0.5, 0.5]))
  assert r['covered'], 'truth at mode should be covered'
  # 2) truth far outside -> not covered
  r2 = hpd_membership(s, np.array([0.9, 0.9]))
  assert not r2['covered'], 'distant truth should not be covered'
  # 3) bimodal: truth at second mode covered, midpoint gap not
  s3 = np.concatenate([rng.normal([0.3, 0.3], 0.02, size=(n // 2, 2)),
                       rng.normal([0.7, 0.7], 0.02, size=(n // 2, 2))])
  assert hpd_membership(s3, np.array([0.7, 0.7]))['covered']
  assert not hpd_membership(s3, np.array([0.5, 0.5]))['covered']
  # 4) empirical calibration: over many repeats, a true draw from the same
  #    gaussian should be covered ~95% of the time
  hits = sum(
      hpd_membership(rng.normal(0.5, 0.05, size=(500, 2)),
                     rng.normal(0.5, 0.05, size=2))['covered']
      for _ in range(200))
  rate = hits / 200
  assert 0.90 <= rate <= 0.99, f'calibration selftest rate {rate}'
  # 5) area sanity: 95% HPD of an isotropic gaussian ~ pi * (2.448*sigma)^2
  area = hpd_membership(rng.normal([0.5, 0.5], 0.05, size=(5000, 2)),
                        np.array([0.5, 0.5]))['hpd_area']
  expect = np.pi * (2.448 * 0.05)**2
  assert abs(area - expect) / expect < 0.15, f'area {area} vs expected {expect}'
  print(f'selftest OK (calibration rate {rate:.3f}, '
        f'gaussian HPD area {area:.4f} vs theory {expect:.4f})')


def main():
  if FLAGS.selftest:
    _selftest()
    return

  assert FLAGS.az_path and FLAGS.output_dir and FLAGS.config is not None
  import jax
  # Same pin as main.py: the anchor split must reproduce the training runs
  # (threefry_partitionable flips jax.random.choice results).
  jax.config.update('jax_threefry_partitionable', False)
  import arviz as az
  from train_flow_allhp import load_data

  config = FLAGS.config
  lalme_dataset = load_data(prng_key=jax.random.PRNGKey(0), config=config)
  # Splits follow the trainer: [anchor_train, anchor_val, anchor_test, floating]
  LPs = np.split(lalme_dataset['LP'],
                 np.cumsum(lalme_dataset['num_profiles_split']))[:-1]
  lp_val = np.array(LPs[1])
  assert lp_val.size > 0, 'config has no val anchors'
  loc_by_lp = {int(lp): lalme_dataset['loc'][i]
               for i, lp in enumerate(lalme_dataset['LP'])}

  lalme_az = az.from_netcdf(FLAGS.az_path)
  post = lalme_az.posterior['loc_floating']

  rows = []
  for lp in lp_val:
    samples = np.array(post.sel(LP_floating=int(lp)))  # (chain, draw, 2)
    samples = samples.reshape(-1, 2)
    res = hpd_membership(samples, np.asarray(loc_by_lp[int(lp)]),
                         hdi_prob=FLAGS.hdi_prob, grid_n=FLAGS.grid_n)
    rows.append({'LP': int(lp), **res})
    print(f"LP {lp}: covered={res['covered']} "
          f"area={res['hpd_area_km2']:.0f} km^2")

  coverage = np.mean([r['covered'] for r in rows])
  out = pathlib.Path(FLAGS.output_dir)
  out.mkdir(parents=True, exist_ok=True)
  import csv
  # stem of the samples file (e.g. lalme_az_eta_0.315) keeps runs at different
  # etas from overwriting each other
  az_stem = pathlib.Path(FLAGS.az_path).stem
  fname = out / f'hpd_coverage_{int(FLAGS.hdi_prob * 100)}_anchor_val_{az_stem}.csv'
  with open(fname, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  print(f'\nCOVERAGE: {sum(r["covered"] for r in rows)}/{len(rows)} '
        f'= {coverage:.3f} (target {FLAGS.hdi_prob})')
  print(f'saved {fname}')


if __name__ == '__main__':
  main()
