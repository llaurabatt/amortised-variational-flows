"""Paper Fig 1b: anchor + fit-technique locations per form of one item.

Data-only by default; pass --az_path (netcdf of posterior samples, as written
by lalme_az_from_samples) to overlay posterior HDI contours of loc_floating.

Run on CPU: export JAX_PLATFORMS=cpu

Example:
  python plot_fig1b.py --output_dir=$HOME/mount/vmp-output/figures
  python plot_fig1b.py --output_dir=... --az_path=<workdir>/samples/lalme_az.nc
"""
import os
# env TeX Live is incomplete (no latex.fmt); use the working SYSTEM TeX at /usr/bin
os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')
import sys
import pathlib

import numpy as np
import matplotlib as mpl
from absl import flags

import arviz as az

from data import load_lalme
from plot import plot_item_locations
from misc import clean_filename

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', None, 'Directory to save the figure.')
flags.DEFINE_string('dataset_id', 'coarsen_all_items', 'LALME dataset id.')
flags.DEFINE_string('item', 'such/', 'Item to plot.')
flags.DEFINE_list('forms', ['SWC', 'SIC', 'SWILC', 'SLIK'],
                  'Forms of the item, one panel each.')
flags.DEFINE_string('az_path', '',
                    'Optional netcdf with posterior samples (InferenceData); '
                    'overlays loc_floating HDI contours.')
flags.DEFINE_list('hdi_probs', ['0.95'], 'HDI levels for the contours.')
flags.DEFINE_list('formats', ['png'],
                  'Output formats. Paper copy in figures/ uses png,pdf; '
                  'per-run workdir copies use png only.')
flags.mark_flags_as_required(['output_dir'])
FLAGS(sys.argv)

mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['text.usetex'] = True  # uses SYSTEM /usr/bin latex (PATH set at top)

lalme_dataset = load_lalme(dataset_id=FLAGS.dataset_id)

# Scale locations to the unit box, preserving aspect ratio -- same
# transformation as load_data in the trainers (train_flow_allhp.py).
loc = lalme_dataset['loc'].astype(float)
loc = loc - loc.min(axis=0, keepdims=True)
loc = loc / loc.max()
lalme_dataset['loc'] = loc

lalme_az = az.from_netcdf(FLAGS.az_path) if FLAGS.az_path else None

fig, _ = plot_item_locations(
    lalme_dataset=lalme_dataset,
    item=FLAGS.item,
    forms=FLAGS.forms,
    lalme_az=lalme_az,
    hdi_probs=[float(p) for p in FLAGS.hdi_probs],
)

out = pathlib.Path(FLAGS.output_dir)
out.mkdir(parents=True, exist_ok=True)
suffix = 'posterior' if lalme_az is not None else 'data_only'
fname = clean_filename(
    f"lalme_fig1b_{FLAGS.item}_{'_'.join(FLAGS.forms)}_{suffix}")
for ext in FLAGS.formats:
  fig.savefig(out / f'{fname}.{ext}', dpi=300, bbox_inches='tight')
  print(f'saved {out / f"{fname}.{ext}"}')
