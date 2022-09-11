"""Main script for running the LALME model."""
import os
import warnings

from absl import app
from absl import flags
from absl import logging

import jax
from ml_collections import config_flags
import tensorflow as tf

import train_flow
import train_vmp_flow
import run_mcmc

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # log to a file
  if FLAGS.log_dir:
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    logging.get_absl_handler().use_absl_log_file()

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  tf.config.experimental.set_visible_devices([], 'TPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  logging.info('JAX device count: %r', jax.device_count())

  if FLAGS.config.method == 'flow':
    if FLAGS.config.iterate_smi_eta == ():
      train_flow.train_and_evaluate(FLAGS.config, FLAGS.workdir)
    else:
      for eta_floating in FLAGS.config.iterate_smi_eta:
        FLAGS.config.flow_kwargs.smi_eta.update({
            'profiles_floating': eta_floating,
        })
        train_flow.train_and_evaluate(FLAGS.config,
                                      FLAGS.workdir + f"_{eta_floating:.3f}")
  elif FLAGS.config.method == 'vmp_flow':
    train_vmp_flow.train_and_evaluate(FLAGS.config, FLAGS.workdir)

  elif FLAGS.config.method == 'mcmc':
    if FLAGS.config.iterate_smi_eta == ():
      run_mcmc.sample_and_evaluate(FLAGS.config, FLAGS.workdir)
    else:
      for eta_floating in FLAGS.config.iterate_smi_eta:
        FLAGS.config.smi_eta.update({
            'profiles_floating': eta_floating,
        })
        run_mcmc.sample_and_evaluate(FLAGS.config,
                                     FLAGS.workdir + f"_{eta_floating:.3f}")

# TODO: Remove when Haiku stop producing "jax.tree_leaves is deprecated" warning
warnings.simplefilter(action='ignore', category=FutureWarning)

# On GPU this may be needed for jax to find the accelerator
os.environ["PATH"] = '/usr/local/cuda/bin:' + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = '/usr/local/cuda/lib64:' + os.environ["LD_LIBRARY_PATH"]

if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
