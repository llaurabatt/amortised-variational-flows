"""Main script for running the LALME model."""
#%%
import debugpy
#%%
debugpy.listen(5678)
print('Waiting for debugger')
debugpy.wait_for_client()
print('Debugger attached')
#%%
import os
import warnings

from absl import app
from absl import flags
from absl import logging

import jax
from ml_collections import config_flags, ConfigDict
from functools import partial
import os
import shutil
import sys
import tensorflow as tf
import traceback
import wandb
import yaml

import train_flow
import train_vmp_flow
import sample_mcmc_blackjax as sample_mcmc
import train_vmp_flow_allhp_smallcondval
import train_vmp_flow_smallcondvals

# import sample_mcmc_tfp as sample_mcmc

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=False)

def configdict_to_dict(config_dict):
    """Recursively convert ConfigDict to dict."""
    if isinstance(config_dict, ConfigDict):
        # Convert ConfigDict to dict
        config_dict = {k: configdict_to_dict(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        # Recursively apply to list elements
        config_dict = [configdict_to_dict(v) for v in config_dict]
    return config_dict

def main(_):
  os.environ["WANDB_DIR"] = os.path.abspath(FLAGS.workdir)
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

  config_yaml = FLAGS.config.to_yaml()
  # Write the YAML string to the file
  with open(FLAGS.workdir + '/configs.yaml', 'w') as file:
      file.write(config_yaml)

  if FLAGS.config.method == 'flow':
    train_flow.train_and_evaluate(config=FLAGS.config, workdir=FLAGS.workdir)
  elif FLAGS.config.method == 'vmp_flow':
    train_vmp_flow_smallcondvals.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  elif FLAGS.config.method == 'vmp_flow_allhp':  
    train_vmp_flow_allhp_smallcondval.train_and_evaluate(FLAGS.config, FLAGS.workdir)
   
  elif FLAGS.config.method == 'hp_opt_vmp_flow_allhp':  
    train_opt = partial(train_vmp_flow_allhp_smallcondval.train_and_evaluate,
      config=FLAGS.config, 
      workdir=FLAGS.workdir)
    
    def pagent():
      try:
          train_opt()
      except Exception as e:
          print("Exception! Printing stack trace")
          print(traceback.print_exc(), file=sys.stderr)

    if FLAGS.config.sweep_id:
      wandb.agent(FLAGS.config.sweep_id, function=pagent, count=FLAGS.config.n_sweep_runs, project=FLAGS.config.wandb_project_name)
    else:
      sweep_id = wandb.sweep(sweep=configdict_to_dict(FLAGS.config.sweep_configuration), project=FLAGS.config.wandb_project_name)
      wandb.agent(sweep_id, function=pagent, count=FLAGS.config.n_sweep_runs) 

  elif FLAGS.config.method == 'mcmc':
    # with jax.profiler.trace("/home/llaurabat/tmp/jax-trace"):
    sample_mcmc.sample_and_evaluate(config=FLAGS.config, workdir=FLAGS.workdir)
  else:
    raise ValueError(f'Unknown method {FLAGS.config.method}')


# TODO: Remove when Haiku stop producing "jax.tree_leaves is deprecated" warning
warnings.simplefilter(action='ignore', category=FutureWarning)

# On GPU this may be needed for jax to find the accelerator
os.environ["PATH"] = '/usr/local/cuda/bin:' + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = '/usr/local/cuda/lib64:' + os.environ[
    "LD_LIBRARY_PATH"]

if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
