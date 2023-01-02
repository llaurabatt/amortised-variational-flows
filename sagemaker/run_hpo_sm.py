"""
Hyper-Parameter Optimization (HOP) with Synetune and Sagemaker.
"""
import os
import pathlib

from absl import app
from absl import flags
from absl import logging

import boto3

from sagemaker_jax import JaxEstimator

from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune import Tuner, StoppingCriterion, config_space

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='config_fn',
    default='8_items_flow_nsf_vmp_flow.py',
    help='Path to the config file with parameters to be optimized.' +
    'The path is relative to the main repo directory.',
)
flags.DEFINE_string(
    name='smi_method',
    default='flow',
    help='Method used for SMI.',
)


def _get_ecr_image(container_name: str):
  client = boto3.client("sts")
  account = client.get_caller_identity()["Account"]

  my_session = boto3.session.Session()
  region = my_session.region_name

  ecr_image = f"{account}.dkr.ecr.{region}.amazonaws.com/{container_name}"

  return ecr_image


def hpo_syne_sm(config_fn: str, smi_method: str) -> None:
  """Hyper-parameter optimisation with Synetune and Sagemaker."""

  training_job_name = config_fn.replace('.py',
                                        '').replace('/', '-').replace('_', '-')

  mode = "min"
  searcher = 'bayesopt'
  n_workers = 10
  stop_criterion = StoppingCriterion(max_num_trials_started=70)
  container_name = "jax-tf:latest"
  instance_type = "ml.p3.2xlarge"

  if smi_method == 'flow':
    metric = 'mean_dist_anchor_val'
    config_space_dict = {
        'config':
            config_fn,
        'workdir':
            '/opt/ml/model/',
        'undefok':
            'st_checkpoint_dir,st_instance_count,st_instance_type',
        "config.log_img_steps":
            -1,
        "config.synetune_metric":
            metric,
        'config.eta_profiles_floating':
            config_space.uniform(0., 1.0),
        "config.kernel_kwargs.amplitude":
            config_space.uniform(0.03, 1.0),
        "config.kernel_kwargs.length_scale":
            config_space.uniform(0.03, 1.0),
        "config.optim_kwargs.lr_schedule_kwargs.peak_value":
            config_space.loguniform(1e-4, 1e-1),
        "config.optim_kwargs.lr_schedule_kwargs.decay_rate":
            config_space.uniform(0.1, 1.0),
        'config.optim_kwargs.lr_schedule_kwargs.transition_steps':
            10_000,
        "config.training_steps":
            50_000,
        "config.eval_steps":
            5_000,
        "config.checkpoint_steps":
            -1,
    }
  elif smi_method == 'vmp_flow':
    metric = 'mean_dist_anchor_val_min'
    config_space_dict = {
        'config':
            config_fn,
        'workdir':
            '/opt/ml/model/',
        'undefok':
            'st_checkpoint_dir,st_instance_count,st_instance_type',
        "config.log_img_steps":
            -1,
        "config.log_img_at_end":
            False,
        "config.synetune_metric":
            metric,
        "config.kernel_kwargs.amplitude":
            config_space.uniform(0.03, 1.0),
        "config.kernel_kwargs.length_scale":
            config_space.uniform(0.03, 1.0),
        "config.optim_kwargs.lr_schedule_kwargs.peak_value":
            config_space.loguniform(1e-5, 1e-2),
        "config.optim_kwargs.lr_schedule_kwargs.decay_rate":
            config_space.uniform(0.1, 1.0),
        'config.optim_kwargs.lr_schedule_kwargs.transition_steps':
            10000,
        "config.training_steps":
            30000,
        "config.eval_steps":
            5000,
        "config.checkpoint_steps":
            -1,
    }
  else:
    raise ValueError('smi_method must be either "flow" or "vmp_flow".')

  sm_estimator = JaxEstimator(
      image_uri=_get_ecr_image(container_name=container_name),
      role=get_execution_role(),
      instance_count=1,
      base_job_name=training_job_name,
      source_dir=str(pathlib.Path(__file__).parent.parent),
      entry_point='main.py',
      instance_type=instance_type,
      # max_run=10 * 60,  # Set MaxRuntimeInSeconds for Training Jobs
  )

  # Run experiments in Sagemaker
  trial_backend = SageMakerBackend(
      sm_estimator=sm_estimator,
      metrics_names=[metric],  # names of metrics to track
      # inputs={'datadir': 's3://datasets/'},
  )

  # Random search without stopping
  scheduler = FIFOScheduler(
      config_space=config_space_dict,
      searcher=searcher,
      metric=metric,
      mode=mode,
      random_seed=123,
  )

  tuner = Tuner(
      trial_backend=trial_backend,
      scheduler=scheduler,
      stop_criterion=stop_criterion,
      n_workers=n_workers,
      sleep_time=5.0,
      tuner_name=training_job_name,
  )

  tuner.run()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # log to a file
  if FLAGS.log_dir:
    if not os.path.exists(FLAGS.log_dir):
      os.makedirs(FLAGS.log_dir)
    logging.get_absl_handler().use_absl_log_file()

  logging.info("HPO over parameters defined by: config/%s", FLAGS.config_fn)
  hpo_syne_sm(config_fn=FLAGS.config_fn, smi_method=FLAGS.smi_method)


if __name__ == "__main__":
  app.run(main)
