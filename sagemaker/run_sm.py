"""
Run experiments on Sagemaker.
"""

import pathlib

from absl import app
from absl import logging

import boto3

from sagemaker_jax import JaxEstimator


def _get_ecr_image():
  """Get the ECR container to use on training jobs."""
  client = boto3.client("sts")
  account = client.get_caller_identity()["Account"]

  my_session = boto3.session.Session()
  region = my_session.region_name

  container_name = "jax-tf"
  tag = ':latest'

  ecr_image = f"{account}.dkr.ecr.{region}.amazonaws.com/{container_name + tag}"

  return ecr_image


def _get_execution_role():
  """Get the execution role for SageMaker."""

  client = boto3.client("sts")
  account = client.get_caller_identity()["Account"]
  exec_role = (f"arn:aws:iam::{account}:role/service-role/" +
               "AmazonSageMaker-ExecutionRole-20220403T231924")
  return exec_role


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # exec_role = sagemaker.get_execution_role()
  exec_role = _get_execution_role()

  ### SMI with single eta fit separately ###
  experiment_names = []
  experiment_names += ['flow_mf']

  eta_values = [1.]

  logging.info('Sending training jobs to Sagemaker...')
  for experiment_ in experiment_names:
    for eta_ in eta_values:
      training_job_name = 'spatial-smi-' + str(experiment_) + f"-eta{eta_:.3f}"
      training_job_name = training_job_name.replace('_', '-').replace('.', 'p')
      logging.info('\t %s', training_job_name)

      sm_estimator = JaxEstimator(
          image_uri=_get_ecr_image(),
          role=exec_role,
          instance_count=1,
          base_job_name=training_job_name,
          source_dir=str(pathlib.Path(__file__).parent.parent),
          entry_point='main.py',
          instance_type="ml.p3.2xlarge",
          hyperparameters={
              'config': f'configs/{experiment_}.py',
              'workdir': '/opt/ml/model/',
              'config.eta_profiles_floating': eta_,
          },
      )
      sm_estimator.fit(wait=False,)
  logging.info('All training jobs send succesfully!')


if __name__ == "__main__":
  app.run(main)
