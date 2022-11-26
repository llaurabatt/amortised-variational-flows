"""
Run experiments on Sagemaker.
"""

import pathlib

from typing import Optional, List
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


def send_experiment_to_sm(
    experiment_names: List[str],
    eta_values: Optional[List[float]] = None,
    hyperparameters_extra: Optional[dict] = None,
):
  """Send a list of experiments to be executed in AWS Sagemaker."""

  # exec_role = sagemaker.get_execution_role()
  exec_role = _get_execution_role()

  logging.info('Sending training jobs to Sagemaker...')
  for experiment_ in experiment_names:
    for eta_ in eta_values or [None]:
      hyperparameters = {
          'config': f'configs/{experiment_}.py',
          'workdir': '/opt/ml/model/',
          'log_dir': '/opt/ml/model/log_dir/',
      }
      training_job_name = 'spatial-smi-' + str(experiment_)

      if eta_values is not None:
        training_job_name += f"-eta{eta_:.3f}"
        hyperparameters['config.eta_profiles_floating'] = eta_

      if hyperparameters_extra is not None:
        hyperparameters.update(hyperparameters_extra)

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
          hyperparameters=hyperparameters,
      )
      sm_estimator.fit(wait=False,)
  logging.info('All training jobs send succesfully!')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  eta_values = [0.001, 0.25, 0.5, 0.75, 1.]

  # SMI via MCMC
  send_experiment_to_sm(
      experiment_names=[
          '8_items_mcmc',
          # 'all_items_mcmc',
      ],
      eta_values=eta_values,
  )

  # Variational SMI with single eta
  send_experiment_to_sm(
      experiment_names=[
          '8_items_flow_mf',
          '8_items_flow_nsf',
          'all_items_flow_mf',
          'all_items_flow_nsf',
      ],
      eta_values=eta_values,
  )

  # Variational SMI across multiple etas via Meta-posterior
  send_experiment_to_sm(experiment_names=[
      '8_items_flow_nsf_vmp_flow',
      'all_items_flow_nsf_vmp_flow',
  ])

  # SMI via MCMC, measure timing
  for num_profiles_floating_keep in [10, 20, 40, 80]:
    for num_items_keep in [8, 16, 32, 64]:
      send_experiment_to_sm(
          experiment_names=[
              'all_items_mcmc',
          ],
          hyperparameters_extra={
              'config.num_profiles_floating_keep': num_profiles_floating_keep,
              'config.num_items_keep': num_items_keep,
          },
      )


if __name__ == "__main__":
  app.run(main)
