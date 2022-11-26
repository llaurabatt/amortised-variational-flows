# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""
Custom Framework Estimator for JAX
"""
from sagemaker.estimator import Framework
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT

import tensorflow as tf

class JaxEstimator(Framework):

  def __init__(
      self,
      entry_point,
      source_dir=None,
      hyperparameters=None,
      image_uri=None,
      tf_version=tf.__version__,
      py_version="py3",
      **kwargs,
  ):
    super().__init__(
        entry_point=entry_point,
        source_dir=source_dir,
        hyperparameters=hyperparameters,
        image_uri=image_uri,
        **kwargs,
    )
    self.framework_version = tf_version
    self.py_version = py_version

  def create_model(
      self,
      role=None,
      vpc_config_override=VPC_CONFIG_DEFAULT,
      entry_point=None,
      source_dir=None,
      dependencies=None,
      **kwargs,
  ):
    """Creates ``TensorFlowModel`` object.
    
    To be used for creating SageMaker model entities.
    """
    
    kwargs["name"] = self._get_or_create_name(kwargs.get("name"))

    if "enable_network_isolation" not in kwargs:
      kwargs["enable_network_isolation"] = self.enable_network_isolation()

    return TensorFlowModel(
        model_data=self.model_data,
        role=role or self.role,
        container_log_level=self.container_log_level,
        framework_version=self.framework_version,
        sagemaker_session=self.sagemaker_session,
        vpc_config=self.get_vpc_config(vpc_config_override),
        entry_point=entry_point,
        source_dir=source_dir,
        dependencies=dependencies,
        **kwargs,
    )