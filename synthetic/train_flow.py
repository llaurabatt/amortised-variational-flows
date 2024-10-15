"""A simple example of a flow model trained on Epidemiology data."""
import pathlib

from absl import logging

import numpy as np
from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

import flows
import log_prob_fun_integrated
import plot
from data import load_synthetic

from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, initial_state_ckpt, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict,
                                      IntLike, List, Optional, PRNGKey,
                                      Sequence, SmiEta, SummaryWriter, Tuple,
                                      Union)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def load_dataset(n_groups: int, 
                 n_obs: int,
                 seed: int = 3) -> Dict[str, Array]:
  """Load synthetic data."""

  data_array, true_params, fig = load_synthetic(seed=seed, n_groups=n_groups, n_obs=n_obs)

  return data_array.to_numpy(), true_params, fig


def make_optimizer(
    lr_schedule_name,
    lr_schedule_kwargs,
    grad_clip_value,
) -> optax.GradientTransformation:
  """Define optimizer to train the Flow."""
  schedule = getattr(optax, lr_schedule_name)(**lr_schedule_kwargs)

  optimizer = optax.chain(*[
      optax.clip_by_global_norm(max_norm=grad_clip_value),
      optax.adabelief(learning_rate=schedule),
  ])
  return optimizer
