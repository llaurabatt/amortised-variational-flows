"""Conditional Masked coupling bijector."""

import distrax
from distrax._src.utils import math

import jax.numpy as jnp

from modularbayes import ConditionalBijector
from modularbayes._src.typing import (Array, BijectorParams, Callable, List,
                                      Tuple)


class EtaConditionalMaskedCouplingIntegrated(ConditionalBijector, distrax.MaskedCoupling):
  """Coupling bijector that uses a mask to specify which inputs are transformed
  and can input an additional context variable
  """

  def __init__(self, 
               **kwargs):
    super().__init__(**kwargs)

  def forward_and_log_det(
      self,
      x: Array,
      context: List[Array],
  ) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|.

    Args:
      x: Array
      context: List of two elements:
        -eta: an array
        -Context: array, could be None
    """
    self._check_forward_input_shape(x)
    masked_x = jnp.where(self._event_mask, x, 0.)
  
    if ((context[0] is None) and(context[1] is None)):
      conditioner_input = masked_x

    elif context[0] is None:
      conditioner_input = jnp.concatenate([
          masked_x,
          jnp.broadcast_to(context[1], x.shape[:-1] + (context[1].shape[-1],)),
      ],
                                          axis=-1)
    elif context[1] is None:
      conditioner_input = jnp.concatenate([
          masked_x,
          jnp.broadcast_to(context[0], x.shape[:-1] + (context[0].shape[-1],)),
      ],
                                          axis=-1)
    else:
      conditioner_input = jnp.concatenate([
          masked_x,
          jnp.broadcast_to(context[1], x.shape[:-1] + (context[1].shape[-1],)),
          jnp.broadcast_to(context[0], x.shape[:-1] + (context[0].shape[-1],)),
      ],
                                          axis=-1)

    params = self._conditioner(conditioner_input) 

    y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
    y = jnp.where(self._event_mask, x, y0)
    logdet = math.sum_last(
        jnp.where(self._mask, 0., log_d),
        self._event_ndims - self._inner_event_ndims)
    return y, logdet

  def inverse_and_log_det(
      self,
      y: Array,
      context: List[Array],
  ) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    masked_y = jnp.where(self._event_mask, y, 0.)

    if ((context[0] is None) and(context[1] is None)):
      conditioner_input = masked_y

    elif context[0] is None:
      conditioner_input = jnp.concatenate([
          masked_y,
          jnp.broadcast_to(context[1], x.shape[:-1] + (context[1].shape[-1],)),
      ],
                                          axis=-1)
    elif context[-1] is None:
      conditioner_input = jnp.concatenate([
          masked_y,
          jnp.broadcast_to(context[0], x.shape[:-1] + (context[0].shape[-1],)),
      ],
                                          axis=-1)
    else:
      conditioner_input = jnp.concatenate([
          masked_y,
          jnp.broadcast_to(context[1], x.shape[:-1] + (context[1].shape[-1],)),
          jnp.broadcast_to(context[0], x.shape[:-1] + (context[0].shape[-1],)),
      ],
                                          axis=-1)

    params = self._conditioner(conditioner_input) 
 
    x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
    x = jnp.where(self._event_mask, y, x0)
    logdet = math.sum_last(
        jnp.where(self._mask, 0., log_d),
        self._event_ndims - self._inner_event_ndims)
    return x, logdet
