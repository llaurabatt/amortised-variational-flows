"""Define normalizing flows for the Epidemiology model."""

import math

from jax import numpy as jnp
import haiku as hk
import distrax
from tensorflow_probability.substrates import jax as tfp

import modularbayes
from modularbayes import EtaConditionalMaskedCouplingIntegrated
from modularbayes._src.typing import Any, Array, Dict, Optional, Sequence

tfb = tfp.bijectors
tfd = tfp.distributions


class CouplingConditioner(hk.Module):

  def __init__(
      self,
      output_dim: int,
      hidden_sizes: Sequence[int],
      num_bijector_params: int,
      name: Optional[str] = 'nsf_conditioner',
  ):
    super().__init__(name=name)
    self.output_dim = output_dim
    self.hidden_sizes = hidden_sizes
    self.num_bijector_params = num_bijector_params

  def __call__(self, inputs):

    out = hk.Flatten(preserve_dims=-1)(inputs)
    out = hk.nets.MLP(self.hidden_sizes, activate_final=True)(out)
    # hidden sizes is "output sizes" arg, i.e. sequence of layer sizes

    # We initialize this linear layer to zero so that the flow is initialized
    # to the identity function.
    out = hk.Linear(
        self.output_dim * self.num_bijector_params,
        w_init=jnp.zeros,
        b_init=jnp.zeros)(
            out)
    out = hk.Reshape(
        (self.output_dim,) + (self.num_bijector_params,), preserve_dims=-1)(
            out)

    return out


def meta_nsf_mu_sigma(
    num_groups: int,
    num_layers: int,
    hidden_sizes_conditioner: Sequence[int],
    num_bins: int,
    range_min: float = 0.,
    range_max: float = 1.,
    **_,
) -> modularbayes.ConditionalTransformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  flow_dim = 2*num_groups

  event_shape = (flow_dim,) #13 + 4

  flow_layers = []

  # Number of parameters required by the bijector (rational quadratic spline)
  num_bijector_params = 3 * num_bins + 1

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=range_min, range_max=range_max)

  # Alternating binary mask.
  mask = jnp.arange(0, math.prod(event_shape)) % 2
  mask = jnp.reshape(mask, event_shape)
  mask = mask.astype(bool)

  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes
  # for a total of `3 * num_bins + 1` parameters.

  for _ in range(num_layers):
    layer = EtaConditionalMaskedCouplingIntegrated(
        mask=mask,
        bijector=bijector_fn,
        conditioner=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner,
            num_bijector_params=num_bijector_params,
            name='conditioner_mu_sigma',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # mu goes to [-Inf,Inf]
  # sigma goes to [0,Inf]
  block_bijectors = [
      distrax.Block(tfb.Identity(), 1),
      distrax.Block(tfb.Softplus(), 1),
  ]
  block_sizes = [num_groups, num_groups]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))


  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  # base_distribution = distrax.Independent(
  #     distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
  #     reinterpreted_batch_ndims=len(event_shape))

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  return modularbayes.ConditionalTransformed(base_distribution, flow)


def split_flow_mu_sigma(
    samples: Array,
    num_groups: int,
    **_,
) -> Dict[str, Any]:
  """Get model parameters by splitting samples from the flow."""

  flow_dim = 2*num_groups # mu's and sigma's

  assert samples.ndim == 2
  assert samples.shape[-1] == flow_dim

  samples_dict = {}

  # beta and tau
  (samples_dict['mu'],
   samples_dict['sigma']) = jnp.split(
       samples, [num_groups], axis=-1)
  
  assert samples_dict['mu'].shape[-1] == samples_dict['sigma'].shape[-1] == num_groups

  return samples_dict
