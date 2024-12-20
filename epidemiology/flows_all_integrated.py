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


class MeanField(hk.Module):
  """Auxiliary Module to assign loc and log_scale to a module.

  These parameters could be directly defined within the mean_field() function,
  but the module makes them discoverable by hk.experimental.tabulate"""

  def __init__(
      self,
      flow_dim: int,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.flow_dim = flow_dim

  def __call__(self,):
    event_shape = (self.flow_dim,)

    loc = hk.get_parameter("loc", event_shape, init=jnp.zeros)
    # log_scale = jnp.zeros(event_shape)
    log_scale = hk.get_parameter("log_scale", event_shape, init=jnp.zeros)

    return loc, log_scale


def mean_field_phi(
    phi_dim: int,
    **_,
) -> distrax.Transformed:
  """Creates a Mean Field Flow."""

  flow_dim = phi_dim
  event_shape = (flow_dim,)

  flow_layers = []

  # Layer 1: Trainable Affine transformation
  mf_module = MeanField(flow_dim=flow_dim) #? why doesn't this run standalone?
  loc, log_scale = mf_module()
  flow_layers.append(
      distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))

      # layer multiplying some x by scale and summing loc. Try out e.g., with
      # distrax.ScalarAffine().forward_and_log_det(jnp.array([10.])

      # A block bijector applies a bijector (here scalar affine)
      # to a k-dimensional array of events, but considers that array of events to be a single event. In practical terms, this
      # means that the log det Jacobian will be summed over its last k dimensions.
      # For example, consider a scalar bijector (such as scalar affine or sigmoid) that operates on
      # scalar events. We may want to apply this bijector identically to a 4D array of
      # shape [N, H, W, C] representing a sequence of N images. Doing so naively will
      # produce a log det Jacobian of shape [N, H, W, C], because the scalar bijector
      # will assume scalar events and so all 4 dimensions will be considered as batch.
      # To promote the scalar bijector to a "block scalar" that operates on the 3D
      # arrays can be done by `Block(bijector, ndims=3)`. Then, applying the block
      # bijector will produce a log det Jacobian of shape [N] as desired.
      # In general, suppose `bijector` operates on n-dimensional events. Then,
      # `Block(bijector, k)` will promote `bijector` to a block bijector that
      # operates on (k + n)-dimensional events, summing the log det Jacobian over its
      # last k dimensions. In practice, this means that the last k batch dimensions
      # will be turned into event dimensions.

  # Last Layer: Map values to parameter domain
  # phi goes to [0,1]
  flow_layers.append(distrax.Block(distrax.Sigmoid(), 1))

  # Chain all layers together
  flow = distrax.Chain(flow_layers[::-1]) # it takes Sequence[BijectorLike]

  #   """Composition of a sequence of bijectors into a single bijector.
  # Bijectors are composable: if `f` and `g` are bijectors, then `g o f` is also
  # a bijector. Given a sequence of bijectors `[f1, ..., fN]`, this class
  # implements the bijector defined by `fN o ... o f1`.
  # NOTE: the bijectors are applied in reverse order from the order they appear in
  # the sequence. For example, consider the following code where `f` and `g` are
  # two bijectors:
  # ```
  # layers = []
  # layers.append(f)
  # layers.append(g)
  # bijector = distrax.Chain(layers)
  # y = bijector.forward(x)
  # ```
  # The above code will transform `x` by first applying `g`, then `f`, so that
  # `y = f(g(x))`.

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = modularbayes.Transformed(base_distribution, flow)
  # this is T(e). You can sample x = T(e) from it and evaluate 
  # x at p_x i.e. log(p_e*|J|^(-1))

  return q_distr


def mean_field_theta(
    theta_dim: int,
    **_,
) -> modularbayes.ConditionalTransformed:
  """Creates a Mean Field Flow."""

  flow_dim = theta_dim

  event_shape = (flow_dim,)

  flow_layers = []

  # Layer 1: Trainable Affine transformation
  mf_module = MeanField(flow_dim=flow_dim)
  loc, log_scale = mf_module()
  flow_layers.append(
      distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))
  # flow_layers.append(tfb.Shift(loc)(tfb.Scale(log_scale=log_scale)))
  

  # Last layer: Map values to parameter domain
  # theta1 goes to [-Inf,Inf]
  # theta2 goes to [0,Inf]
  block_bijectors = [
      distrax.Block(tfb.Identity(), 1),
      distrax.Block(tfb.Softplus(), 1),
  ]
  block_sizes = [
      1,
      1,
  ]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = modularbayes.ConditionalTransformed(base_distribution, flow)

  return q_distr


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


def nsf_phi(
    phi_dim: int,
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
    range_min: float = 0.,
    range_max: float = 1.,
    **_,
) -> distrax.Transformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  flow_dim = phi_dim

  event_shape = (flow_dim,)

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
    layer = distrax.MaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes,
            num_bijector_params=num_bijector_params,
            name='conditioner_phi',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # phi goes to [0,1]
  flow_layers.append(distrax.Block(distrax.Sigmoid(), 1))

  flow = distrax.Chain(flow_layers[::-1])

  # base_distribution = distrax.Independent(
  #     distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
  #     reinterpreted_batch_ndims=len(event_shape))

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  return modularbayes.Transformed(base_distribution, flow)


def nsf_theta(
    theta_dim: int,
    num_layers: int,
    hidden_sizes: Sequence[int],
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

  flow_dim = theta_dim
  event_shape = (flow_dim,)

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
    layer = modularbayes.ConditionalMaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=CouplingConditioner(
            # input_dim=math.prod(event_shape),
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes,
            num_bijector_params=num_bijector_params,
            name='conditioner_theta',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # theta1 goes to [-Inf,Inf]
  # theta2 goes to [0,Inf]
  block_bijectors = [
      distrax.Block(tfb.Identity(), 1),
      distrax.Block(tfb.Softplus(), 1),
  ]
  block_sizes = [
      1,
      1,
  ]
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


def meta_nsf_phi(
    phi_dim: int,
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

  flow_dim = phi_dim #13 + 4

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
            name='conditioner_phi_hps',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # phi goes to [0,1]
  flow_layers.append(distrax.Block(distrax.Sigmoid(), 1))

  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  # base_distribution = distrax.Independent(
  #     distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
  #     reinterpreted_batch_ndims=len(event_shape))

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  return modularbayes.ConditionalTransformed(base_distribution, flow)


def meta_nsf_theta(
    theta_dim: int,
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

  flow_dim = theta_dim
  event_shape = (flow_dim,)

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
            name='conditioner_theta_hps',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # theta1 goes to [-Inf,Inf]
  # theta2 goes to [0,Inf]
  block_bijectors = [
      distrax.Block(tfb.Identity(), 1),
      distrax.Block(tfb.Softplus(), 1),
  ]
  block_sizes = [
      1,
      1,
  ]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))
  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = modularbayes.ConditionalTransformed(base_distribution, flow)

  return q_distr


def split_flow_phi(
    samples: Array,
    phi_dim: int,
    **_,
) -> Dict[str, Any]:
  """Get model parameters by splitting samples from the flow."""

  flow_dim = phi_dim

  assert samples.ndim == 2
  assert samples.shape[-1] == flow_dim

  samples_dict = {}

  # phi: Human-Papilloma virus (HPV) prevalence on each population
  samples_dict['phi'] = samples

  return samples_dict


def split_flow_theta(
    samples: Array,
    theta_dim: int,
    is_aux: bool,
    **_,
) -> Dict[str, Any]:
  """Get model parameters by splitting samples from the flow."""

  flow_dim = theta_dim

  assert samples.ndim == 2
  assert samples.shape[-1] == flow_dim

  samples_dict = {}

  # theta: Intercept and slope of the prevalence-incidence model
  samples_dict['theta' + ('_aux' if is_aux else '')] = samples

  return samples_dict
