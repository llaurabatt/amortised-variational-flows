"""Define Normalizing Flows for the LALME model."""

import math

from jax import numpy as jnp
import haiku as hk
import distrax
from tensorflow_probability.substrates import jax as tfp

import modularbayes
from modularbayes._src.typing import Any, Array, Dict, Optional, Sequence, Tuple

tfb = tfp.bijectors
tfd = tfp.distributions


class MeanField(hk.Module):
  """Auxiliary Module to assign loc and log_scale to a module.

  These parameters could be directly defined within the mean_field() function,
  but the module makes them discoverable by hk.experimental.tabulate"""

  def __init__(
      self,
      flow_dim: int,
      name: Optional[str] = 'mean_field',
  ):
    super().__init__(name=name)
    self.flow_dim = flow_dim

  def __call__(self,):
    event_shape = (self.flow_dim,)

    loc = hk.get_parameter("loc", event_shape, init=jnp.zeros)
    # log_scale = jnp.zeros(event_shape)
    log_scale = hk.get_parameter("log_scale", event_shape, init=jnp.zeros)

    return loc, log_scale


def mean_field_global_params(
    num_forms_tuple: Tuple,
    num_base_gps: int,
    num_inducing_points: int,
    **_,
) -> modularbayes.Transformed:
  """Mean Field Flow for the global parameters in the LALME model.

  This distribution includes the "global" parameters of the LALME model:
  (mu, zeta, Gamma_U, W, a).

  The posterior for unknown locations of profiles is treated separately to
  allow using learnt global posteriors even when varing the number of unknown
  locations.
  """
  num_items = len(num_forms_tuple)

  gamma_inducing_dim = num_base_gps * num_inducing_points
  mixing_weights_dim = sum(
      [num_base_gps * num_forms_i for num_forms_i in num_forms_tuple])
  mixing_offset_dim = sum(num_forms_tuple)
  mu_dim = num_items
  zeta_dim = num_items

  flow_dim = (
      gamma_inducing_dim + mixing_weights_dim + mixing_offset_dim + mu_dim +
      zeta_dim)

  event_shape = (flow_dim,)

  flow_layers = []

  # Layer 1: Trainable Affine transformation
  mf_module = MeanField(flow_dim=flow_dim, name='mf_global_params')
  loc, log_scale = mf_module()
  flow_layers.append(
      distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))

  # Layer 2: Map values to parameter domain
  block_bijectors = [
      # gamma: Identity [-Inf,Inf]
      distrax.Block(tfb.Identity(), 1),
      # mixing_weights: Identity [-Inf,Inf]
      distrax.Block(tfb.Identity(), 1),
      # mixing_offset: Identity [-Inf,Inf]
      distrax.Block(tfb.Identity(), 1),
      # mu: Softplus [0,Inf]
      distrax.Block(tfb.Softplus(), 1),
      # zeta: Sigmoid [0,1]
      distrax.Block(distrax.Sigmoid(), 1),
  ]
  block_sizes = [
      gamma_inducing_dim,
      mixing_weights_dim,
      mixing_offset_dim,
      mu_dim,
      zeta_dim,
  ]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = distrax.Chain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = modularbayes.Transformed(base_distribution, flow)

  return q_distr


def mean_field_locations(
    num_profiles: int,
    loc_x_range: Tuple[float],
    loc_y_range: Tuple[float],
    **_,
) -> modularbayes.ConditionalTransformed:
  """Mean Field Flow for the unknown locations of profiles in the LALME model.
  """

  loc_floating_dim = 2 * num_profiles
  flow_dim = loc_floating_dim

  event_shape = (flow_dim,)

  flow_layers = []

  # Layer 1: Trainable Affine transformation
  # loc = jnp.zeros(event_shape)
  mf_module = MeanField(flow_dim=flow_dim, name='mf_locations')
  loc, log_scale = mf_module()
  flow_layers.append(
      distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))

  # Last layer: Map values to parameter domain
  # all locations to the [0,1] square
  flow_layers.append(distrax.Block(distrax.Sigmoid(), 1))
  # profiles x's go to [0,loc_x_max]
  # profiles y's go to [0,loc_y_max]
  if loc_x_range == (0., 1.):
    loc_x_range_bijector = distrax.Block(tfb.Identity(), 1)
  else:
    loc_x_range_bijector = distrax.Block(
        distrax.ScalarAffine(
            shift=loc_x_range[0], scale=loc_x_range[1] - loc_x_range[0]), 1)

  if loc_y_range == (0., 1.):
    loc_y_range_bijector = distrax.Block(tfb.Identity(), 1)
  else:
    loc_y_range_bijector = distrax.Block(
        distrax.ScalarAffine(
            shift=loc_y_range[0], scale=loc_y_range[1] - loc_y_range[0]), 1)

  block_bijectors = [loc_x_range_bijector, loc_y_range_bijector]
  block_sizes = [num_profiles, num_profiles]
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
  """Auxiliary Module to assign loc and log_scale to a module.

  These parameters could be directly defined within the mean_field() function,
  but the module makes them discoverable by hk.experimental.tabulate"""

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


def nsf_global_params(
    num_forms_tuple: Tuple,
    num_base_gps: int,
    num_inducing_points: int,
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
    spline_range: Tuple[float],
    **_,
) -> modularbayes.Transformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  num_items = len(num_forms_tuple)

  gamma_inducing_dim = num_base_gps * num_inducing_points
  mixing_weights_dim = sum(
      [num_base_gps * num_forms_i for num_forms_i in num_forms_tuple])
  mixing_offset_dim = sum(num_forms_tuple)
  mu_dim = num_items
  zeta_dim = num_items

  flow_dim = (
      gamma_inducing_dim + mixing_weights_dim + mixing_offset_dim + mu_dim +
      zeta_dim)

  event_shape = (flow_dim,)

  flow_layers = []

  # Number of parameters required by the bijector (rational quadratic spline)
  num_bijector_params = 3 * num_bins + 1

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=spline_range[0], range_max=spline_range[1])

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
            # input_dim=math.prod(event_shape),
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes,
            num_bijector_params=num_bijector_params,
            name='conditioner_global_params',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # Layer 2: Map values to parameter domain
  block_bijectors = [
      # gamma: Identity [-Inf,Inf]
      distrax.Block(tfb.Identity(), 1),
      # mixing_weights: Identity [-Inf,Inf]
      distrax.Block(tfb.Identity(), 1),
      # mixing_offset: Identity [-Inf,Inf]
      distrax.Block(tfb.Identity(), 1),
      # mu: Softplus [0,Inf]
      distrax.Block(tfb.Softplus(), 1),
      # zeta: Sigmoid [0,1]
      distrax.Block(distrax.Sigmoid(), 1),
  ]
  block_sizes = [
      gamma_inducing_dim,
      mixing_weights_dim,
      mixing_offset_dim,
      mu_dim,
      zeta_dim,
  ]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = distrax.Chain(flow_layers[::-1])

  # base_distribution = distrax.Independent(
  #     distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
  #     reinterpreted_batch_ndims=len(event_shape))

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  return modularbayes.Transformed(base_distribution, flow)


def nsf_locations(
    num_profiles: int,
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
    spline_range: Tuple[float],
    loc_x_range: Tuple[float],
    loc_y_range: Tuple[float],
    **_,
) -> modularbayes.ConditionalTransformed:
  """Creates the Rational Quadratic Flow for the unknown locations of profiles
  in the LALME model.
  """

  flow_dim = 2 * num_profiles

  event_shape = (flow_dim,)

  flow_layers = []

  # # Layer: Affine transformation
  # loc = hk.get_parameter("loc", event_shape, init=jnp.zeros)
  # # loc = 10*jnp.ones(event_shape)
  # log_scale = hk.get_parameter("log_scale", event_shape, init=jnp.zeros)
  # # log_scale = jnp.zeros(event_shape)
  # flow_layers.append(
  #     distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))
  # # flow_layers.append(tfb.Shift(loc)(tfb.Scale(log_scale=log_scale)))

  # Number of parameters required by the bijector (rational quadratic spline)
  num_bijector_params = 3 * num_bins + 1

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=spline_range[0], range_max=spline_range[1])

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
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes,
            num_bijector_params=num_bijector_params,
            name='conditioner_locations',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # all locations to the [0,1] square
  flow_layers.append(distrax.Block(distrax.Sigmoid(), 1))
  # profiles x's go to [0,loc_x_max]
  # profiles y's go to [0,loc_y_max]
  if loc_x_range == (0., 1.):
    loc_x_range_bijector = distrax.Block(tfb.Identity(), 1)
  else:
    loc_x_range_bijector = distrax.Block(
        distrax.ScalarAffine(
            shift=loc_x_range[0], scale=loc_x_range[1] - loc_x_range[0]), 1)

  if loc_y_range == (0., 1.):
    loc_y_range_bijector = distrax.Block(tfb.Identity(), 1)
  else:
    loc_y_range_bijector = distrax.Block(
        distrax.ScalarAffine(
            shift=loc_y_range[0], scale=loc_y_range[1] - loc_y_range[0]), 1)

  block_bijectors = [loc_x_range_bijector, loc_y_range_bijector]
  block_sizes = [num_profiles, num_profiles]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  # base_distribution = distrax.Independent(
  #     distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
  #     reinterpreted_batch_ndims=len(event_shape))

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  return modularbayes.ConditionalTransformed(base_distribution, flow)


def meta_nsf_global_params(
    num_forms_tuple: Tuple,
    num_base_gps: int,
    num_inducing_points: int,
    num_layers: int,
    hidden_sizes_conditioner: Sequence[int],
    hidden_sizes_conditioner_eta: Sequence[int],
    num_bins: int,
    spline_range: Tuple[float],
    **_,
) -> modularbayes.ConditionalTransformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  num_items = len(num_forms_tuple)

  gamma_inducing_dim = num_base_gps * num_inducing_points
  mixing_weights_dim = sum(
      [num_base_gps * num_forms_i for num_forms_i in num_forms_tuple])
  mixing_offset_dim = sum(num_forms_tuple)
  mu_dim = num_items
  zeta_dim = num_items

  flow_dim = (
      gamma_inducing_dim + mixing_weights_dim + mixing_offset_dim + mu_dim +
      zeta_dim)

  event_shape = (flow_dim,)

  flow_layers = []

  # Number of parameters required by the bijector (rational quadratic spline)
  num_bijector_params = 3 * num_bins + 1

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=spline_range[0], range_max=spline_range[1])

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
    layer = modularbayes.EtaConditionalMaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner_eta=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner_eta,
            num_bijector_params=num_bijector_params,
            name='conditioner_eta_global_params',
        ),
        conditioner=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner,
            num_bijector_params=num_bijector_params,
            name='conditioner_global_params',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # Layer 2: Map values to parameter domain
  block_bijectors = [
      # gamma: Identity [-Inf,Inf]
      distrax.Block(tfb.Identity(), 1),
      # mixing_weights: Identity [-Inf,Inf]
      distrax.Block(tfb.Identity(), 1),
      # mixing_offset: Identity [-Inf,Inf]
      distrax.Block(tfb.Identity(), 1),
      # mu: Softplus [0,Inf]
      distrax.Block(tfb.Softplus(), 1),
      # zeta: Sigmoid [0,1]
      distrax.Block(distrax.Sigmoid(), 1),
  ]
  block_sizes = [
      gamma_inducing_dim,
      mixing_weights_dim,
      mixing_offset_dim,
      mu_dim,
      zeta_dim,
  ]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  # base_distribution = distrax.Independent(
  #     distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
  #     reinterpreted_batch_ndims=len(event_shape))

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  return modularbayes.ConditionalTransformed(base_distribution, flow)


def meta_nsf_locations(
    num_profiles: int,
    num_layers: int,
    hidden_sizes_conditioner: Sequence[int],
    hidden_sizes_conditioner_eta: Sequence[int],
    num_bins: int,
    spline_range: Tuple[float],
    loc_x_range: Tuple[float],
    loc_y_range: Tuple[float],
    **_,
) -> modularbayes.ConditionalTransformed:
  """Creates the Rational Quadratic Flow for the unknown locations of profiles
in the LALME model.
"""

  flow_dim = 2 * num_profiles

  event_shape = (flow_dim,)

  flow_layers = []

  # # Layer: Affine transformation
  # loc = hk.get_parameter("loc", event_shape, init=jnp.zeros)
  # # loc = 10*jnp.ones(event_shape)
  # log_scale = hk.get_parameter("log_scale", event_shape, init=jnp.zeros)
  # # log_scale = jnp.zeros(event_shape)
  # flow_layers.append(
  #     distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))
  # # flow_layers.append(tfb.Shift(loc)(tfb.Scale(log_scale=log_scale)))

  # Number of parameters required by the bijector (rational quadratic spline)
  num_bijector_params = 3 * num_bins + 1

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=spline_range[0], range_max=spline_range[1])

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
    layer = modularbayes.EtaConditionalMaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner_eta=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner_eta,
            num_bijector_params=num_bijector_params,
            name='conditioner_eta_locations',
        ),
        conditioner=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner,
            num_bijector_params=num_bijector_params,
            name='conditioner_locations',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # all locations to the [0,1] square
  flow_layers.append(distrax.Block(distrax.Sigmoid(), 1))
  # profiles x's go to [0,loc_x_max]
  # profiles y's go to [0,loc_y_max]
  if loc_x_range == (0., 1.):
    loc_x_range_bijector = distrax.Block(tfb.Identity(), 1)
  else:
    loc_x_range_bijector = distrax.Block(
        distrax.ScalarAffine(
            shift=loc_x_range[0], scale=loc_x_range[1] - loc_x_range[0]), 1)

  if loc_y_range == (0., 1.):
    loc_y_range_bijector = distrax.Block(tfb.Identity(), 1)
  else:
    loc_y_range_bijector = distrax.Block(
        distrax.ScalarAffine(
            shift=loc_y_range[0], scale=loc_y_range[1] - loc_y_range[0]), 1)

  block_bijectors = [loc_x_range_bijector, loc_y_range_bijector]
  block_sizes = [num_profiles, num_profiles]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  return modularbayes.ConditionalTransformed(base_distribution, flow)


def split_flow_global_params(
    samples: Array,
    num_forms_tuple: Tuple,
    num_base_gps: int,
    num_inducing_points: int,
    **_,
) -> Dict[str, Any]:
  """Get model parameters by splitting samples from the flow."""

  assert samples.ndim == 2
  num_samples, flow_dim = samples.shape

  num_items = len(num_forms_tuple)

  gamma_inducing_dim = num_base_gps * num_inducing_points
  mixing_weights_dims = [
      num_base_gps * num_forms_i for num_forms_i in num_forms_tuple
  ]
  mixing_weights_dim = sum(mixing_weights_dims)
  mixing_offset_dim = sum(num_forms_tuple)
  mu_dim = num_items
  zeta_dim = num_items

  assert flow_dim == (
      gamma_inducing_dim + mixing_weights_dim + mixing_offset_dim + mu_dim +
      zeta_dim)

  samples_dict = {}

  # GP on inducing points
  gamma_inducing, samples = jnp.split(samples, [gamma_inducing_dim], axis=-1)
  gamma_inducing = gamma_inducing.reshape(num_samples, num_base_gps,
                                          num_inducing_points)
  samples_dict['gamma_inducing'] = gamma_inducing

  # Mixing coefficients
  mixing_weights, samples = jnp.split(samples, [mixing_weights_dim], axis=-1)
  # The split cannot be done constructing the vector with cumsum
  # TODO: understand why this is the case
  # _split_idx = jnp.cumsum(jnp.array(mixing_weights_dims))[:-1]
  # mixing_weights_list = jnp.split(mixing_weights, _split_idx, axis=-1)
  # Split compatible with jit
  mixing_weights_list = []
  for num_forms_i in num_forms_tuple[:-1]:
    mixing_weights_i, mixing_weights = jnp.split(
        mixing_weights, [num_base_gps * num_forms_i], axis=-1)
    mixing_weights_list.append(mixing_weights_i)
  mixing_weights_list.append(mixing_weights)
  mixing_weights_list = [
      mixing_weights_list[i].reshape(num_samples, num_base_gps, num_forms_i)
      for i, num_forms_i in enumerate(num_forms_tuple)
  ]
  samples_dict['mixing_weights_list'] = mixing_weights_list

  # Mixing offset
  mixing_offset, samples = jnp.split(samples, [mixing_offset_dim], axis=-1)
  mixing_offset_list = []
  for num_forms_i in num_forms_tuple[:-1]:
    mixing_offset_i, mixing_offset = jnp.split(
        mixing_offset, [num_forms_i], axis=-1)
    mixing_offset_list.append(mixing_offset_i)
  mixing_offset_list.append(mixing_offset)
  samples_dict['mixing_offset_list'] = mixing_offset_list

  mu, zeta = jnp.split(samples, [mu_dim], axis=-1)
  samples_dict['mu'] = mu
  samples_dict['zeta'] = zeta

  return samples_dict


def split_flow_locations(
    samples: Array,
    num_profiles: int,
    is_aux: bool,
    name='loc_floating',
    **_,
) -> Dict[str, Any]:
  """Get model parameters by splitting samples from the flow."""

  assert samples.ndim == 2
  num_samples, flow_dim = samples.shape

  assert flow_dim == 2 * num_profiles

  samples_dict = {}

  # This reshaping guarantees that all profiles x's go first
  # (instead of (x,y) for each profile)
  locations = samples.reshape((num_samples, 2, num_profiles)).swapaxes(1, 2)

  if is_aux:
    samples_dict[name + '_aux'] = locations
  else:
    samples_dict[name] = locations

  return samples_dict
