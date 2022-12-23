"""Define Normalizing Flows for the LALME model."""

import math

from jax import numpy as jnp

import distrax
from tensorflow_probability.substrates import jax as tfp

import modularbayes
from modularbayes._src.typing import Any, Array, Dict, Sequence, Tuple

from log_prob_fun_2 import ModelParamsGlobal, ModelParamsLocations

tfb = tfp.bijectors
tfd = tfp.distributions


def mean_field_global_params(
    num_forms_tuple: Tuple[int, ...],
    num_basis_gps: int,
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

  gamma_inducing_dim = num_basis_gps * num_inducing_points
  mixing_weights_dim = sum(
      [num_basis_gps * num_forms_i for num_forms_i in num_forms_tuple])
  mixing_offset_dim = sum(num_forms_tuple)
  mu_dim = num_items
  zeta_dim = num_items

  flow_dim = (
      gamma_inducing_dim + mixing_weights_dim + mixing_offset_dim + mu_dim +
      zeta_dim)

  event_shape = (flow_dim,)

  flow_layers = []

  # Layer 1: Trainable Affine transformation
  conditioner = modularbayes.MeanFieldConditioner(
      flow_dim=flow_dim, name='mf_global_params')
  loc, log_scale = conditioner()
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
    loc_x_range: Tuple[float, float],
    loc_y_range: Tuple[float, float],
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
  conditioner = modularbayes.MeanFieldConditioner(
      flow_dim=flow_dim, name='mf_locations')
  loc, log_scale = conditioner()
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


def nsf_global_params(
    num_forms_tuple: Tuple[int, ...],
    num_basis_gps: int,
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

  gamma_inducing_dim = num_basis_gps * num_inducing_points
  mixing_weights_dim = sum(
      [num_basis_gps * num_forms_i for num_forms_i in num_forms_tuple])
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

  # NSF layers
  for _ in range(num_layers):
    layer = distrax.MaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=modularbayes.MLPConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes,
            num_bijector_params=num_bijector_params,
            name='nsf_global_params',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Affine transformation layer
  conditioner = modularbayes.MeanFieldConditioner(
      flow_dim=flow_dim, name='affine_global_params')
  loc, log_scale = conditioner()
  flow_layers.append(
      distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))

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
    loc_x_range: Tuple[float, float],
    loc_y_range: Tuple[float, float],
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
    return distrax.RationalQuadraticSpline(params, range_min=0., range_max=1.)

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
        conditioner=modularbayes.MLPConditioner(
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

  block_bijectors = [loc_x_range_bijector, loc_y_range_bijector] * num_profiles
  block_sizes = [1, 1] * num_profiles
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  # base_distribution = distrax.Independent(
  #     distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
  #     reinterpreted_batch_ndims=len(event_shape))

  base_distribution = distrax.Independent(
      distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
      reinterpreted_batch_ndims=1)

  return modularbayes.ConditionalTransformed(base_distribution, flow)


def meta_nsf_global_params(
    num_forms_tuple: Tuple[int, ...],
    num_basis_gps: int,
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

  gamma_inducing_dim = num_basis_gps * num_inducing_points
  mixing_weights_dim = sum(
      [num_basis_gps * num_forms_i for num_forms_i in num_forms_tuple])
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

  # NSF layers
  for _ in range(num_layers):
    layer = modularbayes.EtaConditionalMaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner_eta=modularbayes.MLPConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner_eta,
            num_bijector_params=num_bijector_params,
            name='conditioner_eta_global_params',
        ),
        conditioner=modularbayes.MLPConditioner(
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
    loc_x_range: Tuple[float, float],
    loc_y_range: Tuple[float, float],
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
    return distrax.RationalQuadraticSpline(params, range_min=0., range_max=1.)

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
        conditioner_eta=modularbayes.MLPConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner_eta,
            num_bijector_params=num_bijector_params,
            name='conditioner_eta_locations',
        ),
        conditioner=modularbayes.MLPConditioner(
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

  base_distribution = distrax.Independent(
      distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
      reinterpreted_batch_ndims=1)

  return modularbayes.ConditionalTransformed(base_distribution, flow)


def get_global_params_shapes(
    num_forms_tuple: Tuple[int, ...],
    num_basis_gps: int,
    num_inducing_points: int,
) -> Dict[str, Any]:
  """Computes shapes of global parameters as expected by the model."""

  num_items = len(num_forms_tuple)

  params_shapes = {
      'gamma_inducing': (num_basis_gps, num_inducing_points),
      'mixing_weights_list': [
          (num_basis_gps, num_forms_i) for num_forms_i in num_forms_tuple
      ],
      'mixing_offset_list': [(num_forms_i,) for num_forms_i in num_forms_tuple],
      'mu': (num_items,),
      'zeta': (num_items,),
  }

  return params_shapes


def get_global_params_dim(
    num_forms_tuple: Tuple[int, ...],
    num_basis_gps: int,
    num_inducing_points: int,
) -> int:
  """Computes dimension of vector with global parameters."""
  num_items = len(num_forms_tuple)
  gamma_inducing_dim = num_basis_gps * num_inducing_points
  mixing_weights_dims = [
      num_basis_gps * num_forms_i for num_forms_i in num_forms_tuple
  ]
  mixing_weights_dim = sum(mixing_weights_dims)
  mixing_offset_dim = sum(num_forms_tuple)
  mu_dim = num_items
  zeta_dim = num_items
  global_params_dim = (
      gamma_inducing_dim + mixing_weights_dim + mixing_offset_dim + mu_dim +
      zeta_dim)
  return global_params_dim


def split_flow_global_params(
    samples: Array,
    num_forms_tuple: Tuple[int, ...],
    num_basis_gps: int,
    num_inducing_points: int,
    **_,
) -> ModelParamsGlobal:
  """Dictionary with posterior samples.

  Split an array with samples of the model into a dictionary with the
  corresponding LALME parameters. This function is for the posterior of the
  global parameters.

  Args:
    samples: Array with samples, assumed to be concatenated by their last axis.
      Expected shape: (num_samples, flow_dim).
    num_profiles: Number of profiles.
    is_aux: For SMI, indicates whether the samples are auxiliary or not.
      If true, appends the suffix '_aux' to the key.
    name: Base name assigned to the samples.

  Returns:
    A dictionary with the following keys (assuming name='loc_floating' and
    is_aux=False):
      -gamma_inducing: Array of shape (num_samples, num_basis_gps,
        num_inducing_points) with samples of the basis GPs at the inducing
        points.
      -mixing_weights_list: List of Arrays, with the same lenght of
        num_forms_tuple, and each element with shape (num_samples,
        num_basis_gps, num_forms_tuple[i])
      -mixing_offset_list: List of Arrays, with the same lenght of
        num_forms_tuple, and each element with shape (num_samples,
        num_forms_tuple[i])
      -mu: Array of shape (num_samples, num_items) with samples of the item
        occurrence rates.
      -zeta: Array of shape (num_samples, num_items) with samples of the
        zero-inflation of the item occurrence rates.

  Note:
    Here the last dimension of the input is expected to be:
      flow_dim=(gamma_inducing_dim
        + mixing_weights_dim
        + mixing_offset_dim
        + mu_dim
        + zeta_dim)
  """

  assert samples.ndim == 2
  num_samples, flow_dim = samples.shape

  num_items = len(num_forms_tuple)

  gamma_inducing_dim = num_basis_gps * num_inducing_points
  mixing_weights_dims = [
      num_basis_gps * num_forms_i for num_forms_i in num_forms_tuple
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
  gamma_inducing = gamma_inducing.reshape(num_samples, num_basis_gps,
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
        mixing_weights, [num_basis_gps * num_forms_i], axis=-1)
    mixing_weights_list.append(mixing_weights_i)
  mixing_weights_list.append(mixing_weights)
  mixing_weights_list = [
      mixing_weights_list[i].reshape(num_samples, num_basis_gps, num_forms_i)
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

  model_params_global_sample = ModelParamsGlobal(**samples_dict)

  return model_params_global_sample


def split_flow_locations(
    samples: Array,
    num_profiles: int,
    name='loc_floating',
    **_,
) -> ModelParamsLocations:
  """Dictionary with posterior samples.

  Split an array with samples of the model into a dictionary with the
  corresponding LALME parameters. This function is for the posterior of the
  profiles locations.

  Args:
    samples: Array with samples, assumed to be concatenated by their last axis.
      Expected shape: (num_samples, flow_dim).
    num_profiles: Number of profiles.
    is_aux: For SMI, indicates whether the samples are auxiliary or not.
      If true, appends the suffix '_aux' to the key.
    name: Base name assigned to the samples.

  Returns:
    A dictionary with the following keys (assuming name='loc_floating' and
    is_aux=False):
      -loc_floating: Array of shape (num_samples, num_profiles, 2) with
        samples of the locations.

  Note:
    Here the last dimension of the input is expected to be:
      flow_dim=2*num_profiles
  """

  assert samples.ndim == 2
  num_samples, flow_dim = samples.shape

  assert flow_dim == 2 * num_profiles

  samples_dict = {}

  # Two options to reshape the flow into paired locations
  # Option 2 is favoured because the coupling conditioner mask alternates 0,1,0,1,...

  # # 1) Assume that coordinates in the flow come (x1,...,xn,y1,...,yn)
  # locations = samples.reshape((num_samples, 2, num_profiles)).swapaxes(1, 2)
  # 2) Assume that coordinates in the flow come (x1,y1,...,xn,yn)
  locations = samples.reshape((num_samples, num_profiles, 2))

  samples_dict[name] = locations
  model_params_locations_sample = ModelParamsLocations(**samples_dict)

  return model_params_locations_sample


def concat_samples_global_params(samples_dict: Dict[str, Any]) -> Array:
  """Undo split_flow_global_params"""

  # Get sizes
  num_samples, num_basis_gps, num_inducing_points = samples_dict[
      'gamma_inducing'].shape

  # pylint: disable=consider-using-generator
  num_forms_tuple = tuple(
      [x.shape[-1] for x in samples_dict['mixing_weights_list']])

  samples = []

  # GPs on inducing points
  samples.append(samples_dict['gamma_inducing'].reshape(num_samples, -1))
  assert samples[-1].shape == (num_samples, num_basis_gps * num_inducing_points)

  # mixing weights
  samples.append(
      jnp.concatenate([
          x.reshape(num_samples, -1)
          for x in samples_dict['mixing_weights_list']
      ],
                      axis=-1))
  assert samples[-1].shape == (num_samples,
                               num_basis_gps * sum(num_forms_tuple))

  # mixing offset
  samples.append(
      jnp.concatenate([
          x.reshape(num_samples, -1) for x in samples_dict['mixing_offset_list']
      ],
                      axis=-1))
  assert samples[-1].shape == (num_samples, sum(num_forms_tuple))

  # mu
  samples.append(samples_dict['mu'])

  # zeta
  samples.append(samples_dict['zeta'])

  # Concatenate all samples
  samples = jnp.concatenate(samples, axis=-1)

  return samples


def concat_samples_locations(
    samples_dict: Dict[str, Any],
    is_aux: bool,
    name='loc_floating',
) -> Array:
  """Undo split_flow_locations"""

  key = name + ('_aux' if is_aux else '')
  # Get sizes
  num_samples, num_profiles, _ = samples_dict[key].shape

  # Unfolding the location samples depend on the choice made in split_flow_locations
  # Two options to reshape the flow into paired locations
  # Option 2 is favoured because the coupling conditioner mask alternates 0,1,0,1,...

  # # 1) Assume that coordinates in the flow come (x1,...,xn,y1,...,yn)
  # samples = samples_dict[key].swapaxes(1, 2).reshape(num_samples, -1)
  # 2) Assume that coordinates in the flow come (x1,y1,...,xn,yn)
  samples = samples_dict[key].reshape(num_samples, -1)

  assert samples.shape == (num_samples, 2 * num_profiles)

  return samples
