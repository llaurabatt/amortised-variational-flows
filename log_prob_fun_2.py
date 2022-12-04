"""Probability functions for the LALME model."""

from collections import namedtuple

import jax
import jax.numpy as jnp

import distrax
from tensorflow_probability.substrates import jax as tfp

import haiku as hk

from modularbayes._src.typing import (Any, Array, Batch, Dict, List, Optional,
                                      PRNGKey, SmiEta, Union, Tuple)

from misc import log1mexpm, force_symmetric

tfd = tfp.distributions
kernels = tfp.math.psd_kernels

ModelParamsGlobal = namedtuple("modelparams_global", [
    'gamma_inducing',
    'mixing_weights_list',
    'mixing_offset_list',
    'mu',
    'zeta',
])
ModelParamsLocations = namedtuple("modelparams_locations", [
    'loc_floating',
    'loc_floating_aux',
    'loc_random_anchor',
])
ModelParamsGammaProfiles = namedtuple("modelparams_gamma_profiles", [
    'gamma_anchor',
    'gamma_floating',
    'gamma_floating_aux',
    'gamma_random_anchor',
])


## Observational model ##
def log_prob_y_equal_1(
    gamma_profiles: Array,
    mixing_weights_list: List[Array],
    mixing_offset_list: List[Array],
    mu: Array,
    zeta: Array,
) -> List[Array]:
  """Observational model for LALME data.
  
  Compute the log probability of form ocurrence given the model parameters.
  """
  # Get Probability fields
  # Linear transformation of GPs
  phi_prob_item_list_unbounded = [
      jnp.einsum(
          "bf,bp->fp",
          weights_i,
          gamma_profiles,
          precision='highest',
      ) + jnp.expand_dims(offset_i, axis=-1)
      for weights_i, offset_i in zip(mixing_weights_list, mixing_offset_list)
  ]
  # softmax transform over the form axis.
  phi_prob_item_list = [
      jax.nn.softmax(phi_prob, axis=0)
      for phi_prob in phi_prob_item_list_unbounded
  ]

  log_prob_y_equal_1_list = [
      jnp.log(1 - zeta[i]) + log1mexpm(mu[i] * phi_prob_item)
      for i, phi_prob_item in enumerate(phi_prob_item_list)
  ]

  return log_prob_y_equal_1_list


def log_prob_y_integrated_over_gamma_profiles(
    batch: Batch,
    model_params_global: ModelParamsGlobal,
    model_params_gamma_profiles: ModelParamsGammaProfiles,
    smi_eta: Optional[SmiEta] = None,
) -> Array:
  """Log-probability of data given global parameters and loc_floating

  Integrate over multiple samples of gamma_profiles.
  """

  # Concatenate samples of gamma for all profiles, anchor and floating
  gamma_profiles = jnp.concatenate([
      model_params_gamma_profiles.gamma_anchor,
      model_params_gamma_profiles.gamma_floating
  ],
                                   axis=-1)

  # num_samples_gamma_profiles = model_params_gamma_profiles.gamma_anchor.shape[0]

  # Computes log_prob_y_equal_1 over the sampled values gamma_profiles
  log_prob_y_equal_1_pointwise_list = jax.vmap(
      lambda gamma_p: log_prob_y_equal_1(
          gamma_profiles=gamma_p,
          mixing_weights_list=model_params_global.mixing_weights_list,
          mixing_offset_list=model_params_global.mixing_offset_list,
          mu=model_params_global.mu,
          zeta=model_params_global.zeta,
      ))(
          gamma_profiles)
  # assert all(x.shape == (num_samples_gamma_profiles,
  #                        batch['num_forms_tuple'][i], batch['num_profiles'])
  #            for i, x in enumerate(log_prob_y_equal_1_pointwise_list))

  # Average log_prob_y_equal_1 over samples of gamma_profiles and add over forms
  log_prob_y_item_profile_ = []
  for y_i, log_prob_y_eq_1_i in zip(batch['y'],
                                    log_prob_y_equal_1_pointwise_list):
    log_prob_y_item_profile_.append(
        jnp.where(y_i, log_prob_y_eq_1_i,
                  log1mexpm(-log_prob_y_eq_1_i)).mean(axis=0).sum(axis=0))
  log_prob_y_item_profile = jnp.stack(log_prob_y_item_profile_, axis=0)

  # assert log_prob_y_item_profile.shape == (batch['num_items'],
  #                                          batch['num_profiles'])

  # The eta power is used to temper the likelihood of profiles and items
  if smi_eta is not None:
    if ('profiles' in smi_eta) and (smi_eta['profiles'] is not None):
      log_prob_y_item_profile *= jnp.expand_dims(smi_eta['profiles'], [0])
    if ('items' in smi_eta) and (smi_eta['items'] is not None):
      log_prob_y_item_profile *= jnp.expand_dims(smi_eta['items'], [1])
  log_prob = log_prob_y_item_profile.sum()

  return log_prob


def sample_gamma_profiles_given_gamma_inducing(
    batch: Batch,
    model_params_global: ModelParamsGlobal,
    model_params_locations: ModelParamsLocations,
    prng_key: PRNGKey,
    kernel_name: str,
    kernel_kwargs: Dict[str, Any],
    gp_jitter: float,
    num_samples_gamma_profiles: int,
    is_smi: bool,
    include_random_anchor: bool,
) -> ModelParamsGammaProfiles:
  """Sample from the conditional distribution p(gamma_p|gamma_u).

  Integrate over multiple samples of gamma_profiles.

  Args:
    batch: Batch of data from the LALME dataset.
    posterior_sample_dict: Dictionary with samples of parameters in the LALME
      model.
    prng_key: Random seed.
    kernel_name: String specifiying the Kernel function to be used.
    kernel_kwargs: Dictionary with keyword arguments for the Kernel function.
    gp_jitter: float, jitter to add to the diagonal of the conditional
      covariance matrices, only for numerical stability.
    num_samples_gamma_profiles: Integer indicating the number of samples used
      in the montecarlo integration of gamma_profiles.
    is_smi: Boolean indicating if a SMI posterior is being used.
    include_random_anchor: Boolean indicating if the random anchor should be
      produced as part of the samples.

  Returns:
    Dictionary with samples of the GPs at the profiles locations.
  """

  # num_samples_global, num_basis_gps, _ = posterior_sample_dict[
  #     'gamma_inducing'].shape

  ### Sample gamma_profiles ###
  gamma_sample_dict = {}
  prng_seq = hk.PRNGSequence(prng_key)

  ### Sample Gamma on Anchor profiles

  # Conditional mean and covariance of GP on anchor locations, given GP on inducing values
  # (vmap over basis fields)
  gamma_anchor_mean, gamma_anchor_cov = jax.vmap(
      lambda gamma_inducing: conditional_gaussian_x_given_y(
          y=gamma_inducing,
          cov_x=batch['cov_anchor'],
          cov_xy=batch['cov_anchor_inducing'],
          cov_y_inv=batch['cov_inducing_inv'],
      ))(
          model_params_global.gamma_inducing)

  gamma_anchor_cov += gp_jitter * jnp.broadcast_to(
      jnp.eye(gamma_anchor_cov.shape[-1]), gamma_anchor_cov.shape)

  p_anchor_given_inducing = distrax.Independent(
      distrax.MultivariateNormalTri(
          loc=gamma_anchor_mean,
          scale_tri=jax.vmap(jnp.linalg.cholesky)(gamma_anchor_cov),
      ),
      reinterpreted_batch_ndims=1)

  gamma_sample_dict['gamma_anchor'] = p_anchor_given_inducing.sample(
      seed=next(prng_seq), sample_shape=(num_samples_gamma_profiles,))
  # assert gamma_sample_dict['gamma_anchor'].shape == (
  #     num_samples_gamma_profiles, num_basis_gps,
  #     batch['num_profiles_anchor'])

  ### Sample Gamma on Floating profiles

  kernel = getattr(kernels, kernel_name)

  # Compute covariance (kernel) between floating locations.
  cov_floating = kernel(**kernel_kwargs).matrix(
      model_params_locations.loc_floating,
      model_params_locations.loc_floating,
  )
  # Compute covariance (kernel) between floating and inducing locations.
  cov_floating_inducing = kernel(**kernel_kwargs).matrix(
      model_params_locations.loc_floating,
      batch['loc_inducing'],
  )

  # Conditional mean and covariance GP on floating locations, given GP on inducing values
  # (vmap over basis fields)
  gamma_floating_mean, gamma_floating_cov = jax.vmap(
      lambda gamma_inducing: conditional_gaussian_x_given_y(
          y=gamma_inducing,
          cov_x=cov_floating,
          cov_xy=cov_floating_inducing,
          cov_y_inv=batch['cov_inducing_inv'],
      ))(
          model_params_global.gamma_inducing)

  gamma_floating_cov += gp_jitter * jnp.broadcast_to(
      jnp.eye(gamma_floating_cov.shape[-1]), gamma_floating_cov.shape)

  p_floating_given_inducing = distrax.Independent(
      distrax.MultivariateNormalTri(
          loc=gamma_floating_mean,
          scale_tri=jax.vmap(jnp.linalg.cholesky)(gamma_floating_cov),
      ),
      reinterpreted_batch_ndims=1)

  gamma_sample_dict['gamma_floating'] = p_floating_given_inducing.sample(
      seed=next(prng_seq), sample_shape=(num_samples_gamma_profiles,))
  # assert gamma_sample_dict['gamma_floating'].shape == (
  #     num_samples_gamma_profiles, num_basis_gps,
  #     batch['num_profiles_floating'])

  ### Sample Gamma on Floating (aux) profiles
  if is_smi:
    # Compute covariance (kernel) on auxiliary floating locations.
    cov_floating_aux = kernel(**kernel_kwargs).matrix(
        model_params_locations.loc_floating_aux,
        model_params_locations.loc_floating_aux,
    )
    cov_floating_aux_inducing = kernel(**kernel_kwargs).matrix(
        model_params_locations.loc_floating_aux,
        batch['loc_inducing'],
    )

    # Conditional mean and covariance GP on floating_aux locations, given GP on inducing values
    # (vmap over basis fields)
    gamma_floating_aux_mean, gamma_floating_aux_cov = jax.vmap(
        lambda gamma_inducing: conditional_gaussian_x_given_y(
            y=gamma_inducing,
            cov_x=cov_floating_aux,
            cov_xy=cov_floating_aux_inducing,
            cov_y_inv=batch['cov_inducing_inv'],
        ))(
            model_params_global.gamma_inducing)

    gamma_floating_aux_cov += gp_jitter * jnp.broadcast_to(
        jnp.eye(gamma_floating_aux_cov.shape[-1]), gamma_floating_aux_cov.shape)

    p_floating_aux_given_inducing = distrax.Independent(
        distrax.MultivariateNormalTri(
            loc=gamma_floating_aux_mean,
            scale_tri=jax.vmap(jnp.linalg.cholesky)(gamma_floating_aux_cov),
        ),
        reinterpreted_batch_ndims=1)

    gamma_sample_dict['gamma_floating_aux'] = (
        p_floating_aux_given_inducing.sample(
            seed=next(prng_seq), sample_shape=(num_samples_gamma_profiles,)))
    # assert gamma_sample_dict['gamma_floating_aux'].shape == (
    #     num_samples_gamma_profiles, num_basis_gps,
    #     batch['num_profiles_floating'])
  else:
    gamma_sample_dict['gamma_floating_aux'] = None

  ### Sample Gamma on (random) Anchor locations
  if include_random_anchor:
    # Compute covariance (kernel) on random anchor locations.
    cov_random_anchor = kernel(**kernel_kwargs).matrix(
        model_params_locations.loc_random_anchor,
        model_params_locations.loc_random_anchor,
    )
    cov_random_anchor_inducing = kernel(**kernel_kwargs).matrix(
        model_params_locations.loc_random_anchor,
        batch['loc_inducing'],
    )

    # Conditional mean and covariance GP on random_anchor locations, given GP on inducing values
    # (vmap over basis fields)
    gamma_random_anchor_mean, gamma_random_anchor_cov = jax.vmap(
        lambda gamma_inducing: conditional_gaussian_x_given_y(
            y=gamma_inducing,
            cov_x=cov_random_anchor,
            cov_xy=cov_random_anchor_inducing,
            cov_y_inv=batch['cov_inducing_inv'],
        ))(
            model_params_global.gamma_inducing)

    gamma_random_anchor_cov += gp_jitter * jnp.broadcast_to(
        jnp.eye(gamma_random_anchor_cov.shape[-1]),
        gamma_random_anchor_cov.shape)

    p_random_anchor_given_inducing = distrax.Independent(
        distrax.MultivariateNormalTri(
            loc=gamma_random_anchor_mean,
            scale_tri=jax.vmap(jnp.linalg.cholesky)(gamma_random_anchor_cov),
        ),
        reinterpreted_batch_ndims=1)

    gamma_sample_dict[
        'gamma_random_anchor'] = p_random_anchor_given_inducing.sample(
            seed=next(prng_seq), sample_shape=(num_samples_gamma_profiles,))
  else:
    gamma_sample_dict['gamma_random_anchor'] = None

  model_params_gamma = ModelParamsGammaProfiles(**gamma_sample_dict)

  return model_params_gamma


def log_prob_joint(
    batch: Batch,
    model_params_global: ModelParamsGlobal,
    model_params_locations: ModelParamsLocations,
    model_params_gamma_profiles: ModelParamsGammaProfiles,
    smi_eta: Optional[SmiEta] = None,
    w_prior_scale: float = 1.,
    a_prior_scale: float = 10.,
    mu_prior_concentration: float = 1.,
    mu_prior_rate: float = 10.,
    zeta_prior_a: float = 1.,
    zeta_prior_b: float = 1.,
) -> Array:
  """Log-density for the LALME model.

  Args:
    batch: Batch of data from the LALME dataset. a dictionary with the following
      keys:
      -cov_inducing_chol: covariance matrix between the inducing points.
    model_params_global: Named tuple with values of global parameters of the
      LALME model, with elements:
      -gamma_inducing: Array of shape (num_basis_gps, num_inducing_points) with
        the values of basis GPs at the inducing points.
      -mixing_weights_list: List of Arrays, with the same lenght of
        num_forms_tuple, and each element with shape (num_basis_gps,num_forms_tuple[i])
      -mixing_offset_list: List of Arrays, with the same lenght of
        num_forms_tuple, and each element with shape (num_forms_tuple[i])
      -mu: Array of shape (num_items) with item's occurrence rates.
      -zeta: Array of shape (num_items) with the zero-inflation of item's
        occurrence rates.
    model_params_locations: Named tuple with values of profiles locations, with
      elments:
      -loc_floating: Array of shape (num_profiles_floating, 2) with locations
        associated to floating profiles.
      -loc_floating_aux: ditto, but for auxiliary copy of floating profiles,
        used for SMI.
      -loc_random_anchor: Array of shape (num_profiles_anchor, 2) with locations
        associated with anchor profiles, treated as if they were unknown.
        samples of the locations.
    model_params_locations: Named tuple with values of basis fields at profiles
      locations, with elments:
      -gamma_anchor: Array of shape (num_basis_gps, num_profiles_anchor) with
        the value of basis GPs at the anchor profiles locations.
      -gamma_floating: Array of shape (num_basis_gps, num_profiles_floating) with
        the value of basis GPs at the floating profiles locations.
        floating points.
      -gamma_floating_aux: ditto, but for the auxiliary copy of floating
        profiles, used for SMI.
      -gamma_random_anchor: Basis fields at random anchor locations when treated
        as if they were unknown.
    smi_eta : Dictionary specifying the SMI influence parameters with two keys:
      -profiles:
      -items:
  """

  # Get model dimensions
  num_inducing_points = model_params_global.gamma_inducing.shape[-1]
  # pylint: disable=consider-using-generator
  num_forms_tuple = tuple(
      [x.shape[-1] for x in model_params_global.mixing_weights_list])
  num_items = len(num_forms_tuple)

  ## Observational model ##
  # We integrate the likelihood over samples of gamma_profiles
  # Such samples are not part of the posterior approximation q(Theta), but we
  # assume they come in the dictionary 'posterior_sample_dict' for convenience.

  ## Priors ##

  # P(Gamma_Z) : Prior on the GPs on inducing points
  log_prob_gamma = distrax.Independent(
      distrax.MultivariateNormalTri(
          loc=jnp.zeros((1, num_inducing_points)),
          scale_tri=batch['cov_inducing_chol']),
      reinterpreted_batch_ndims=1).log_prob

  # P(loc_floating) : Prior on the floating locations
  # TODO: more likely where there are more items already
  def log_prob_loc_floating(_):
    return 0.

  # P(W) : prior on the mixing weights
  def log_prob_weights(mixing_weights_list: List[Array]):
    # Induce sparsity on the representation of fields
    log_prob = jnp.stack([
        distrax.Independent(
            tfd.Laplace(
                loc=jnp.zeros(weights_i.shape),
                scale=w_prior_scale * jnp.ones(weights_i.shape),
            ),
            reinterpreted_batch_ndims=2).log_prob(weights_i)
        for weights_i in mixing_weights_list
    ],
                         axis=-1).sum(axis=-1)
    return log_prob

  # P(offset) : prior on the mixing offset
  def log_prob_offset(mixing_offset_list):
    # Gaussian prior
    log_prob = jnp.stack([
        distrax.Independent(
            distrax.Normal(
                loc=jnp.zeros(offset_i.shape),
                scale=a_prior_scale * jnp.ones(offset_i.shape),
            ),
            reinterpreted_batch_ndims=1).log_prob(offset_i)
        for offset_i in mixing_offset_list
    ],
                         axis=-1).sum(axis=-1)
    return log_prob

  # P(mu) : Prior on rate of occurence for each item
  log_prob_mu = distrax.Independent(
      tfd.Gamma(
          concentration=mu_prior_concentration * jnp.ones(num_items),
          rate=mu_prior_rate * jnp.ones(num_items)),
      reinterpreted_batch_ndims=1).log_prob

  # P(zeta) : Prior on the item zero-inflation parameter: Beta
  log_prob_zeta = distrax.Independent(
      tfd.Beta(
          concentration1=zeta_prior_a * jnp.ones(num_items),
          concentration0=zeta_prior_b * jnp.ones(num_items),
      ),
      reinterpreted_batch_ndims=1).log_prob

  log_prob = (
      # P(Y | Phi_X, mu, zeta, Gamma_U, W, a)
      log_prob_y_integrated_over_gamma_profiles(
          batch=batch,
          model_params_global=model_params_global,
          model_params_gamma_profiles=model_params_gamma_profiles,
          smi_eta=smi_eta,
      ) +
      # P(mu)
      log_prob_mu(model_params_global.mu) +
      # P(zeta)
      log_prob_zeta(model_params_global.zeta) +
      # P(Gamma_U)
      log_prob_gamma(model_params_global.gamma_inducing) +
      # P(W)
      log_prob_weights(model_params_global.mixing_weights_list) +
      # P(a)
      log_prob_offset(model_params_global.mixing_offset_list) +
      # P(loc_floating)
      log_prob_loc_floating(model_params_locations.loc_floating))

  return log_prob


def conditional_gaussian_x_given_y(
    y: Array,
    cov_x: Array,
    cov_xy: Array,
    cov_y_inv: Array,
    mu_x: Union[Array, float] = 0,
    mu_y: Union[Array, float] = 0,
    only_mean: bool = False,
) -> Tuple[Array, Array]:
  """Conditional Gaussian Mean and Covariance.

  Assuming a joint gaussian distribution for X and Y, with mean (mu_x,mu_y) and
  covariance [[cov_x,cov_xy],[cov_xy^T, cov_y]], this function computes the
  conditional mean and covariance of X given Y=y.
  """

  mat_a = cov_xy @ cov_y_inv

  mu_x_given_y = mu_x + mat_a @ (y - mu_y)

  if only_mean:
    cov_x_given_y = None
  else:
    cov_x_given_y = cov_x - force_symmetric(mat_a @ cov_xy.T)

  return mu_x_given_y, cov_x_given_y
