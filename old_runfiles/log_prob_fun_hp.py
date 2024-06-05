"""Probability functions for the LALME model."""

from collections import namedtuple

import jax
import jax.numpy as jnp

import distrax
from tensorflow_probability.substrates import jax as tfp

import haiku as hk

from modularbayes._src.typing import (Any, Array, Batch, Dict, List, Optional,
                                      PRNGKey, SmiEta, Union, Tuple)

from misc import log1mexpm, force_symmetric, issymmetric

tfd = tfp.distributions
kernels = tfp.math.psd_kernels

ModelParamsGlobal = namedtuple("modelparams_global", [
    'gamma_inducing',
    'mixing_weights_list',
    'mixing_offset_list',
    'mu',
    'zeta',
])
ModelParamsLocations = namedtuple(
    "modelparams_locations",
    [
        'loc_floating',
        'loc_floating_aux',
        'loc_random_anchor',
    ],
    defaults=[None] * 3,
)
ModelParamsGammaProfiles = namedtuple(
    "modelparams_gamma_profiles",
    [
        'gamma_anchor',
        'gamma_floating',
        'gamma_floating_aux',
        'gamma_random_anchor',
    ],
    defaults=[None] * 4,
)

PriorHparams = namedtuple(
    "prior_hparams",
    field_names=('w_prior_scale', 'a_prior_scale', 
                 'mu_prior_concentration', 'mu_prior_rate', 
                 'zeta_prior_a', 'zeta_prior_b', 
                 'kernel_amplitude', 'kernel_length_scale'),
    defaults=(5., 10., 1., 0.5, 1., 1., 0.2, 0.3),
)

SmiEtaNamedTuple = namedtuple(
    "smi_eta",
    field_names=('profiles', 'items'),
)

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


def log_prob_y_given_model_params(
    batch: Batch,
    model_params_global: ModelParamsGlobal,
    model_params_gamma_profiles: ModelParamsGammaProfiles,
    is_smi:bool,
    smi_eta: Optional[SmiEta] = None,
    random_anchor: bool = False,
) -> Array:
  """Log-probability of data given global parameters and loc_floating

  Integrate over multiple samples of gamma_profiles.
  """

  # Concatenate samples of gamma for all profiles, anchor and floating
  gamma_profiles = jnp.concatenate(
      [(model_params_gamma_profiles.gamma_anchor if not random_anchor else
        model_params_gamma_profiles.gamma_random_anchor),
       model_params_gamma_profiles.gamma_floating],
      axis=-1)

  # Computes log_prob_y_equal_1
  log_prob_y_equal_1_pointwise_list = log_prob_y_equal_1(
      gamma_profiles=gamma_profiles,
      mixing_weights_list=model_params_global.mixing_weights_list,
      mixing_offset_list=model_params_global.mixing_offset_list,
      mu=model_params_global.mu,
      zeta=model_params_global.zeta,
  )
  # assert all(x.shape == (batch['num_forms_tuple'][i], batch['num_profiles'])
  #            for i, x in enumerate(log_prob_y_equal_1_pointwise_list))

  # Sum log_prob_y_equal_1 over forms
  log_prob_y_item_profile_ = []
  for y_i, log_prob_y_eq_1_i in zip(batch['y'],
                                    log_prob_y_equal_1_pointwise_list):
    log_prob_y_item_profile_.append(
        jnp.where(y_i, log_prob_y_eq_1_i,
                  log1mexpm(-log_prob_y_eq_1_i)).sum(axis=0))
  log_prob_y_item_profile = jnp.stack(log_prob_y_item_profile_, axis=0)

  # assert log_prob_y_item_profile.shape == (batch['num_items'],
  #                                          batch['num_profiles'])

  # The eta power is used to temper the likelihood of profiles and items

  if is_smi:
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
    # kernel_kwargs: Dict[str, Any],
    prior_hparams:Dict[str, Any],
    gp_jitter: float,
    include_random_anchor: bool,
) -> Tuple[ModelParamsGammaProfiles, Dict[str, Array]]:
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

  kernel = getattr(kernels, kernel_name)

  gamma_anchor_cov = kernel(amplitude=prior_hparams.kernel_amplitude,
                        length_scale=prior_hparams.kernel_length_scale).matrix(
          x1=batch['loc'][:batch['num_profiles_anchor'], :],
          x2=batch['loc'][:batch['num_profiles_anchor'], :],
      )

 ####################################################################
  gamma_inducing_cov = kernel(amplitude=prior_hparams.kernel_amplitude,
                        length_scale=prior_hparams.kernel_length_scale).matrix(
                                        x1=batch['loc_inducing'],
                                        x2=batch['loc_inducing'],
                                    )

  # Add jitter
  gamma_inducing_cov = gamma_inducing_cov + gp_jitter * jnp.eye(
      batch['num_inducing_points'])
  # Check that the covarince is symmetric
  assert issymmetric(
      gamma_inducing_cov), 'Covariance Matrix is not symmetric'

  # Cholesky factor of covariance
  gamma_inducing_cov_chol = jnp.linalg.cholesky(gamma_inducing_cov)

  # Inverse of covariance of inducing values
  # dataset['cov_inducing_inv'] = jnp.linalg.inv(dataset['cov_inducing'])
  gamma_inducing_cov_chol_inv = jax.scipy.linalg.solve_triangular(
      a=gamma_inducing_cov_chol,
      b=jnp.eye(batch['num_inducing_points']),
      lower=True,
  )
  gamma_inducing_cov_inv = jnp.matmul(
      gamma_inducing_cov_chol_inv.T, gamma_inducing_cov_chol_inv, precision='highest')

  # Check that the inverse is symmetric
  assert issymmetric(
      gamma_inducing_cov_inv), 'Covariance Matrix is not symmetric'
  # Check that there are no NaNs
  assert ~jnp.any(jnp.isnan(gamma_inducing_cov_inv))
  # Cross covariance between anchor and inducing values
  gamma_anchor_inducing_cov = kernel(amplitude=prior_hparams.kernel_amplitude,
                        length_scale=prior_hparams.kernel_length_scale).matrix(
          x1=batch['loc'][:batch['num_profiles_anchor'], :],
          x2=batch['loc_inducing'],
      )
  ################################################################################

  # num_samples_global, num_basis_gps, _ = posterior_sample_dict[
  #     'gamma_inducing'].shape

  ### Sample gamma_profiles ###
  gamma_sample_dict = {}
  gamma_logprob_dict = {}
  prng_seq = hk.PRNGSequence(prng_key)

  ### Sample Gamma on Anchor profiles

  # Conditional mean and covariance of GP on anchor locations, given GP on inducing values
  # (vmap over basis fields)
  gamma_cond_anchor_mean, gamma_cond_anchor_cov = jax.vmap(
      lambda gamma_inducing: conditional_gaussian_x_given_y(
          y=gamma_inducing,
          cov_x=gamma_anchor_cov,
          cov_xy=gamma_anchor_inducing_cov,
          cov_y_inv=gamma_inducing_cov_inv,
      ))(
          model_params_global.gamma_inducing)

  gamma_cond_anchor_cov += gp_jitter * jnp.broadcast_to(
      jnp.eye(gamma_cond_anchor_cov.shape[-1]), gamma_cond_anchor_cov.shape)

  p_anchor_given_inducing = distrax.Independent(
      distrax.MultivariateNormalTri(
          loc=gamma_cond_anchor_mean,
          scale_tri=jax.vmap(jnp.linalg.cholesky)(gamma_cond_anchor_cov),
      ),
      reinterpreted_batch_ndims=1)

  gamma_sample_dict['gamma_anchor'], gamma_logprob_dict[
      'gamma_anchor'] = p_anchor_given_inducing.sample_and_log_prob(
          seed=next(prng_seq))
  # assert gamma_sample_dict['gamma_anchor'].shape == (
  #     num_samples_gamma_profiles, num_basis_gps,
  #     batch['num_profiles_anchor'])

  ### Sample Gamma on Floating profiles

  

  # Compute covariance (kernel) between floating locations.
  cov_floating = kernel(amplitude=prior_hparams.kernel_amplitude,
                        length_scale=prior_hparams.kernel_length_scale).matrix(
      model_params_locations.loc_floating,
      model_params_locations.loc_floating,
  )
  # Compute covariance (kernel) between floating and inducing locations.
  cov_floating_inducing = kernel(amplitude=prior_hparams.kernel_amplitude,
                        length_scale=prior_hparams.kernel_length_scale).matrix(
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
          cov_y_inv=gamma_inducing_cov_inv,
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

  gamma_sample_dict['gamma_floating'], gamma_logprob_dict[
      'gamma_floating'] = p_floating_given_inducing.sample_and_log_prob(
          seed=next(prng_seq))
  # assert gamma_sample_dict['gamma_floating'].shape == (
  #     num_samples_gamma_profiles, num_basis_gps,
  #     batch['num_profiles_floating'])

  ### Sample Gamma on (random) Anchor locations
  if include_random_anchor:
    # Compute covariance (kernel) on random anchor locations.
    cov_random_anchor = kernel(amplitude=prior_hparams.kernel_amplitude,
                        length_scale=prior_hparams.kernel_length_scale).matrix(
        model_params_locations.loc_random_anchor,
        model_params_locations.loc_random_anchor,
    )
    cov_random_anchor_inducing = kernel(amplitude=prior_hparams.kernel_amplitude,
                        length_scale=prior_hparams.kernel_length_scale).matrix(
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
            cov_y_inv=gamma_inducing_cov_inv,
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

    gamma_sample_dict['gamma_random_anchor'], gamma_logprob_dict[
        'gamma_random_anchor'] = p_random_anchor_given_inducing.sample_and_log_prob(
            seed=next(prng_seq))
  else:
    gamma_sample_dict['gamma_random_anchor'] = None

  model_params_gamma = ModelParamsGammaProfiles(**gamma_sample_dict)

  return model_params_gamma, gamma_logprob_dict


def logprob_joint(
    batch: Batch,
    model_params_global: ModelParamsGlobal,
    model_params_locations: ModelParamsLocations,
    model_params_gamma_profiles: ModelParamsGammaProfiles,
    gamma_profiles_logprob: Dict[str, Array],
    prior_hparams: Dict[str, float],
    is_smi:bool,
    smi_eta: Optional[SmiEta] = None,
    random_anchor: bool = False,
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

  ## Priors ##

  # P(Gamma_Z) : Prior on the GPs on inducing points
  log_prob_gamma_inducing = distrax.Independent(
      distrax.MultivariateNormalTri(
          loc=jnp.zeros((1, num_inducing_points)),
          scale_tri=batch['cov_inducing_chol']),
      reinterpreted_batch_ndims=1).log_prob

  # P(loc_floating) : Prior on the floating locations
  # TODO: more likely where there are more items already
  def log_prob_locations(_):
    return 0.

  # P(W) : prior on the mixing weights
  def log_prob_weights(mixing_weights_list: List[Array]):
    # Induce sparsity on the representation of fields
    log_prob = jnp.stack([
        distrax.Independent(
            tfd.Laplace(
                loc=jnp.zeros(weights_i.shape),
                scale=prior_hparams.w_prior_scale * jnp.ones(weights_i.shape),
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
                scale=prior_hparams.a_prior_scale * jnp.ones(offset_i.shape),
            ),
            reinterpreted_batch_ndims=1).log_prob(offset_i)
        for offset_i in mixing_offset_list
    ],
                         axis=-1).sum(axis=-1)
    return log_prob

  # P(mu) : Prior on rate of occurence for each item
  log_prob_mu = distrax.Independent(
      tfd.Gamma(
          concentration=prior_hparams.mu_prior_concentration * jnp.ones(num_items),
          rate=prior_hparams.mu_prior_rate * jnp.ones(num_items)),
      reinterpreted_batch_ndims=1).log_prob

  # P(zeta) : Prior on the item zero-inflation parameter: Beta
  log_prob_zeta = distrax.Independent(
      tfd.Beta(
          concentration1=prior_hparams.zeta_prior_a * jnp.ones(num_items),
          concentration0=prior_hparams.zeta_prior_b * jnp.ones(num_items),
      ),
      reinterpreted_batch_ndims=1).log_prob

  log_prob = (
      # P(Y | mu, zeta, a, W, Gamma_anchor, Gamma_floating)
      log_prob_y_given_model_params(
          batch=batch,
          model_params_global=model_params_global,
          model_params_gamma_profiles=model_params_gamma_profiles,
          is_smi=is_smi,
          smi_eta=smi_eta,
          random_anchor=random_anchor,
      ) +
      # P(Gamma_anchor | Gamma_U)
      gamma_profiles_logprob['gamma_anchor'] +
      # P(Gamma_floating | Gamma_U)
      gamma_profiles_logprob['gamma_floating'] +
      # P(Gamma_U)
      log_prob_gamma_inducing(model_params_global.gamma_inducing) +
      # P(mu)
      log_prob_mu(model_params_global.mu) +
      # P(zeta)
      log_prob_zeta(model_params_global.zeta) +
      # P(W)
      log_prob_weights(model_params_global.mixing_weights_list) +
      # P(a)
      log_prob_offset(model_params_global.mixing_offset_list) +
      # P(loc_floating)
      log_prob_locations(model_params_locations.loc_floating) +
      # P(loc_random_anchor)
      (log_prob_locations(model_params_locations.loc_random_anchor)
       if random_anchor else 0))
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


def sample_priorhparams_values(
    prng_key: PRNGKey,
    num_samples: int,
    w_sampling_scale_alpha: float,
    w_sampling_scale_beta: float,
    a_sampling_scale_alpha: float,
    a_sampling_scale_beta: float,
    kernel_sampling_amplitude_alpha: float,
    kernel_sampling_amplitude_beta: float,
    kernel_sampling_lengthscale_alpha: float,
    kernel_sampling_lengthscale_beta: float,
) -> PriorHparams:
  """Generate a sample of the prior hyperparameters values applicable to the model."""
  prng_keys = jax.random.split(prng_key, num=4)
  priorhps_sample = PriorHparams(
     w_prior_scale=tfd.InverseGamma(
      concentration=w_sampling_scale_alpha, 
      scale=w_sampling_scale_beta).sample(
      sample_shape=(num_samples,), seed=prng_keys[0]),
     a_prior_scale=tfd.InverseGamma(
      concentration=a_sampling_scale_alpha, 
      scale=a_sampling_scale_beta).sample(
      sample_shape=(num_samples,), seed=prng_keys[1]),
    kernel_amplitude=tfd.InverseGamma(
      concentration=kernel_sampling_amplitude_alpha, 
      scale=kernel_sampling_amplitude_beta).sample(
      sample_shape=(num_samples,), seed=prng_keys[3]),
     kernel_length_scale=jax.random.gamma(
      key=prng_keys[3],
      a=kernel_sampling_lengthscale_alpha, 
      shape=(num_samples,),
      )/kernel_sampling_lengthscale_beta,
     mu_prior_concentration=jnp.ones((num_samples,))*1.,
     mu_prior_rate=jnp.ones((num_samples,))*0.5,
     zeta_prior_a=jnp.ones((num_samples,))*1.,
     zeta_prior_b=jnp.ones((num_samples,))*1.,
  )
  return priorhps_sample
