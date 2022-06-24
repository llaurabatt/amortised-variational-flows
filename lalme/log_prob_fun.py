"""Probability functions for the LALME model."""

import jax
import jax.numpy as jnp

import distrax
from tensorflow_probability.substrates import jax as tfp

import haiku as hk

import modularbayes
from modularbayes import log1mexpm, force_symmetric
from modularbayes._src.typing import (Any, Array, Batch, Dict, List, Optional,
                                      PRNGKey, SmiEta)

tfd = tfp.distributions
kernels = tfp.math.psd_kernels


## Observational model ##
def log_prob_y_equal_1(
    gamma: Array,
    mixing_weights_list: List[Array],
    mixing_offset_list: List[Array],
    mu: Array,
    zeta: Array,
) -> List[Array]:
  # Get Probability fields
  # Linear transformation of GPs
  phi_prob_item_list_unbounded = [
      jnp.einsum(
          "bf,bp->fp",
          weights_i,
          gamma,
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
    posterior_sample_dict: Dict[str, Any],
    gamma_anchor: Array,
    gamma_floating: Array,
    smi_eta: Optional[SmiEta] = None,
) -> Array:
  """Log-probability of data given global parameters and loc_floating

  Integrate over multiple samples of gamma_profiles.

  Args:
    batch: Batch of data from the LALME dataset.
    posterior_sample_dict: Dictionary with samples of parameters in the LALME
      model.
    gamma_anchor: Array with samples from gamma on the location of anchor
      profiles
    gamma_floating: Array with samples from gamma on the location of
      floating profiles, corresponding to posterior_sample_dict['loc_floating'].
    smi_eta: Power to apply on likelihood elements.

  Returns:
    Matrix with pointwise log-likelihoods (item and profile dimensions expanded).
  """

  # num_samples_global = posterior_sample_dict['mu'].shape[0]

  # Concatenate samples of gamma for all profiles, anchor and floating
  gamma_profiles = jnp.concatenate([gamma_anchor, gamma_floating], axis=-1)

  # Computes log_prob_y_equal_1 over the sampled values gamma_profiles
  # First, iterate over samples from the global parameters
  def log_prob_y_along_global_fn(gamma_p):
    return jax.vmap(log_prob_y_equal_1)(
        gamma=gamma_p,
        mixing_weights_list=posterior_sample_dict['mixing_weights_list'],
        mixing_offset_list=posterior_sample_dict['mixing_offset_list'],
        mu=posterior_sample_dict['mu'],
        zeta=posterior_sample_dict['zeta'],
    )

  # Second, iterate over samples of gamma_profiles for each global sample
  log_prob_y_equal_1_pointwise_list = jax.vmap(log_prob_y_along_global_fn)(
      gamma_profiles.swapaxes(0, 1))
  # assert len(log_prob_y_equal_1_pointwise_list) == batch['num_items']
  # assert all([
  #     x.shape == (num_samples_gamma_profiles, num_samples_global,
  #                 batch['num_forms_tuple'][i], batch['num_profiles'])
  #     for i, x in enumerate(log_prob_y_equal_1_pointwise_list)
  # ])

  # Average log_prob_y_equal_1 over samples of gamma_profiles
  # and add over forms
  log_prob_y_item_profile_ = []
  for y_i, log_prob_y_eq_1_i in zip(batch['y'],
                                    log_prob_y_equal_1_pointwise_list):
    try:
      log_prob_y_item_profile_.append(
          jnp.where(y_i, log_prob_y_eq_1_i,
                    log1mexpm(-log_prob_y_eq_1_i)).mean(axis=0).sum(axis=1))
    except:
      breakpoint()
  log_prob_y_item_profile = jnp.stack(log_prob_y_item_profile_, axis=1)

  # assert log_prob_y_item_profile.shape == (num_samples_global,
  #                                          batch['num_items'],
  #                                          batch['num_profiles'])

  # The eta power is used to temper the likelihood of profiles and items
  if smi_eta is not None:
    if ('profiles' in smi_eta) and (smi_eta['profiles'] is not None):
      log_prob_y_item_profile = (
          log_prob_y_item_profile *
          jnp.expand_dims(smi_eta['profiles'], [0, 1]))
    if ('items' in smi_eta) and (smi_eta['items'] is not None):
      log_prob_y_item_profile = (
          log_prob_y_item_profile * jnp.expand_dims(smi_eta['items'], [0, 2]))
  log_prob = log_prob_y_item_profile.sum(axis=(1, 2))

  return log_prob


def sample_gamma_profiles_given_gamma_inducing(
    batch: Batch,
    posterior_sample_dict: Dict[str, Any],
    prng_key: PRNGKey,
    kernel_name: str,
    kernel_kwargs: Dict[str, Any],
    gp_jitter: float,
    num_samples_gamma_profiles: int,
    is_smi: bool,
    include_random_anchor: bool,
) -> Dict[str, Array]:
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

  ### Anchor profiles

  # Sample GP values on anchor locations, conditional on GP inducing values
  p_anchor_given_inducing = gp_F_given_U(
      u=posterior_sample_dict['gamma_inducing'],
      cov_x=jnp.expand_dims(batch['cov_anchor'], axis=0),
      cov_x_z=jnp.expand_dims(batch['cov_anchor_inducing'], axis=0),
      cov_z_inv=batch['cov_inducing_inv'],
      gp_jitter=gp_jitter,
  )
  gamma_sample_dict['gamma_anchor'] = p_anchor_given_inducing.sample(
      seed=next(prng_seq),
      sample_shape=(num_samples_gamma_profiles,)).swapaxes(0, 1)
  # assert gamma_sample_dict['gamma_anchor'].shape == (
  #     num_samples_global, num_samples_gamma_profiles, num_basis_gps,
  #     batch['num_profiles_anchor'])

  ### Floating profiles

  kernel = getattr(kernels, kernel_name)
  cov_fn1 = jax.vmap(kernel(**kernel_kwargs).matrix)
  cov_fn2 = jax.vmap(lambda x1: kernel(**kernel_kwargs).matrix(
      x1=x1, x2=batch['loc_inducing']))

  # Compute covariance (kernel) on floating locations.
  cov_floating = cov_fn1(
      posterior_sample_dict['loc_floating'],
      posterior_sample_dict['loc_floating'],
  )
  cov_floating_inducing = cov_fn2(posterior_sample_dict['loc_floating'])

  # Sample GP values on floating locations conditional on GP inducing values
  p_floating_given_inducing = gp_F_given_U(
      u=posterior_sample_dict['gamma_inducing'],
      cov_x=cov_floating,
      cov_x_z=cov_floating_inducing,
      cov_z_inv=batch['cov_inducing_inv'],
      gp_jitter=gp_jitter,
  )
  gamma_sample_dict['gamma_floating'] = p_floating_given_inducing.sample(
      seed=next(prng_seq),
      sample_shape=(num_samples_gamma_profiles,)).swapaxes(0, 1)
  # assert gamma_sample_dict['gamma_floating'].shape == (
  #     num_samples_global, num_samples_gamma_profiles, num_basis_gps,
  #     batch['num_profiles_floating'])

  if is_smi:
    # Compute covariance (kernel) on auxiliary floating locations.
    cov_floating_aux = cov_fn1(
        posterior_sample_dict['loc_floating_aux'],
        posterior_sample_dict['loc_floating_aux'],
    )
    cov_floating_inducing_aux = cov_fn2(
        posterior_sample_dict['loc_floating_aux'])

    # Sample GP values on floating locations conditional on GP inducing values
    # floating profiles
    p_floating_given_inducing_aux = gp_F_given_U(
        u=posterior_sample_dict['gamma_inducing'],
        cov_x=cov_floating_aux,
        cov_x_z=cov_floating_inducing_aux,
        cov_z_inv=batch['cov_inducing_inv'],
        gp_jitter=gp_jitter,
    )
    gamma_sample_dict['gamma_floating_aux'] = (
        p_floating_given_inducing_aux.sample(
            seed=next(prng_seq),
            sample_shape=(num_samples_gamma_profiles,))).swapaxes(0, 1)
    # assert gamma_sample_dict['gamma_floating_aux'].shape == (
    #     num_samples_global, num_samples_gamma_profiles, num_basis_gps,
    #     batch['num_profiles_floating'])

  ### Anchor as floating profiles
  if include_random_anchor:
    # Compute covariance (kernel) on random anchor locations.
    cov_random_anchor = cov_fn1(
        posterior_sample_dict['loc_random_anchor'],
        posterior_sample_dict['loc_random_anchor'],
    )
    cov_random_anchor_inducing = cov_fn2(
        posterior_sample_dict['loc_random_anchor'])

    # Sample GP values on floating locations conditional on GP inducing values
    p_random_anchor_given_inducing = gp_F_given_U(
        u=posterior_sample_dict['gamma_inducing'],
        cov_x=cov_random_anchor,
        cov_x_z=cov_random_anchor_inducing,
        cov_z_inv=batch['cov_inducing_inv'],
        gp_jitter=gp_jitter,
    )
    gamma_sample_dict[
        'gamma_random_anchor'] = p_random_anchor_given_inducing.sample(
            seed=next(prng_seq),
            sample_shape=(num_samples_gamma_profiles,)).swapaxes(0, 1)

  return gamma_sample_dict


def log_prob_joint(
    batch: Batch,
    posterior_sample_dict: Dict[str, Any],
    smi_eta: Optional[SmiEta] = None,
    random_anchor: bool = False,
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
    posterior_sample_dict: Dictionary with samples of parameters of LALME model,
      with keys:
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
      -loc_floating: Array of shape (num_samples, num_profiles, 2) with
        samples of the locations.
      -gamma_floating: Array of shape (num_samples, num_basis_gps,
        num_floating_points) with samples of the basis GPs at the corresponfing
        floating points.
    smi_eta : Dictionary specifying the SMI influence parameters with two keys:
      -profiles:
      -items:
  """

  # Get model dimensions
  num_samples, num_basis_gps, num_inducing_points = posterior_sample_dict[
      'gamma_inducing'].shape
  # pylint: disable=consider-using-generator
  num_forms_tuple = tuple(
      [x.shape[-1] for x in posterior_sample_dict['mixing_weights_list']])
  num_items = len(num_forms_tuple)

  ## Observational model ##
  # We integrate the likelihood over samples of gamma_profiles
  # Such samples are not part of the posterior approximation q(Theta), but we
  # assume they come in the dictionary 'posterior_sample_dict' for convenience.

  ## Priors ##

  # P(Gamma_Z) : Prior on the GPs on inducing points
  log_prob_gamma = distrax.Independent(
      modularbayes.MultivariateNormalTriL(
          loc=jnp.zeros((1, 1, num_inducing_points)),
          scale_tril=batch['cov_inducing_chol']),
      reinterpreted_batch_ndims=1).log_prob

  # P(loc_floating) : Prior on the floating locations
  # TODO: more likely where there are more items already
  def log_prob_locations(_):
    return jnp.zeros(num_samples)

  # P(W) : prior on the mixing weights
  def log_prob_weights(mixing_weights_list: List[Array]):
    # Induce sparsity on the representation of fields
    log_prob = jnp.stack([
        distrax.Independent(
            tfd.Laplace(
                loc=jnp.zeros(weights_i.shape[1:]),
                scale=w_prior_scale * jnp.ones(weights_i.shape[1:]),
            ),
            reinterpreted_batch_ndims=2).log_prob(weights_i)
        for weights_i in mixing_weights_list
    ],
                         axis=-1).sum(axis=-1)
    # # Flat prior
    # log_prob = jnp.zeros((mixing_weights_list[0].shape[0],))
    return log_prob

  # P(offset) : prior on the mixing offset
  def log_prob_offset(mixing_offset_list):
    # Gaussian prior
    log_prob = jnp.stack([
        distrax.Independent(
            distrax.Normal(
                loc=jnp.zeros(offset_i.shape[1:]),
                scale=a_prior_scale * jnp.ones(offset_i.shape[1:]),
            ),
            reinterpreted_batch_ndims=1).log_prob(offset_i)
        for offset_i in mixing_offset_list
    ],
                         axis=-1).sum(axis=-1)
    # # Flat prior
    # log_prob = jnp.zeros((mixing_offset_list[0].shape[0],))
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
          posterior_sample_dict=posterior_sample_dict,
          gamma_anchor=posterior_sample_dict[
              'gamma_' + ('random_' if random_anchor else '') + 'anchor'],
          gamma_floating=posterior_sample_dict['gamma_floating'],
          smi_eta=smi_eta,
      ) +
      # P(mu)
      log_prob_mu(posterior_sample_dict['mu']) +
      # P(zeta)
      log_prob_zeta(posterior_sample_dict['zeta']) +
      # P(Gamma_U)
      log_prob_gamma(posterior_sample_dict['gamma_inducing']) +
      # P(W)
      log_prob_weights(posterior_sample_dict['mixing_weights_list']) +
      # P(offset)
      log_prob_offset(posterior_sample_dict['mixing_offset_list']) +
      # P(loc_floating)
      log_prob_locations(posterior_sample_dict['loc_floating']))

  return log_prob


def conditional_gp_params_F_given_U(
    u: Array,
    cov_x_z: Array,
    cov_z_inv: Array,
    only_mean: bool = False,
    cov_x: Optional[Array] = None,
    gp_jitter: float = 1e-5,
):
  """Conditional GP distribution.

  Compute the conditional mean and covariance of a Gaussian Process, E(F|U) and
  Cov(F|U), where F=f(X) and U=f(Z) are the values of the GP evaluated at
  locations X and Z, respectively.
  Let K(,) be the kernel of the gaussian process. Define
    A = K(X,Z) @ inv(K(X,Z))
    B = A @ K(Z,X)
  The conditional distribution is a gaussian with
    E[F|U] = A @ U
    Cov[F|U] = K(X,X) - B

  Args:
    u: Array with the conditioning values U. Shape (num_samples, gp_dim, z_dim)
    cov_x_z: Array with the covariance of the GP between the new locations X
      and the conditioning locations Z. Shape (1 or num_samples, x_dim, z_dim).
    cov_z_inv: Array with the inverse of the covariance of the GP at the
      conditioning points.
    only_mean: Boolean indicating whether to only compute the conditional mean.
    cov_x: Array with the covariance of the GP at the new locations X. Only used
      if only_mean is False. Shape (1 or num_samples, x_dim, x_dim).
    gp_jitter: float, jitter to add to the diagonal of the conditional
      covariance matrix.
  """

  assert u.ndim == 3
  num_samples, gp_dim, z_dim = u.shape
  x_dim = cov_x_z.shape[-2]

  assert cov_x_z.ndim == 3
  assert cov_x_z.shape[-2:] == (x_dim, z_dim)
  assert cov_z_inv.shape == (z_dim, z_dim)

  matrix_a = jnp.einsum("rxz,zu->rxu", cov_x_z, cov_z_inv, precision='highest')

  # Conditional mean
  mean_f_given_u = jnp.einsum(
      'sxz,sgz->sgx',
      jnp.broadcast_to(matrix_a, (num_samples, x_dim, z_dim)),
      # u.shape == (num_samples, gp_dim, z_dim)
      u,
      precision='highest',
  )
  assert mean_f_given_u.shape == (num_samples, gp_dim, x_dim)

  if only_mean:
    cov_f_given_u = None
  else:
    assert cov_x.ndim == 3
    assert cov_x.shape[-2:] == (x_dim, x_dim)

    matrix_b = jnp.einsum(
        "rxz,rfz->rxf", matrix_a, cov_x_z, precision='highest')
    matrix_b = jax.vmap(force_symmetric)(matrix_b)

    # Conditional covariance
    # cov_f_given_u.shape == (num_samples, num_profiles_x, num_profiles_x)
    cov_f_given_u = cov_x - matrix_b
    cov_f_given_u = cov_f_given_u + (
        gp_jitter *
        jnp.broadcast_to(jnp.eye(cov_f_given_u.shape[-1]), cov_f_given_u.shape))

  return mean_f_given_u, cov_f_given_u


def gp_F_given_U(
    u: Array,
    cov_x: Array,
    cov_x_z: Array,
    cov_z_inv: Array,
    gp_jitter: float = 1e-5,
):
  """Conditional GP distribution.

  Compute the conditional distribution p(F|U) where F=f(X) and U=f(Z) are the
  values of a gaussian process evaluated at locations X and Z, respectively.
  Let K(,) be the kernel of the gaussian process. Define
    A = K(X,Z) @ inv(K(X,Z))
    B = A @ K(Z,X)
  The conditional distribution is a gaussian with
    E[F|U] = A @ U
    Cov[F|U] = K(X,X) - B

  Args:
    u: Array with the conditioning values U. Shape (num_samples, gp_dim, z_dim)
    cov_x: Array with the covariance of the GP at the new locations X.
      Shape (num_samples, x_dim, x_dim).
    cov_x_z: Array with the covariance of the GP between the new locations X
      and the conditioning locations Z. Shape (num_samples, x_dim, z_dim).
    cov_z_inv: Array with the inverse of the covariance of the GP at the
      conditioning points.
    gp_jitter: float, jitter to add to the diagonal of the conditional
      covariance matrix.
  """

  mean_f_given_u, cov_f_given_u = conditional_gp_params_F_given_U(
      u=u,
      cov_x_z=cov_x_z,
      cov_z_inv=cov_z_inv,
      only_mean=False,
      cov_x=cov_x,
      gp_jitter=gp_jitter,
  )

  # assert jnp.all(jax.vmap(jnp.diag)(cov_f_given_u) > 0)
  # assert jnp.all(jax.vmap(issymmetric)(cov_anchor_given_y))
  cov_tril_f_given_u = jax.vmap(jnp.linalg.cholesky)(cov_f_given_u)
  # assert ~jnp.any(jnp.isnan(cov_tril_f_given_u))
  # Expand dimension for independent GPs
  cov_tril_f_given_u = jnp.expand_dims(cov_tril_f_given_u, axis=1)
  assert cov_tril_f_given_u.ndim == 4
  x_dim = cov_x.shape[-1]
  assert cov_tril_f_given_u.shape[-2:] == (x_dim, x_dim)

  p_f_given_u = distrax.Independent(
      modularbayes.MultivariateNormalTriL(
          loc=mean_f_given_u, scale_tril=cov_tril_f_given_u),
      reinterpreted_batch_ndims=1)

  return p_f_given_u
