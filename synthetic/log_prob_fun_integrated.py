"""Probability functions for the Epidemiology model."""
from collections import namedtuple
import jax
import jax.numpy as jnp
import distrax
from tensorflow_probability.substrates import jax as tfp
import haiku as hk


tfd = tfp.distributions

from modularbayes._src.typing import (Any, Array, Batch, Dict, List, Optional,
                                 SmiEta)


_BasePriorHparams = namedtuple(
    "prior_hparams",
    field_names=('mu_prior_mean_m', 'mu_prior_scale_s', 
                 'sigma_prior_concentration', 'sigma_prior_scale'),
    defaults=(0., 1., 1.5, 0.5),
)

class PriorHparams(_BasePriorHparams):
    _defaults = {'mu_prior_mean_m': None, 'mu_prior_scale_s': None, 
                 'sigma_prior_concentration': None, 'sigma_prior_scale': None, 
      }

    @classmethod
    def set_defaults(cls, **new_defaults):
        cls._defaults.update(new_defaults)
        cls.__new__.__defaults__ = tuple(cls._defaults.values())





# Pointwise log-likelihood
def log_lik(mask_Y: Array, # n_obs, mu_dim
            Y: int, # n_obs, mu_dim
            mu: Array, # mu_dim
            sigma: Array, # sigma_dim = mu_dim
            ) -> Array:

  log_prob_Y = distrax.Normal(loc=mu, 
                    scale=sigma).log_prob(Y)
  return log_prob_Y*mask_Y

# vectorise over n_samples
log_lik_vectorised = jax.vmap(log_lik, in_axes=[None, None, 0, 0])


# Joint distribution (data and params)
def log_prob_joint(
    batch: Batch,
    posterior_sample_dict: Dict[str, Any],
    prior_hparams: Dict[str, Any],
    mask_Y: Array,
) -> Array:
  """Compute the joint probability for the HPV model.

  The joint log probability of the model is given by

  .. math::

    log_prob(Y,mu) &= log_prob(Y \| \mu) \\
                            &+ log_prob(\mu).

  Args:
    batch: Dictionary with the data. Must contains 4 items 'Z','Y','T' and 'N',
      each one of shape (n,).
    posterior_sample_dict: Dictionary with values for the model parameters. Must
      contain 2 items: 'phi' and 'theta', arrays with shapes (s, n) and (s, 2),
      respectively.
    smi_eta: Optional dictionary with the power to be applied on each module of
      the likelihood. Must contain 'modules', an array of shape
      (1,2).

  Output:
    Array of shape (s,) with log joint probability evaluated on each value of
    the model parameters.
  """
  n_obs = batch.shape[0]

  num_samples, mu_dim = posterior_sample_dict['mu'].shape
  _, sigma_dim = posterior_sample_dict['sigma'].shape

  # Batched log-likelihood
  
  mask_Y = jnp.reshape(mask_Y, (n_obs, mu_dim))
  loglik = log_lik_vectorised(mask_Y, batch, posterior_sample_dict['mu'], posterior_sample_dict['sigma']) 
  assert loglik.shape == (num_samples, n_obs, mu_dim)

  # Add over observations and Y_j dimension (n_groups)
  loglik = loglik.sum(axis=(1,2))

  assert loglik.shape == (num_samples,)

  # Define priors (sum over n_groups handled automatically by Independent class)
  log_prob_mu = distrax.Independent(distrax.Normal(loc=jnp.ones(mu_dim)*prior_hparams.mu_prior_mean_m, 
                                                   scale=prior_hparams.mu_prior_scale_s),
                                                   reinterpreted_batch_ndims=1).log_prob

  log_prob_sigma = distrax.Independent(tfd.InverseGamma(concentration=jnp.ones(mu_dim)*prior_hparams.sigma_prior_concentration,
                                                        scale=jnp.ones(sigma_dim)*prior_hparams.sigma_prior_scale),
                                                        reinterpreted_batch_ndims=1).log_prob
  
  

  # Everything together
  log_prob = (
      loglik + log_prob_mu(posterior_sample_dict['mu']) + log_prob_sigma(posterior_sample_dict['sigma']))
  assert log_prob.shape == (num_samples,)

  return {'log_prob':log_prob, 'log_lik':loglik, 'log_mu':log_prob_mu(posterior_sample_dict['mu']),
          'log_sigma':log_prob_sigma(posterior_sample_dict['sigma'])}



def sample_priorhparams_values(
    prng_seq: hk.PRNGSequence,
    num_samples: int,
    cond_hparams: List,
    mu_m_gaussian_mean: Optional[float],
    mu_m_gaussian_scale: Optional[float],
    mu_s_gamma_a_shape: Optional[float],
    mu_s_gamma_b_rate: Optional[float],
    sigma_hps_gamma_a_shape: Optional[float],
    sigma_hps_gamma_b_rate: Optional[float],
) -> PriorHparams:
  """Generate a sample of the prior hyperparameters values applicable to the model."""
  defaults = PriorHparams()
  priorhps_sample = PriorHparams(
     mu_prior_mean_m=tfd.Normal(loc=mu_m_gaussian_mean, scale=mu_m_gaussian_scale).sample(sample_shape=(num_samples,),
                                                                                                              seed=next(prng_seq)) 
                                  if 'mu_prior_mean_m' in cond_hparams
                                  else jnp.ones((num_samples,))*defaults.mu_prior_mean_m,
     mu_prior_scale_s=tfd.Gamma(concentration=mu_s_gamma_a_shape,rate=mu_s_gamma_b_rate).sample(sample_shape=(num_samples,),
                                                                                                              seed=next(prng_seq)) 
                                  if 'mu_prior_scale_s' in cond_hparams 
                                  else jnp.ones((num_samples,))*defaults.mu_prior_scale_s,
     sigma_prior_concentration=tfd.Gamma(concentration=sigma_hps_gamma_a_shape,rate=sigma_hps_gamma_b_rate).sample(sample_shape=(num_samples,),
                                                                                                                                seed=next(prng_seq))
                      if 'sigma_prior_concentration' in cond_hparams
                      else jnp.ones((num_samples,))*defaults.sigma_prior_concentration,
     sigma_prior_scale=tfd.Gamma(concentration=sigma_hps_gamma_a_shape,rate=sigma_hps_gamma_b_rate).sample(sample_shape=(num_samples,),
                                                                                                                                       seed=next(prng_seq))
                          if 'sigma_prior_scale' in cond_hparams
                          else jnp.ones((num_samples,))*defaults.sigma_prior_scale,
  )

  if len(cond_hparams)>0:
    cond_prior_hparams_values = []
    for k in priorhps_sample._fields:
      if k in cond_hparams:
        cond_prior_hparams_values.append(getattr(priorhps_sample, k)[:,None])
    cond_prior_hparams_values = jnp.hstack(cond_prior_hparams_values)
    return priorhps_sample, cond_prior_hparams_values
  else:
    return priorhps_sample, None

# def sample_priorhparams_values(
#     prng_seq: hk.PRNGSequence,
#     num_samples: int,
#     cond_hparams: List,
#     mu_m_gaussian_mean: Optional[float],
#     mu_m_gaussian_scale: Optional[float],
#     mu_s_gamma_a_shape: Optional[float],
#     mu_s_gamma_b_rate: Optional[float],
#     sigma_hps_gamma_a_shape: Optional[float],
#     sigma_hps_gamma_b_rate: Optional[float],
# ) -> PriorHparams:
#   """Generate a sample of the prior hyperparameters values applicable to the model."""
#   defaults = PriorHparams()
#   priorhps_sample = PriorHparams(
#      mu_prior_mean_m=tfd.Normal(loc=mu_m_gaussian_mean, scale=mu_m_gaussian_scale).sample(sample_shape=(num_samples,),
#                                                                                                               seed=next(prng_seq)) 
#                                   if 'mu_prior_mean_m' in cond_hparams
#                                   else jnp.ones((num_samples,))*defaults.mu_prior_mean_m,
#      mu_prior_scale_s=tfd.Gamma(concentration=2,rate=1/2).sample(sample_shape=(num_samples,),
#                                                                                                                                 seed=next(prng_seq))
#                                   if 'mu_prior_scale_s' in cond_hparams 
#                                   else jnp.ones((num_samples,))*defaults.mu_prior_scale_s,
#      sigma_prior_concentration=tfd.Gamma(concentration=sigma_hps_gamma_a_shape,rate=sigma_hps_gamma_b_rate).sample(sample_shape=(num_samples,),
#                                                                                                                                 seed=next(prng_seq))
#                       if 'sigma_prior_concentration' in cond_hparams
#                       else jnp.ones((num_samples,))*defaults.sigma_prior_concentration,
#      sigma_prior_scale=tfd.Gamma(concentration=sigma_hps_gamma_a_shape,rate=sigma_hps_gamma_b_rate).sample(sample_shape=(num_samples,),
#                                                                                                                                        seed=next(prng_seq))

#                           if 'sigma_prior_scale' in cond_hparams
#                           else jnp.ones((num_samples,))*defaults.sigma_prior_scale,
#   )

#   if len(cond_hparams)>0:
#     cond_prior_hparams_values = []
#     for k in priorhps_sample._fields:
#       if k in cond_hparams:
#         cond_prior_hparams_values.append(getattr(priorhps_sample, k)[:,None])
#     cond_prior_hparams_values = jnp.hstack(cond_prior_hparams_values)
#     return priorhps_sample, cond_prior_hparams_values
#   else:
#     return priorhps_sample, None

