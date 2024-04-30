"""Flow model for the LALME model."""

import math
import pathlib

from absl import logging
import arviz as az
from c2st import c2st_scores
import numpy as np
from matplotlib import pyplot as plt
from arviz import InferenceData

from flax.metrics import tensorboard
import syne_tune

import jax
from jax import numpy as jnp

import haiku as hk
import optax
import os
import ot
import pickle
import psutil

from tensorflow_probability.substrates import jax as tfp
import wandb

import log_prob_fun_allhp
from log_prob_fun_allhp import (ModelParamsGlobal, ModelParamsLocations, PriorHparams, fix_priorhparams_values,
                                sample_priorhparams_values, logprob_rho)
import flows
import plot
from train_flow_allhp import (load_data, make_optimizer, get_inducing_points,
                        error_locations_estimate, logprob_lalme)

from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, initial_state_ckpt, update_states, update_state,
                          save_checkpoint, plot_to_image, normalize_images)
from modularbayes._src.typing import (Any, Array, Batch, Callable, ConfigDict, Dict, List, NamedTuple,
                                      Optional, PRNGKey, SmiEta, SummaryWriter,
                                      Tuple)

kernels = tfp.math.psd_kernels

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)

def make_optimizer_hparams(learning_rate: float) -> optax.GradientTransformation:
  optimizer = optax.adabelief(learning_rate=learning_rate)
  return optimizer

def q_distr_global(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    cond_values: Optional[Array],
    num_samples: float,
    # eta: Array,
) -> Dict[str, Any]:
  """Sample from the posterior of the LALME model.

  The posterior distribution is factorized as:
    q(mu, zeta, Gamma_U, a, W) q(loc_floating | Gamma_U, a, W)

  The first term is the distribution of "global" parameters, and the second term
  is the posterior for locations of floating profiles.

  We make this separation so we are able to use trained posteriors for the
  global parameters even after changing the number of floating profiles.

  Args:
    flow_name: String specifying the type of flow to be used.
    flow_kwargs: Dictionary with keyword arguments for the flow function.
    kernel_name: String specifiying the Kernel function to be used.
    kernel_kwargs: Dictionary with keyword arguments for the Kernel function.
    sample_shape: Sample shape.
    batch: Data batch.
  """

  # params = state.params
  # prng_key=next(prng_seq)
  # batch = train_ds
  # flow_sample, flow_log_prob = q_distr_global_sample.apply(
  #      params, prng_key, sample_shape=(config.num_samples_elbo,), batch=batch)

  q_distr_out = {}
  # if not cond_values:
  #   cond_values = None

  ## Posterior for global parameters ##
  # Define normalizing flow
  q_distr = getattr(flows, flow_name + '_global_params')(**flow_kwargs)

  # num_samples = cond_values.shape[0]

  # Sample from flow
  (sample_flow_concat, sample_logprob,
   sample_base) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=[cond_values, None],
   )

  # Split flow into model parameters
  q_distr_out['sample'] = flows.split_flow_global_params(
      samples=sample_flow_concat,
      **flow_kwargs,
  )

  # Log_probabilities of the sample
  q_distr_out['sample_logprob'] = sample_logprob

  # Sample from base distribution are preserved
  # These are used for the posterior of profiles locations
  q_distr_out['sample_base'] = sample_base

  return q_distr_out


def q_distr_loc_floating(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    global_params_base_sample: Array,
    cond_values:Array,
    # eta: Array,
    name: str = 'loc_floating',
) -> Dict[str, Any]:
  """Sample from the posterior of floating locations

  Conditional on global parameters.

  The posterior distribution is factorized as:
    q(mu, zeta, Gamma_U, a, W) q(loc_floating | Gamma_U, a, W)

  The first term is the distribution of "global" parameters, and the second term
  is the posterior for locations of floating profiles.

  We make this separation so we are able to use trained posteriors for the
  global parameters even after changing the number of floating profiles.

  Args:
    flow_name: String specifying the type of flow to be used.
    flow_kwargs: Dictionary with keyword arguments for the flow function.
    kernel_name: String specifiying the Kernel function to be used.
    kernel_kwargs: Dictionary with keyword arguments for the Kernel function.
    sample_shape: Sample shape.
    batch: Data batch.
  """

  # params = state.params
  # prng_key=next(prng_seq)
  # batch = train_ds
  # flow_sample, flow_log_prob = q_distr_global_sample.apply(
  #      params, prng_key, sample_shape=(config.num_samples_elbo,), batch=batch)

  q_distr_out = {}

  # if not cond_values:
  #   cond_values = None

  ## Posterior for locations of floating profiles ##
  # Define normalizing flow
  q_distr = getattr(flows, flow_name + '_locations')(
      num_profiles=flow_kwargs['num_profiles_floating'], **flow_kwargs)

  # Sample from flow
  num_samples = global_params_base_sample.shape[0]
  sample_flow_concat, sample_logprob = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
      context=[cond_values, global_params_base_sample],
  )

  # Split flow into model parameters
  # (and add to existing posterior_sample_dict)
  q_distr_out['sample'] = flows.split_flow_locations(
      samples=sample_flow_concat,
      num_profiles=flow_kwargs['num_profiles_floating'],
      name=name,
  )

  q_distr_out['sample_logprob'] = sample_logprob

  return q_distr_out


def q_distr_loc_random_anchor(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    global_params_base_sample: Array,
    cond_values: Array,
    # eta: Array,
) -> Dict[str, Any]:
  """Sample from the posterior of floating locations

  Conditional on global parameters.

  The posterior distribution is factorized as:
    q(mu, zeta, Gamma_U, a, W) q(loc_floating | Gamma_U, a, W)

  The first term is the distribution of "global" parameters, and the second term
  is the posterior for locations of floating profiles.

  We make this separation so we are able to use trained posteriors for the
  global parameters even after changing the number of floating profiles.

  Args:
    flow_name: String specifying the type of flow to be used.
    flow_kwargs: Dictionary with keyword arguments for the flow function.
    kernel_name: String specifiying the Kernel function to be used.
    kernel_kwargs: Dictionary with keyword arguments for the Kernel function.
    sample_shape: Sample shape.
    batch: Data batch.
  """

  q_distr_out = {}
  # if not cond_values:
  #   cond_values = None

  ## Posterior for locations of anchor profiles treated as unkbowb locations##

  # Define normalizing flow
  q_distr = getattr(flows, flow_name + '_locations')(
      num_profiles=flow_kwargs['num_profiles_anchor'], **flow_kwargs)

  # Sample from flow
  num_samples = global_params_base_sample.shape[0]
  (sample_flow_concat, sample_logprob) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
      context=[cond_values, global_params_base_sample],
  )

  # Split flow into model parameters
  # (and add to existing posterior_sample_dict)
  q_distr_out['sample'] = flows.split_flow_locations(
      samples=sample_flow_concat,
      num_profiles=flow_kwargs['num_profiles_anchor'],
      name='loc_random_anchor',
  )

  # Log_probabilities of the sample
  q_distr_out['sample_logprob'] = sample_logprob

  return q_distr_out


def sample_all_flows(
    params_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    num_samples: float,
    # smi_eta: SmiEta,
    cond_values: Optional[Array],
    include_random_anchor: bool,
) -> Dict[str, Any]:
  """Generate a sample from the entire flow posterior."""

  prng_seq = hk.PRNGSequence(prng_key)

  q_distr_out = {}
    
  # if not cond_values:
  #   cond_values = None

  # Global parameters
  q_distr_out_global = hk.transform(q_distr_global).apply(
      params_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      cond_values=cond_values,
    #   eta=smi_eta['profiles'],
      num_samples=num_samples,
  )
  q_distr_out.update({f"global_{k}": v for k, v in q_distr_out_global.items()})

  # Floating profiles locations
  q_distr_out_loc_floating = hk.transform(q_distr_loc_floating).apply(
      params_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      global_params_base_sample=q_distr_out_global['sample_base'],
      cond_values=cond_values,
    #   eta=smi_eta['profiles'],
      name='loc_floating',
  )
  q_distr_out['loc_floating_logprob'] = q_distr_out_loc_floating[
      'sample_logprob']

  # Auxiliary Floating profiles locations
  q_distr_out_loc_floating_aux = hk.transform(q_distr_loc_floating).apply(
      params_tuple[2],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      global_params_base_sample=q_distr_out_global['sample_base'],
      cond_values=cond_values,
    #   eta=smi_eta['profiles'],
      name='loc_floating_aux',
  )
  q_distr_out['locations_sample'] = ModelParamsLocations(
      loc_floating=q_distr_out_loc_floating['sample'].loc_floating,
      loc_floating_aux=q_distr_out_loc_floating_aux['sample'].loc_floating_aux,
  )
  q_distr_out['loc_floating_aux_logprob'] = q_distr_out_loc_floating_aux[
      'sample_logprob']

  # Anchor profiles locations
  if include_random_anchor:
    q_distr_out_loc_random_anchor = hk.transform(
        q_distr_loc_random_anchor).apply(
            params_tuple[3],
            next(prng_seq),
            flow_name=flow_name,
            flow_kwargs=flow_kwargs,
            global_params_base_sample=q_distr_out_global['sample_base'],
            cond_values=cond_values,
            # eta=smi_eta['profiles'],
        )
    q_distr_out['locations_sample'] = ModelParamsLocations(
        loc_floating=q_distr_out_loc_floating['sample'].loc_floating,
        loc_floating_aux=q_distr_out_loc_floating_aux['sample']
        .loc_floating_aux,
        loc_random_anchor=q_distr_out_loc_random_anchor['sample']
        .loc_random_anchor,
    )
    q_distr_out['loc_random_anchor_logprob'] = q_distr_out_loc_random_anchor[
        'sample_logprob']

  return q_distr_out

def sample_locations_floating(
    state_list: List[TrainState],
    cond_values: Array,
    lalme_dataset: dict,
    prng_key: PRNGKey,
    num_samples: float,
    config: ConfigDict,
    num_samples_chunk: int = 1_000,
) -> InferenceData:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)


  locations_sample = []

#   assert all(cond_values.ndim == 2) # what is this line for?

  # Sampling divided into chunks, to avoid OOM on GPU
  # Split etas into chunks
  split_idx_ = np.arange(num_samples_chunk, num_samples,
                        num_samples_chunk).tolist()
  
  if cond_values is not None:
    cond_values_chunked_ = jnp.split(cond_values, split_idx_, axis=0)
  else:
    cond_values_chunked_ = [None]*len(split_idx_)

  for cond_val_ in cond_values_chunked_:#,prior_hparams_chunked_):
    # Sample from variational posterior
    q_distr_out = sample_all_flows(
        params_tuple=[state.params for state in state_list],
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        cond_values=cond_val_,
        # smi_eta=smi_eta_,
        include_random_anchor=config.include_random_anchor,
        num_samples=num_samples_chunk,
    )

    locations_sample.append(q_distr_out['locations_sample'])

  locations_sample = jax.tree_map(  # pylint: disable=no-value-for-parameter
      lambda *x: jnp.concatenate([xi[None, ...] for xi in x], axis=1),
      *locations_sample)

  # produce az object
  samples_dict = {'loc_floating':locations_sample._asdict()['loc_floating']}
  num_chains = samples_dict['loc_floating'].shape[0]
  num_samples = samples_dict['loc_floating'].shape[1]

  coords_lalme= {"LP_floating":
            lalme_dataset['LP'][-lalme_dataset['num_profiles_floating']:],
        "coords": ["x", "y"],
    }

  assert locations_sample.loc_floating.ndim == 4 and (
          locations_sample.loc_floating.shape[0] <
          locations_sample.loc_floating.shape[1]), (
              "Arrays in model_params_locations" +
              "are expected to have shapes:" + "(num_chains, num_samples, ...)")
  dims_lalme = {"loc_floating": ["LP_floating", "coords"]}

  # Remove empty entries in samples_dict
  for k, v in samples_dict.copy().items():
    if v is None:
      del samples_dict[k]

  lalme_az_locations = az.convert_to_inference_data(
      samples_dict,
      coords=coords_lalme,
      dims=dims_lalme,
  )
  
  return locations_sample.loc_floating, lalme_az_locations


def sample_lalme_az(
    state_list: List[TrainState],
    batch: Batch,
    cond_values: Array,
    prior_hparams:PriorHparams,
    # smi_eta: SmiEta,
    prng_key: PRNGKey,
    num_samples: float,
    config: ConfigDict,
    lalme_dataset: Dict[str, Any],
    include_gamma: bool = False,
    num_samples_chunk: int = 1_000,
) -> InferenceData:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)


  global_sample = []
  locations_sample = []
  gamma_sample = []

#   assert all(cond_values.ndim == 2) # what is this line for?

  # Sampling divided into chunks, to avoid OOM on GPU
  # Split etas into chunks
  split_idx_ = np.arange(num_samples_chunk, num_samples,
                        num_samples_chunk).tolist()
  
  if cond_values is not None:
    cond_values_chunked_ = jnp.split(cond_values, split_idx_, axis=0)
  else:
    cond_values_chunked_ = [None]*len(split_idx_)
  # prior_hparams_chunked_ = jax.tree_map(lambda x: jnp.split(x, split_idx_, axis=0),
  #                                  prior_hparams)

#   split_idx_ = np.arange(num_samples_chunk, smi_eta['profiles'].shape[0],
#                          num_samples_chunk).tolist() # list of idxs multiple of num_samples_chunk up to tot samples
#   smi_eta_chunked_ = jax.tree_map(lambda x: jnp.split(x, split_idx_, axis=0),
#                                   smi_eta) # turns into {key: list of arrays of num_samples_chunk samples}
#   # dict of lists -> list of dicts
#   smi_eta_chunked_ = [
#       dict(zip(smi_eta_chunked_, t)) for t in zip(*smi_eta_chunked_.values()) # turns into list of len len(split_idx_) of {key: array of num_samples_chunk samples}
#   ]

  for cond_val_ in cond_values_chunked_:#,prior_hparams_chunked_):
    # Sample from variational posterior
    q_distr_out = sample_all_flows(
        params_tuple=[state.params for state in state_list],
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        cond_values=cond_val_,
        # smi_eta=smi_eta_,
        include_random_anchor=config.include_random_anchor,
        num_samples=num_samples_chunk,
    )

    global_sample.append(q_distr_out['global_sample'])
    locations_sample.append(q_distr_out['locations_sample'])

    if include_gamma:
      # Get a sample of the basis GPs on profiles locations
      # conditional on values at the inducing locations.
      gamma_sample_, _, _ = jax.vmap(
          lambda key_, global_, locations_: log_prob_fun_allhp.
          sample_gamma_profiles_given_gamma_inducing(
              batch=batch,
              model_params_global=global_,
              model_params_locations=locations_,
              prng_key=key_,
              prior_hparams=PriorHparams(*prior_hparams),#(*prior_hparams_[0]),
              kernel_name=config.kernel_name,
              # kernel_kwargs=config.kernel_kwargs,
              gp_jitter=config.gp_jitter,
              num_profiles_anchor=config.num_profiles_anchor,
              num_inducing_points=config.num_inducing_points,
              include_random_anchor=config.include_random_anchor,
          ))(
              jax.random.split(next(prng_seq), cond_val_.shape[0]),
              q_distr_out['global_sample'],
              q_distr_out['locations_sample'],
          )

      gamma_sample.append(gamma_sample_)

  global_sample = jax.tree_map(  # pylint: disable=no-value-for-parameter
      lambda *x: jnp.concatenate([xi[None, ...] for xi in x], axis=1),
      *global_sample)
  locations_sample = jax.tree_map(  # pylint: disable=no-value-for-parameter
      lambda *x: jnp.concatenate([xi[None, ...] for xi in x], axis=1),
      *locations_sample)
  if include_gamma:
    gamma_sample = jax.tree_map(  # pylint: disable=no-value-for-parameter
        lambda *x: jnp.concatenate([xi[None, ...] for xi in x], axis=1),
        *gamma_sample)
  else:
    gamma_sample = None
  


  ### Posterior visualisation with Arviz

  # Create InferenceData object
  lalme_az = plot.lalme_az_from_samples(
      lalme_dataset=lalme_dataset,
      model_params_global=global_sample,
      model_params_locations=locations_sample,
      model_params_gamma=gamma_sample,
  )

  return lalme_az

def get_cond_values(
    cond_hparams_names: List,
    num_samples: float,
    eta_init: Optional[float],
    prior_hparams_init: Optional[NamedTuple]=None,
    ):
  if 'eta' in cond_hparams_names:
    eta_init = jnp.ones((num_samples, ))*eta_init

  cond_prior_hparams_names = cond_hparams_names.copy()
  try:
    cond_prior_hparams_names.remove('eta')
  except:
    pass
  if len(cond_prior_hparams_names)>0:
    cond_prior_hparams_init = []
    for k in prior_hparams_init._fields:
      if k in cond_hparams_names:
          val = getattr(prior_hparams_init, k)
          sample_val = jnp.ones((num_samples,))*val
          cond_prior_hparams_init.append(sample_val[:,None])
    cond_prior_hparams_init = jnp.hstack(cond_prior_hparams_init)

  # cond values for flows
  if (('eta' in cond_hparams_names) & (len(cond_hparams_names)>1)):
    cond_values_init = jnp.hstack([cond_prior_hparams_init, eta_init[:,None]])
  elif 'eta' in cond_hparams_names: 
    cond_values_init = eta_init[:,None]
  else:
    cond_values_init = cond_prior_hparams_init
  return cond_values_init
  
def elbo_estimate_along_eta(
    params_tuple: Tuple[hk.Params],
    batch: Optional[Batch],
    prng_key: PRNGKey,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    cond_hparams: List,
    sample_priorhparams_fn:Callable,
    sample_priorhparams_kwargs: Dict[str, Any],
    include_random_anchor: bool,
    profile_is_anchor: Array,
    num_profiles_anchor:int,
    num_inducing_points:int,
    eta_sampling_a: Optional[float],
    eta_sampling_b: Optional[float],
    # prior_hparams: Dict[str, Any],
    kernel_name: Optional[str] = None,
    # kernel_kwargs: Optional[Dict[str, Any]] = None,
    num_samples_gamma_profiles: Optional[int] = 0,
    gp_jitter: Optional[float] = None,
    eta_fixed: Optional[float] = 1.,
    training: Optional[bool] = True,
    cond_hparams_values_evaluation: Optional[Dict] = {},
) -> Dict[str, Array]:
  # params_tuple = [state.params for state in state_list]

  prng_seq = hk.PRNGSequence(prng_key)

  ## hyperparams for log-joint:
  # Sample hparams

  if training:
    prior_hparams_sample, cond_prior_hparams_values = sample_priorhparams_fn(
        prng_seq=prng_seq,
        num_samples=num_samples, cond_hparams=cond_hparams,
        **sample_priorhparams_kwargs,
    )


    # Sample eta values
    etas_profiles_floating = jax.random.beta(key=next(prng_seq),
                                            a=eta_sampling_a,
                                            b=eta_sampling_b,
                                            shape=(num_samples,)) if 'eta' in cond_hparams else jnp.ones((num_samples,))*eta_fixed
  elif not training:
    # evaluation
    prior_hparams_sample, cond_prior_hparams_values = fix_priorhparams_values(
        num_samples=num_samples, 
        cond_hparams=cond_hparams,
        cond_hparams_values_evaluation=cond_hparams_values_evaluation,
    )
    if 'eta' in cond_hparams:
      etas_profiles_floating = jnp.ones((num_samples,))*cond_hparams_values_evaluation['eta']
    else:
      etas_profiles_floating = jnp.ones((num_samples,))

  eta_profiles = jax.vmap(lambda eta_: jnp.where(
              profile_is_anchor,1.,eta_, ))(etas_profiles_floating) #(n_samples, 367)
  eta_items = jnp.ones((num_samples, len(batch['num_forms_tuple']))) #(n_samples, 71)

  smi_eta_elbo = {
      'profiles':eta_profiles,
      'items':eta_items,
  }

  if cond_hparams:
  # # cond values for flows
    cond_values =jnp.hstack([cond_prior_hparams_values, etas_profiles_floating[:,None]]) if (('eta' in cond_hparams) & (len(cond_hparams)>1)) else (etas_profiles_floating[:,None] if 'eta' in cond_hparams else cond_prior_hparams_values)

  else:
    cond_values = None
  # cond_values = jnp.hstack([jnp.stack(prior_hparams_sample, axis=-1),
  #                           eta_profiles, eta_items,
  #                           ]) #(n_samples, n_hps+367+71)

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      prng_key=next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      cond_values=cond_values,
    #   smi_eta=smi_eta_elbo,
      include_random_anchor=include_random_anchor,
      num_samples=num_samples,
  )

  # ELBO stage 1: Power posterior
  locations_stg1_ = ModelParamsLocations(
      loc_floating=q_distr_out['locations_sample'].loc_floating_aux,
      loc_floating_aux=None,
      loc_random_anchor=None,
  )
  log_prob_joint_stg1 = jax.vmap(
      lambda key_, global_, locations_, prior_hparams_i, smi_eta_: logprob_lalme(
          batch=batch,
          prng_key=key_,
          model_params_global=global_,
          model_params_locations=locations_,
          prior_hparams=prior_hparams_i,
          kernel_name=kernel_name,
          # kernel_kwargs=kernel_kwargs,
          num_samples_gamma_profiles=num_samples_gamma_profiles,
          smi_eta_profiles=smi_eta_,
          gp_jitter=gp_jitter,
          num_profiles_anchor=num_profiles_anchor,
          num_inducing_points=num_inducing_points,
          random_anchor=False,
      ))(
          jax.random.split(next(prng_seq), num_samples),
          q_distr_out['global_sample'],
          locations_stg1_,
          prior_hparams_sample,
          smi_eta_elbo['profiles'],
      )

  log_q_stg1 = (
      q_distr_out['global_sample_logprob'] +
      q_distr_out['loc_floating_aux_logprob'])

  elbo_stg1 = log_prob_joint_stg1 - log_q_stg1

  # ELBO stage 2: Refit locations floating profiles
  locations_stg2_ = ModelParamsLocations(
      loc_floating=q_distr_out['locations_sample'].loc_floating,
      loc_floating_aux=None,
      loc_random_anchor=None,
  )
  log_prob_joint_stg2 = jax.vmap(
      lambda key_, global_, locations_, prior_hparams_i : logprob_lalme(
          batch=batch,
          prng_key=key_,
          model_params_global=global_,
          model_params_locations=locations_,
          prior_hparams=prior_hparams_i,
          kernel_name=kernel_name,
          # kernel_kwargs=kernel_kwargs,
          num_samples_gamma_profiles=num_samples_gamma_profiles,
          smi_eta_profiles=None,
          gp_jitter=gp_jitter,
          num_profiles_anchor=num_profiles_anchor,
          num_inducing_points=num_inducing_points,
          random_anchor=False,
      ))(
          jax.random.split(next(prng_seq), num_samples),
          jax.lax.stop_gradient(q_distr_out['global_sample']),
          locations_stg2_,
          prior_hparams_sample,
      )
  log_q_stg2 = (
      jax.lax.stop_gradient(q_distr_out['global_sample_logprob']) +
      q_distr_out['loc_floating_logprob'])

  elbo_stg2 = log_prob_joint_stg2 - log_q_stg2

  # ELBO stage 3: fit posteriors for locations of anchor profiles
  # mainly for model evaluation (choosing eta)
  if include_random_anchor:
    locations_stg3_ = ModelParamsLocations(
        loc_floating=jax.lax.stop_gradient(
            q_distr_out['locations_sample'].loc_floating),
        loc_floating_aux=None,
        loc_random_anchor=q_distr_out['locations_sample'].loc_random_anchor,
    )
    log_prob_joint_stg3 = jax.vmap(
        lambda key_, global_, locations_, prior_hparams_i: logprob_lalme(
            batch=batch,
            prng_key=key_,
            model_params_global=global_,
            model_params_locations=locations_,
            prior_hparams=prior_hparams_i,
            kernel_name=kernel_name,
            # kernel_kwargs=kernel_kwargs,
            num_samples_gamma_profiles=num_samples_gamma_profiles,
            smi_eta_profiles=None,
            gp_jitter=gp_jitter,
            num_profiles_anchor=num_profiles_anchor,
            num_inducing_points=num_inducing_points,
            random_anchor=True,
        ))(
            jax.random.split(next(prng_seq), num_samples),
            jax.lax.stop_gradient(q_distr_out['global_sample']),
            locations_stg3_,
            prior_hparams_sample,
        )

    log_q_stg3 = (
        jax.lax.stop_gradient(q_distr_out['global_sample_logprob']) +
        jax.lax.stop_gradient(q_distr_out['loc_floating_logprob']) +
        q_distr_out['loc_random_anchor_logprob'])

    elbo_stg3 = log_prob_joint_stg3 - log_q_stg3
  else:
    elbo_stg3 = 0.

  elbo_dict = {
      'stage_1': elbo_stg1,
      'stage_2': elbo_stg2,
      'stage_3': elbo_stg3,
  }

  return elbo_dict


def loss(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_estimate_along_eta(
      params_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(
      jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2'] +
                  elbo_dict['stage_3']))

  return loss_avg


def error_locations_vector_estimate(
    state_list: List[TrainState],
    batch: Optional[Batch],
    prng_key: PRNGKey,
    config: ConfigDict,
    cond_hparams_names: List,
    eta_eval_grid: Array,
    num_samples: int,
) -> Dict[str, Array]:
  """Compute average distance error along eta_eval_grid.
  
  Note:
    This could be computed using vmap, but ran into OOM issues.
  """

  prng_seq = hk.PRNGSequence(prng_key)
  error_grid = []

  prior_defaults = PriorHparams()
  # prior_defaults = jnp.stack(PriorHparams())
  # prior_hparams=jnp.ones((num_samples, 
  #                                 len(prior_defaults)))*prior_defaults # init params right?
 
  LPs = batch['LP']
  LPs_split = np.split(
  LPs,
  np.cumsum(batch['num_profiles_split']),
  )[:-1]

  if config.floating_anchor_copies:
      train_idxs = jnp.where(jnp.isin(LPs_split[3], LPs_split[0]*1000))[0]
  else:
      train_idxs = None

  if config.ad_hoc_val_profiles:
      val_idxs = jnp.where(jnp.isin(LPs_split[1], jnp.array(config.ad_hoc_val_list)))[0]
  else:
      val_idxs = None
    
  for eta_i in eta_eval_grid:
    if cond_hparams_names:
      cond_values = get_cond_values(cond_hparams_names=cond_hparams_names,
                        num_samples=num_samples,
                        eta_init=eta_i,
                        prior_hparams_init=prior_defaults
                        )
    else:
      cond_values=None
    # # eta_i = eta_eval_grid[0]
    # eta_i_profiles = eta_i * jnp.ones((num_samples, config.num_profiles))
    # eta_i_profiles = jax.vmap(lambda eta_: jnp.where(
    #             jnp.arange(config.num_profiles) < config.num_profiles_anchor,
    #             1.,eta_, ))(eta_i_profiles)
    
    # eta_i_items = jnp.ones((num_samples, len(config.num_forms_tuple)))
    # smi_eta_ = {
    #     'profiles':eta_i_profiles,
    #     'items':eta_i_items,
    # }
    # cond_values = jnp.hstack([prior_hparams, eta_i_profiles, eta_i_items])

    q_distr_out = sample_all_flows(
        params_tuple=[state.params for state in state_list],
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        cond_values=cond_values,
        # smi_eta=smi_eta_,
        include_random_anchor=config.include_random_anchor,
        num_samples=num_samples,
    )

    error_grid.append(
        error_locations_estimate(
            locations_sample=q_distr_out['locations_sample'],
            num_profiles_split=batch['num_profiles_split'],
            loc=batch['loc'],
            # LPs=batch['LP'],
            floating_anchor_copies=config.floating_anchor_copies,
            ad_hoc_val_profiles=config.ad_hoc_val_profiles,
            train_idxs=train_idxs,
            val_idxs=val_idxs,
            # batch=batch,
        ))

  error_loc_dict = jax.tree_map(lambda *x: jnp.stack(x, axis=0), *error_grid)  # pylint: disable=no-value-for-parameter

  return error_loc_dict


def log_images(
    state_list: List[TrainState],
    prng_key: PRNGKey,
    config: ConfigDict,
    lalme_dataset: Dict[str, Any],
    batch: Batch,
    cond_hparams_names:List,
    show_mu: bool = False,
    show_zeta: bool = False,
    show_basis_fields: bool = False,
    show_W_items: Optional[List[str]] = None,
    show_a_items: Optional[List[str]] = None,
    lp_floating: Optional[List[int]] = None,
    lp_floating_traces: Optional[List[int]] = None,
    lp_floating_grid10: Optional[List[int]] = None,
    lp_anchor_val_grid30:Optional[List[int]] = None,
    lp_anchor_val_grid28:Optional[List[int]] = None,
    lp_anchor_val_grid10:Optional[List[int]] = None,
    lp_anchor_val_grid21:Optional[List[int]] = None,
    mcmc_img: Optional[str] = None,
    lp_random_anchor: Optional[List[int]] = None,
    lp_random_anchor_grid10: Optional[List[int]] = None,
    show_lp_anchor_val: Optional[bool] = False,
    show_lp_anchor_test: Optional[bool] = False,
    show_location_priorhp_compare: Optional[bool] = False,
    loc_inducing: Optional[Array] = None,
    show_eval_metric: bool = False,
    eta_eval_grid: Optional[Array] = None,
    num_samples_chunk: int = 1_000,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  profile_is_anchor = jnp.arange(
      config.num_profiles) < config.num_profiles_anchor

  # List of LP for validation
  if show_lp_anchor_val:
    lp_anchor_val = np.split(lalme_dataset['LP'],
                             np.cumsum(lalme_dataset['num_profiles_split']))[1]
  else:
    lp_anchor_val = None
  # List of LP for test
  if show_lp_anchor_test:
    lp_anchor_test = np.split(lalme_dataset['LP'],
                              np.cumsum(lalme_dataset['num_profiles_split']))[2]
  else:
    lp_anchor_test = None
    
  prior_defaults = PriorHparams()
  prior_defaults = jnp.stack(PriorHparams())
  prior_hparams=jnp.ones((config.num_samples_plot, 
                                  len(prior_defaults)))*prior_defaults # init params right?

  # Plot posterior samples
  for eta_i in config.eta_plot:

    eta_i_profiles = jax.vmap(lambda eta_: jnp.where(
        profile_is_anchor,1., eta_,
    ))(eta_i * jnp.ones((config.num_samples_plot, config.num_profiles)))

    eta_i_items = jnp.ones((config.num_samples_plot, len(config.num_forms_tuple)))
    smi_eta_ = {
        'profiles':eta_i_profiles,
        'items':eta_i_items,
    }
    cond_values = jnp.hstack([prior_hparams,eta_i_profiles,eta_i_items])
    if config.prior_hparams_plot_optim:
      prior_hparams_init = PriorHparams(*config.prior_hparams_plot_optim)
      prior_hparams_init_vals = jnp.array(config.prior_hparams_plot_optim)
    else:
      prior_hparams_init = prior_defaults
      prior_hparams_init_vals = jnp.stack(prior_defaults)
    if cond_hparams_names:
      cond_values = get_cond_values(cond_hparams_names=cond_hparams_names,
                        num_samples=config.num_samples_plot,
                        eta_init=eta_i,
                        prior_hparams_init=prior_hparams_init,
                        )
    else:
      cond_values = None
    
    lalme_az_ = sample_lalme_az(
        state_list=state_list,
        batch=batch,
        cond_values=cond_values,
        prior_hparams=jnp.stack(prior_defaults),
        # smi_eta=smi_eta_,
        prng_key=next(prng_seq),
        config=config,
        lalme_dataset=lalme_dataset,
        include_gamma=show_basis_fields,
        num_samples=config.num_samples_plot,
        num_samples_chunk=num_samples_chunk,
    )

    plot.lalme_plots_arviz(
        lalme_az=lalme_az_,
        lalme_dataset=lalme_dataset,
        step=state_list[0].step,
        show_mu=show_mu,
        show_zeta=show_zeta,
        show_basis_fields=show_basis_fields,
        show_W_items=show_W_items,
        show_a_items=show_a_items,
        lp_floating=lp_floating,
        lp_floating_traces=lp_floating_traces,
        lp_floating_grid10=lp_floating_grid10,
        lp_anchor_val_grid30=lp_anchor_val_grid30,
        lp_anchor_val_grid28=lp_anchor_val_grid28,
        lp_anchor_val_grid21=lp_anchor_val_grid21,
        lp_anchor_val_grid10=lp_anchor_val_grid10,
        mcmc_img=(mcmc_img if eta_i==1.000 else None),
        lp_random_anchor=lp_random_anchor,
        lp_random_anchor_grid10=lp_random_anchor_grid10,
        lp_anchor_val=lp_anchor_val,
        lp_anchor_test=lp_anchor_test,
        loc_inducing=loc_inducing,
        workdir_png=workdir_png,
        summary_writer=summary_writer,
        suffix=f"_eta_floating_{float(eta_i):.3f}_sigma_a_{prior_hparams_init_vals[0]:.3f}_sigma_w_{prior_hparams_init_vals[1]:.3f}_sigma_K_{prior_hparams_init_vals[-2]:.3f}_ls_K_{prior_hparams_init_vals[-1]:.3f}",
        scatter_kwargs={"alpha": 0.10},
    )

    if show_location_priorhp_compare:
      print('Plotting comparing results...')
      lalme_az_list = []
      prior_hparams_str_list = []
      for prior_hparams_i in config.prior_hparams_plot:
        print('Samples per prior hparam set')

        # prior_hparams_i_samples =jnp.ones((config.num_samples_plot, 
        #                         len(prior_defaults)))*jnp.array(prior_hparams_i)
        
        if cond_hparams_names:
          cond_values = get_cond_values(cond_hparams_names=cond_hparams_names,
                        num_samples=config.num_samples_plot,
                        eta_init=eta_i,
                        prior_hparams_init=PriorHparams(*prior_hparams_i)
                        )
        else:
          cond_values=None

        # cond_values = jnp.hstack([prior_hparams_i_samples, eta_i_profiles,eta_i_items])

        lalme_az_ = sample_lalme_az(
            state_list=state_list,
            batch=batch,
            cond_values=cond_values,
            prior_hparams=jnp.array(prior_hparams_i), #prior_hparams_i_samples,
            # smi_eta=smi_eta_,
            prng_key=next(prng_seq),
            config=config,
            lalme_dataset=lalme_dataset,
            include_gamma=show_basis_fields,
            num_samples=config.num_samples_plot,
            num_samples_chunk=num_samples_chunk,
        )
        lalme_az_list.append(lalme_az_)
        prior_hparams_str_list.append(fr'$\sigma_a$: {prior_hparams_i[0]}, $\sigma_w$: {prior_hparams_i[1]}, $\sigma_K$: {prior_hparams_i[-2]}, $ls_K$: {prior_hparams_i[-1]}')

      plot.lalme_priorhparam_compare_plots_arviz(
          lalme_az_list=lalme_az_list,
          lalme_dataset=lalme_dataset,
          prior_hparams_str_list=prior_hparams_str_list,
          step=state_list[0].step,
          show_basis_fields=show_basis_fields,
          show_W_items=show_W_items,
          show_a_items=show_a_items,
          lp_floating_grid10=lp_floating_grid10,
          lp_anchor_val_grid10=lp_anchor_val_grid10,
          lp_random_anchor_grid10=lp_random_anchor_grid10,
          lp_anchor_val=lp_anchor_val,
          lp_anchor_test=lp_anchor_test,
          loc_inducing=loc_inducing,
          workdir_png=workdir_png,
          summary_writer=summary_writer,
          suffix=f"_eta_floating_{float(eta_i):.3f}_priorhp_compare",
          scatter_kwargs={"alpha": 0.10},
          )

  ### Evaluation metrics ###
  
  if show_eval_metric:

    error_loc_dict = error_locations_vector_estimate(
        state_list=state_list,
        batch=batch,
        prng_key=next(prng_seq),
        cond_hparams_names=cond_hparams_names,
        config=config,
        eta_eval_grid=eta_eval_grid,
        num_samples=num_samples_chunk,
    )
    images = []

    if 'mean_dist_anchor_val' in error_loc_dict:
      plot_name = 'lalme_vmp_mean_dist_anchor_val'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['mean_dist_anchor_val'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean posterior distance')
      axs.set_title(
          'Error distance for held-out (validation) anchor profiles\n' +
          '(Mean difference posterior vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if 'mean_sq_dist_anchor_val' in error_loc_dict:
      plot_name = 'lalme_vmp_mean_sq_dist_anchor_val'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot square distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['mean_sq_dist_anchor_val'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean posterior distance')
      axs.set_title(
          'Error square distance for held-out (validation) anchor profiles\n' +
          '(Mean distance^2 posterior vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if 'dist_mean_anchor_val' in error_loc_dict:
      plot_name = 'lalme_vmp_dist_mean_anchor_val'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['dist_mean_anchor_val'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Distance to posterior mean')
      axs.set_title(
          'Error distance for held-out (validation) anchor profiles\n' +
          '(Posterior mean vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if 'mean_dist_anchor_test' in error_loc_dict:
      plot_name = 'lalme_vmp_mean_dist_anchor_test'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['mean_dist_anchor_test'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean posterior distance')
      axs.set_title('Error distance for held-out (test) anchor profiles\n' +
                    '(Mean difference posterior vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if 'mean_sq_dist_anchor_test' in error_loc_dict:
      plot_name = 'lalme_vmp_mean_sq_dist_anchor_test'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot square distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['mean_sq_dist_anchor_test'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean posterior distance')
      axs.set_title(
          'Error square distance for held-out (test) anchor profiles\n' +
          '(Mean distance^2 posterior vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if 'dist_mean_anchor_test' in error_loc_dict:
      plot_name = 'lalme_vmp_dist_mean_anchor_test'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['dist_mean_anchor_test'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Distance to posterior mean')
      axs.set_title('Error distance for held-out (test) Anchor profiles\n' +
                    '(Posterior mean vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    plot_name = 'lalme_vmp_mean_dist_floating'
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    # Plot distance as a function of eta
    axs.plot(eta_eval_grid, error_loc_dict['mean_dist_floating'])
    axs.set_xlabel('eta_floating')
    axs.set_ylabel('Mean distance')
    axs.set_title('Error distance for floating profiles\n' +
                  '(posterior vs. fit-technique)')
    fig.tight_layout()
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    plot_name = 'lalme_vmp_dist_mean_floating'
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    # Plot distance as a function of eta
    axs.plot(eta_eval_grid, error_loc_dict['dist_mean_floating'])
    axs.set_xlabel('eta_floating')
    axs.set_ylabel('Distance to posterior mean')
    axs.set_title('Error distance for floating profiles\n' +
                  '(posterior vs. fit-technique)')
    fig.tight_layout()
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    if config.floating_anchor_copies:
      plot_name = 'lalme_vmp_mean_dist_floating_copies_only'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['mean_dist_floating_copies_only'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean distance')
      axs.set_title('Error distance for floating profiles\n' +
                    '(posterior vs. fit-technique)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

      plot_name = 'lalme_vmp_dist_mean_floating_copies_only'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['dist_mean_floating_copies_only'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Distance to posterior mean')
      axs.set_title('Error distance for floating profiles\n' +
                    '(posterior vs. fit-technique)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if config.ad_hoc_val_profiles:
      plot_name = 'lalme_vmp_mean_dist_anchor_val_subset'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['mean_dist_anchor_val_subset'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean distance')
      axs.set_title('Error distance for SUBSET of held-out (val) Anchor profiles\n' +
                    '(Mean difference posterior vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

      plot_name = 'lalme_vmp_dist_mean_anchor_val_subset'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['dist_mean_anchor_val_subset'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Distance to posterior mean')
      axs.set_title('Error distance for SUBSET of held-out (val) Anchor profiles\n' +
                    '(distance to posterior mean)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if config.include_random_anchor:
      plot_name = 'lalme_vmp_mean_dist_random_anchor'
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
      # Plot distance as a function of eta
      axs.plot(eta_eval_grid, error_loc_dict['mean_dist_random_anchor'])
      axs.set_xlabel('eta_floating')
      axs.set_ylabel('Mean posterior distance')
      axs.set_title('Error distance for anchor profiles\n' +
                    '(posterior vs. truth)')
      fig.tight_layout()
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if summary_writer:
      plot_name = 'lalme_vmp_distance'
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=state_list[0].step,
          max_outputs=len(images),
      )



def train_and_evaluate(config: ConfigDict, workdir: str) -> None:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  # Remove trailing slash
  workdir = workdir.rstrip("/")

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)


  # Add some parameters to config
    # WANDB
  if config.use_wandb:
      # if I AM tuning hps
      if config.sweep:
        wandb.init(
          project=config.wandb_project_name,
        )
        # config.kernel_kwargs.amplitude = wandb.config.kernel_amplitude
        # config.kernel_kwargs.length_scale = wandb.config.kernel_length_scale
        config.optim_kwargs.lr_schedule_kwargs.peak_value = wandb.config.peak_value
        config.optim_kwargs.lr_schedule_kwargs.decay_rate = wandb.config.decay_rate

        checkpoint_dir_single_sweep_run = str(pathlib.Path(workdir) / f'checkpoints_{wandb.run.id}')
        if not os.path.exists(checkpoint_dir_single_sweep_run):
          os.makedirs(checkpoint_dir_single_sweep_run, exist_ok=True)

      # if I am NOT tuning hps
      else:
        wandb.init(
          # set the wandb project where this run will be logged
          project=config.wandb_project_name,
          config=config.fixed_configs_wandb,
        )
        # config.kernel_kwargs.amplitude = config.fixed_configs_wandb.kernel_amplitude
        # config.kernel_kwargs.length_scale = config.fixed_configs_wandb.kernel_length_scale
        config.optim_kwargs.lr_schedule_kwargs.peak_value = config.fixed_configs_wandb.peak_value
        config.optim_kwargs.lr_schedule_kwargs.decay_rate = config.fixed_configs_wandb.decay_rate

  new_defaults = {'w_prior_scale': config.prior_hparams.w_prior_scale, 
                  'a_prior_scale': config.prior_hparams.a_prior_scale, 
                 'mu_prior_concentration': config.prior_hparams.mu_prior_concentration, 
                 'mu_prior_rate': config.prior_hparams.mu_prior_rate, 
                 'zeta_prior_a': config.prior_hparams.zeta_prior_a, 
                 'zeta_prior_b': config.prior_hparams.zeta_prior_b, 
                 'kernel_amplitude': config.kernel_kwargs.amplitude, 
                 'kernel_length_scale': config.kernel_kwargs.length_scale}
  
  PriorHparams.set_defaults(**new_defaults)

  if (config.path_MCMC_samples != ''):
      lalme_az_mcmc = az.from_netcdf(config.path_MCMC_samples)
      loc_samples_mcmc = jnp.vstack(jnp.array(lalme_az_mcmc.posterior[f'loc_floating'].reindex(LP_floating=config.lp_floating_grid10)))
      # loc_samples_mcmc = loc_samples_mcmc[1000:] # just last 500 samples
      n_samples_mcmc = loc_samples_mcmc.shape[0]
      if n_samples_mcmc > config.max_wass_samples:
          idxs = jax.random.choice(key=next(prng_seq), a=n_samples_mcmc, shape=(config.max_wass_samples,))
          loc_samples_mcmc = loc_samples_mcmc[idxs]
          n_samples_mcmc = config.max_wass_samples

  # Load and process LALME dataset
  lalme_dataset = load_data(
      prng_key=jax.random.PRNGKey(0),  # use fixed seed for data loading
      config=config,
  )

      
  config.num_profiles = lalme_dataset['num_profiles']
  config.num_profiles_anchor = lalme_dataset['num_profiles_anchor']
  config.num_profiles_floating = lalme_dataset['num_profiles_floating']
  config.num_forms_tuple = lalme_dataset['num_forms_tuple']
  config.num_inducing_points = math.prod(config.flow_kwargs.inducing_grid_shape)

  # For training, we need a Dictionary compatible with jit
  # we remove string vectors
  train_ds = {
      k: v for k, v in lalme_dataset.items() if k not in ['items', 'forms']
  }

  # Compute GP covariance between anchor profiles
  # train_ds['cov_anchor'] = getattr(
  #     kernels, config.kernel_name)(**config.kernel_kwargs).matrix(
  #         x1=train_ds['loc'][:train_ds['num_profiles_anchor'], :],
  #         x2=train_ds['loc'][:train_ds['num_profiles_anchor'], :],
  #     )

  train_ds = get_inducing_points(
      dataset=train_ds,
      inducing_grid_shape=config.flow_kwargs.inducing_grid_shape,
      # kernel_name=config.kernel_name,
      # kernel_kwargs=config.kernel_kwargs,
      # gp_jitter=config.gp_jitter,
  )

  LPs = np.split(
      train_ds['LP'],
      np.cumsum(train_ds['num_profiles_split']),#np.cumsum(batch['num_profiles_split'][1:]),
  )[:-1]
  print(f"TRAIN LPs: {LPs[0] if config.num_lp_anchor_train>0 else 'NONE'} \n VAL LPs: {LPs[1] if config.num_lp_anchor_val>0 else 'NONE'} \n TEST LPs: {LPs[2] if config.num_lp_anchor_test>0 else 'NONE'} \n FLOATING LPs: {LPs[3] }")

  # These parameters affect the dimension of the flow
  # so they are also part of the flow parameters
  config.flow_kwargs.num_profiles_anchor = lalme_dataset['num_profiles_anchor']
  config.flow_kwargs.num_profiles_floating = lalme_dataset[
      'num_profiles_floating']
  config.flow_kwargs.num_forms_tuple = lalme_dataset['num_forms_tuple']
  config.flow_kwargs.num_inducing_points = int(
      math.prod(config.flow_kwargs.inducing_grid_shape))
  config.flow_kwargs.is_smi = True

  # Get locations bounds
  # These define the range of values produced by the posterior of locations
  loc_bounds = np.stack(
      [lalme_dataset['loc'].min(axis=0), lalme_dataset['loc'].max(axis=0)],
      axis=1).astype(np.float32)
  config.flow_kwargs.loc_x_range = tuple(loc_bounds[0])
  config.flow_kwargs.loc_y_range = tuple(loc_bounds[1])

  ### Initialize States ###
  # Here we use three different states defining three separate flow models:
  #   -Global parameters
  #   -Posterior locations for floating profiles
  #   -Posterior locations for anchor profiles (treated as floating)
  # cond_hparams_names = ['w_prior_scale', 'a_prior_scale', 'kernel_amplitude', 'kernel_length_scale', 'eta']
  eta_init = 1.
  prior_hparams_init = PriorHparams()


  if config.cond_hparams_names:
    cond_values_init = get_cond_values(cond_hparams_names=config.cond_hparams_names,
                        num_samples=config.num_samples_elbo,
                        eta_init=eta_init,
                        prior_hparams_init=prior_hparams_init
                        )
  else:
    cond_values_init = None

#   prior_defaults = jnp.stack(PriorHparams())
#   prior_hparams=jnp.ones((config.num_samples_elbo, 
#                                   len(prior_defaults)))*prior_defaults # init params right?


#   eta_profiles = jnp.ones((config.num_samples_elbo, train_ds['num_profiles']))
#   eta_items = jnp.ones((config.num_samples_elbo, len(train_ds['num_forms_tuple'])))
# #   smi_eta_init = {
# #       'profiles':eta_profiles,
# #       'items':eta_items,
# #   }
#   cond_values_init = jnp.hstack([prior_hparams, eta_profiles, eta_items])


  # Global parameters
  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints') #_ng26re76')
  logging.info(f'Checkpoint directory: {checkpoint_dir}')


  state_name_list = [
      'global', 'loc_floating', 'loc_floating_aux', 'loc_random_anchor'
  ]
  state_list = []

  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[0]}',
          forward_fn=hk.transform(q_distr_global),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'cond_values':cond_values_init,
            #   'eta': smi_eta_init['profiles'],
              'num_samples':config.num_samples_elbo,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  if config.sweep:
    checkpoint_dir_sweeprun_sub = f'{checkpoint_dir_single_sweep_run}/{state_name_list[0]}'
    if not os.path.exists(checkpoint_dir_sweeprun_sub):
          os.makedirs(checkpoint_dir_sweeprun_sub, exist_ok=True)

  # Get an initial sample of global parameters
  # (used below to initialize floating locations)
  global_sample_base_ = hk.transform(q_distr_global).apply(
      state_list[0].params,
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      cond_values=cond_values_init,
    #   eta=smi_eta_init['profiles'],
    num_samples=config.num_samples_elbo,
  )['sample_base']

  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[1]}',
          forward_fn=hk.transform(q_distr_loc_floating),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'global_params_base_sample': global_sample_base_,
              'cond_values':cond_values_init,
            #   'eta': smi_eta_init['profiles'],
              'name': 'loc_floating',
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  if config.sweep:
    checkpoint_dir_sweeprun_sub = f'{checkpoint_dir_single_sweep_run}/{state_name_list[1]}'
    if not os.path.exists(checkpoint_dir_sweeprun_sub):
          os.makedirs(checkpoint_dir_sweeprun_sub, exist_ok=True)

  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[2]}',
          forward_fn=hk.transform(q_distr_loc_floating),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'global_params_base_sample': global_sample_base_,
              'cond_values':cond_values_init,
            #   'eta': smi_eta_init['profiles'],
              'name': 'loc_floating_aux',
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  if config.sweep:
    checkpoint_dir_sweeprun_sub = f'{checkpoint_dir_single_sweep_run}/{state_name_list[2]}'
    if not os.path.exists(checkpoint_dir_sweeprun_sub):
          os.makedirs(checkpoint_dir_sweeprun_sub, exist_ok=True)

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0 and state_list[0].step < config.training_steps:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))

    # Syne-tune for HPO
    synetune_report = syne_tune.Reporter()
  else:
    summary_writer = None
    synetune_report = None
  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(flatten_dict(config))

  if not config.sweep:
    # Print a useful summary of the execution of the flows.
    logging.info('FLOW GLOBAL PARAMETERS:')
    tabulate_fn_ = hk.experimental.tabulate(
        f=lambda state, prng_key: hk.transform(q_distr_global).apply(
            state.params,
            prng_key,
            flow_name=config.flow_name,
            flow_kwargs=config.flow_kwargs,
            cond_values=cond_values_init,
          #   eta=smi_eta_init['profiles'],
          num_samples=config.num_samples_elbo,
        ),
        columns=(
            "module",
            "owned_params",
            "params_size",
            "params_bytes",
        ),
        filters=("has_params",),
    )
    summary = tabulate_fn_(state_list[0], next(prng_seq))
    for line in summary.split("\n"):
      logging.info(line)

    logging.info('FLOW LOCATION FLOATING PROFILES:')
    tabulate_fn_ = hk.experimental.tabulate(
        f=lambda state, prng_key: hk.transform(q_distr_loc_floating).apply(
            state.params,
            prng_key,
            flow_name=config.flow_name,
            flow_kwargs=config.flow_kwargs,
            global_params_base_sample=global_sample_base_,
            cond_values=cond_values_init,
          #   eta=smi_eta_init['profiles'],
            name='loc_floating',
        ),
        columns=(
            "module",
            "owned_params",
            "params_size",
            "params_bytes",
        ),
        filters=("has_params",),
    )
    summary = tabulate_fn_(state_list[1], next(prng_seq))
    for line in summary.split("\n"):
      logging.info(line)

  if config.include_random_anchor:
    state_list.append(
        initial_state_ckpt(
            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[3]}',
            forward_fn=hk.transform(q_distr_loc_random_anchor),
            forward_fn_kwargs={
                'flow_name': config.flow_name,
                'flow_kwargs': config.flow_kwargs,
                'global_params_base_sample': global_sample_base_,
                'cond_values':cond_values_init,
                # 'eta': smi_eta_init['profiles'],
            },
            prng_key=next(prng_seq),
            optimizer=make_optimizer(**config.optim_kwargs),
        ))
    if not config.sweep:
      logging.info('FLOW RANDOM LOCATION OF ANCHOR PROFILES:')
      tabulate_fn_ = hk.experimental.tabulate(
          f=lambda state, prng_key: hk.transform(q_distr_loc_random_anchor).apply(
              state.params,
              prng_key,
              flow_name=config.flow_name,
              flow_kwargs=config.flow_kwargs,
              global_params_base_sample=global_sample_base_,
              cond_values=cond_values_init,
              # eta=smi_eta_init['profiles'],
              num_samples=config.num_samples_elbo,
          ),
          columns=(
              "module",
              "owned_params",
              "params_size",
              "params_bytes",
          ),
          filters=("has_params",),
      )
      summary = tabulate_fn_(state_list[3], next(prng_seq))
      for line in summary.split("\n"):
        logging.info(line)

  profile_is_anchor = jnp.arange(
      train_ds['num_profiles']) < train_ds['num_profiles_anchor']

  @jax.jit
  def update_states_jit(state_list, batch, prng_key):
    return update_states(
        state_list=state_list,
        batch=batch,
        prng_key=prng_key,
        optimizer=make_optimizer(**config.optim_kwargs),
        loss_fn=loss,
        loss_fn_kwargs={
            'num_samples': config.num_samples_elbo,
            'flow_name': config.flow_name,
            'flow_kwargs': config.flow_kwargs,
            'eta_sampling_a': config.eta_sampling_a,
            'eta_sampling_b': config.eta_sampling_b,
            'cond_hparams':config.cond_hparams_names,
            'include_random_anchor': config.include_random_anchor,
            # 'prior_hparams': config.prior_hparams,
            'profile_is_anchor': profile_is_anchor,
            'kernel_name': config.kernel_name,
            # 'kernel_kwargs': config.kernel_kwargs,
            'sample_priorhparams_fn': sample_priorhparams_values,
            'sample_priorhparams_kwargs': config.prior_hparams_hparams,
            'num_samples_gamma_profiles': config.num_samples_gamma_profiles,
            'gp_jitter': config.gp_jitter,
            'num_profiles_anchor':config.num_profiles_anchor,
            'num_inducing_points':config.num_inducing_points,
        },
    )

  # globals().update(loss_fn_kwargs)

  # @jax.jit
  # def elbo_validation_jit(state_list, batch, prng_key):
  #   return elbo_estimate_along_eta(
  #       params_tuple=[state.params for state in state_list],
  #       batch=batch,
  #       prng_key=prng_key,
  #       num_samples=config.num_samples_eval,
  #       flow_name=config.flow_name,
  #       flow_kwargs=config.flow_kwargs,
  #       eta_sampling_a=1.0,
  #       eta_sampling_b=1.0,
  #       include_random_anchor=config.include_random_anchor,
  #       prior_hparams=config.prior_hparams,
  #       profile_is_anchor=profile_is_anchor,
  #       kernel_name=config.kernel_name,
  #       kernel_kwargs=config.kernel_kwargs,
  #       num_samples_gamma_profiles=config.num_samples_gamma_profiles,
  #       gp_jitter=config.gp_jitter,
  #   )
    
  # error_locations_estimate_jit = lambda state_list, batch, prng_key: error_locations_estimate(
  #     params_tuple=[state.params for state in state_list],
  #     batch=batch,
  #     prng_key=prng_key,
  #     num_samples=config.num_samples_eval,
  #     flow_name=config.flow_name,
  #     flow_kwargs=config.flow_kwargs,
  #     include_random_anchor=config.include_random_anchor,
  #     kernel_name=config.kernel_name,
  #     kernel_kwargs=config.kernel_kwargs,
  #     num_samples_gamma_profiles=config.num_samples_gamma_profiles,
  #     gp_jitter=config.gp_jitter,
  # )
  # # TODO: This doesn't work after jitting.
  # # error_locations_estimate_jit = jax.jit(error_locations_estimate_jit)

  save_last_checkpoint = False
  if ((state_list[0].step < config.training_steps) and (config.save_last_checkpoint == True)):
    logging.info('Training variational posterior...')
    # Reset random keys
    prng_seq = hk.PRNGSequence(config.seed)
    save_last_checkpoint = True

  loss_ = []
  mean_dist_anchor_val_min_list = []
  if (config.path_MCMC_samples != ''):
    wd_means = []
    wd_joints = []
  while state_list[0].step < config.training_steps:
    # step = 0

    # Plots to monitor training
    if config.log_img_steps > 0:
      if (state_list[0].step > 0) and (state_list[0].step %
                                       config.log_img_steps) == 0:
        logging.info("Logging plots...")
        log_images(
            state_list=state_list,
            prng_key=next(prng_seq),
            config=config,
            lalme_dataset=lalme_dataset,
            batch=train_ds,
            cond_hparams_names=config.cond_hparams_names,
            show_mu=True,
            show_zeta=True,
            lp_floating_grid10=config.lp_floating_grid10,
            lp_random_anchor_grid10=config.lp_random_anchor_10,
            show_eval_metric=True,
            eta_eval_grid=jnp.linspace(0, 1, 21),
            num_samples_chunk=config.num_samples_chunk_plot,
            summary_writer=summary_writer,
            workdir_png=workdir,
        )
        logging.info("...done.")
    # Log learning rate
    summary_writer.scalar(
        tag='learning_rate',
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(state_list[0].step),
        step=state_list[0].step,
    )

    # Training step
    state_list, metrics = update_states_jit(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
    )

    loss_.append(float(metrics['train_loss']))
    if len(loss_) >= config.max_steps_nan:
      loss_ = loss_[-config.max_steps_nan:]
      if jnp.isnan(jnp.array(loss_).astype(float)).all():
        logging.warning('Training stopped, %d steps with NaN loss',
                        config.max_steps_nan)
        break

    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state_list[0].step - 1,
    )
    if config.use_wandb:
      wandb.log({'learning_rate': getattr(optax, config.optim_kwargs.lr_schedule_name)(
              **config.optim_kwargs.lr_schedule_kwargs)(state_list[0].step),
              'train_loss': metrics['train_loss'],
            },
        step=state_list[0].step)

    if state_list[0].step == 1:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    if config.checkpoint_steps > 0:
      if state_list[0].step % config.checkpoint_steps == 0:
        for state, state_name in zip(state_list, state_name_list):
          save_checkpoint(
              state=state,
              checkpoint_dir=f'{checkpoint_dir}/{state_name}',
              keep=config.checkpoints_keep,
          )
          
    # Metrics for evaluation
    if ((state_list[0].step == 1) or (state_list[0].step == 10000) or (state_list[0].step % config.eval_steps == 0)):
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

      logging.info(f"Process uses {psutil.Process().memory_info().rss / (1024 * 1024)} MB memory.")

      # # Multi-stage ELBO
      # elbo_dict_eval = elbo_validation_jit(
      #     state_list=state_list,
      #     batch=train_ds,
      #     prng_key=next(prng_seq),
      # )
      # for k, v in elbo_dict_eval.items():
      #   summary_writer.scalar(
      #       tag=f'elbo_{k}',
      #       value=v.mean(),
      #       step=state_list[0].step,
      #   )

      # WD vs MCMC samples
      if (config.path_MCMC_samples != ''):
          prior_defaults = PriorHparams()
          if config.cond_hparams_names:
            cond_values = get_cond_values(cond_hparams_names=config.cond_hparams_names,
                            num_samples=config.num_samples_plot,
                            eta_init=config.wandb_evaleta,
                            prior_hparams_init=prior_defaults
                            )
          else:
            cond_values = None

          logging.info(prior_defaults)


          _, az_locations = sample_locations_floating(
              lalme_dataset=lalme_dataset,
              state_list=state_list,
              cond_values=cond_values,
            #   smi_eta=smi_eta_,
              prng_key=next(prng_seq),
              config=config,
              num_samples=config.num_samples_plot,
              num_samples_chunk=config.num_samples_chunk_plot,
          )

          # Find the indices that would sort a1 according to the order in a3

          loc_samples_VI = jnp.array(az_locations.posterior[f'loc_floating'].reindex(LP_floating=config.lp_floating_grid10).squeeze())
          n_samples_VI = loc_samples_VI.shape[0]
          if n_samples_VI > config.max_wass_samples:
              idxs = jax.random.choice(key=next(prng_seq), a=n_samples_VI, shape=(config.max_wass_samples,))
              loc_samples_VI = loc_samples_VI[idxs]
              n_samples_VI = config.max_wass_samples
          a_mcmc = jnp.ones((n_samples_mcmc,)) / n_samples_mcmc
          b_VI = jnp.ones((n_samples_VI,)) / n_samples_VI
          
          # Need to swap axis from (n_samples, n_locations, ...) to (n_locations, n_samples, ...)
          WD = jnp.array([ot.emd2(a_mcmc, b_VI, ot.dist(x, y, metric='euclidean')) 
                                          for (x,y) in zip(loc_samples_mcmc.swapaxes(0,1), loc_samples_VI.swapaxes(0,1))])
          
          
          plot.lalme_plots_arviz(
              lalme_az=az_locations,
              lalme_dataset=lalme_dataset,
              step=state_list[0].step,
              lp_floating_grid10=config.lp_floating_grid10,
              mcmc_img=(config.path_mcmc_img if config.dataset_id=='coarsen_8_items' else None),
              workdir_png=workdir,
              summary_writer=summary_writer,
              use_wandb=config.use_wandb,
              suffix=f"eta_{config.wandb_evaleta}",
              scatter_kwargs={"alpha": 0.10},
          )
          WD_mean = WD.mean()
          wd_means.append(WD_mean)

          summary_writer.scalar(
              tag='WD_vs_MCMC',
              value=float(WD_mean),
              step=state_list[0].step,
          )
          
          summary_writer.scalar(
              tag='min_WD_vs_MCMC',
              value=float(min(wd_means)),
              step=state_list[0].step,
          )
          if config.use_wandb:

            wandb.log({'WD_vs_MCMC': float(WD_mean)},
                  step=state_list[0].step)
            wandb.log({'min_WD_vs_MCMC': float(min(wd_means))},
                    step=state_list[0].step)
            
          mcmc_samples_flat = loc_samples_mcmc.reshape(-1, 20)
          VI_samples_flat = loc_samples_VI.reshape(-1, 20)

          # Compute the cost matrix between the samples
          M = ot.dist(mcmc_samples_flat, VI_samples_flat)

          # Compute the WD between the two distributions
          a_mcmc_flat = jnp.ones((mcmc_samples_flat.shape[0],)) / mcmc_samples_flat.shape[0]
          b_VI_flat = jnp.ones((VI_samples_flat.shape[0],)) / VI_samples_flat.shape[0]
          wd_joint = ot.emd2(a_mcmc_flat,b_VI_flat, M, numItermax=1000000)
          wd_joints.append(wd_joint)

          summary_writer.scalar(
              tag='WD_joint_vs_MCMC',
              value=float(wd_joint),
              step=state_list[0].step,
          )
          summary_writer.scalar(
              tag='min_WD_joint_vs_MCMC',
              value=float(min(wd_joints)),
              step=state_list[0].step,
          )
          if config.use_wandb:
            wandb.log({'WD_joint_vs_MCMC': float(wd_joint)},
                    step=state_list[0].step)
            wandb.log({'min_WD_joint_vs_MCMC': float(min(wd_joints))},
                    step=state_list[0].step)

      # Estimate posterior distance to true locations
      eta_eval_grid_ = jnp.linspace(0, 1, 21)
      # Each element is a vector across eta
      error_loc_all_eta_dict = error_locations_vector_estimate(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
          cond_hparams_names=config.cond_hparams_names,
          config=config,
          eta_eval_grid=eta_eval_grid_,
          num_samples=config.num_samples_eval,
      )

      # Summarize the distances
      error_loc_dict = {}
      for k, v in error_loc_all_eta_dict.items():
        error_loc_dict[k + '_min'] = jnp.min(v)
        error_loc_dict[k + '_min_eta'] = eta_eval_grid_[jnp.argmin(v)]
        error_loc_dict[k + '_max'] = jnp.max(v)
        error_loc_dict[k + '_max_eta'] = eta_eval_grid_[jnp.argmax(v)]
        if config.wandb_evaleta:
          idx = jnp.where(eta_eval_grid_ == config.wandb_evaleta)[0][0]
          error_loc_dict[k + f'_eta{config.wandb_evaleta}'] = float(v[idx])


      for k, v in error_loc_dict.items():
        summary_writer.scalar(
            tag=k,
            value=float(v),
            step=state_list[0].step,
        )
        if config.use_wandb:
          wandb.log({f'{k}': float(v)},
                  step=state_list[0].step)
        # Report the metric used by syne-tune
        if k == config.synetune_metric:
          synetune_report(**{k: float(v)})
          # synetune_report(**{k + '_max': float(jnp.max(v))})
      # if config.flow_kwargs.num_profiles_anchor > 0:
      #   mean_dist_anchor_val_min_list.append(error_loc_dict["mean_dist_anchor_val_min"])
      #   summary_writer.scalar(
      #       tag='minimum_mean_dist_anchor_val_min',
      #       value=float(min(mean_dist_anchor_val_min_list)),
      #       step=state_list[0].step,
      #   )
      #   if config.use_wandb:
      #     wandb.log({'minimum_mean_dist_anchor_val_min': float(min(mean_dist_anchor_val_min_list))},
      #             step=state_list[0].step)


    # Wait until computations are done before the next step
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    logging.info("STEP: %5d terminated; training loss: %.3f", state_list[0].step,
                   metrics["train_loss"])
    # logging.info('Final training step: %i', state_list[0].step)
  
  # Saving checkpoint at the end of the training process
  if save_last_checkpoint:
    for state, state_name in zip(state_list, state_name_list):
      save_checkpoint(
          state=state,
          checkpoint_dir=f'{checkpoint_dir_single_sweep_run}/{state_name}',
          keep=config.checkpoints_keep,
      )
    del state

  # Estimate posterior distance to true locations
  if config.eval_last and summary_writer is not None:
    logging.info("Logging distance error on last state...")
    eta_eval_grid_ = jnp.linspace(0, 1, 21)
    # Each element is a vector across eta
    error_loc_all_eta_dict = error_locations_vector_estimate(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
        cond_hparams_names=config.cond_hparams_names,
        config=config,
        eta_eval_grid=eta_eval_grid_,
        num_samples=config.num_samples_eval,
    )

    # Summarize the distances
    error_loc_dict = {}
    for k, v in error_loc_all_eta_dict.items():
      error_loc_dict[k + '_min'] = jnp.min(v)
      error_loc_dict[k + '_min_eta'] = eta_eval_grid_[jnp.argmin(v)]
      error_loc_dict[k + '_max'] = jnp.max(v)
      error_loc_dict[k + '_max_eta'] = eta_eval_grid_[jnp.argmax(v)]

    for k, v in error_loc_dict.items():
      summary_writer.scalar(
          tag=k,
          value=float(v),
          step=state_list[0].step,
      )
      # Report the metric used by syne-tune
      if k == config.synetune_metric and synetune_report is not None:
        synetune_report(**{k: float(v)})
    logging.info("...done!")

  # Save samples from the last state
  if config.save_samples:
    logging.info("Saving samples of VMP...")
    for eta_i in config.eta_plot:
      # prior_defaults = jnp.stack(PriorHparams())
      # prior_hparams=jnp.ones((config.num_samples_plot, 
      #                             len(prior_defaults)))*prior_defaults # init params right?
      
      # eta_i_profiles = eta_i * jnp.ones(
      #     (config.num_samples_plot, config.num_profiles))
      # eta_i_profiles = jax.vmap(lambda eta_: jnp.where(profile_is_anchor,
      #             1.,eta_,))(eta_i_profiles)
      # eta_i_items = jnp.ones((config.num_samples_plot, len(config.num_forms_tuple)))
      # smi_eta_ = {
      #     'profiles':eta_i_profiles,
      #     'items':eta_i_items,
      # }
      # cond_values = jnp.hstack([prior_hparams, eta_i_profiles, eta_i_items])
      prior_defaults = PriorHparams()
      if config.cond_hparams_names:
        cond_values = get_cond_values(cond_hparams_names=config.cond_hparams_names,
                        num_samples=config.num_samples_save,
                        eta_init=eta_i,
                        prior_hparams_init=prior_defaults
                        )
      else:
        cond_values = None

      lalme_az_ = sample_lalme_az(
          state_list=state_list,
          batch=train_ds,
          cond_values=cond_values,
          prior_hparams=jnp.stack(PriorHparams()),#prior_hparams,
        #   smi_eta=smi_eta_,
          prng_key=next(prng_seq),
          config=config,
          lalme_dataset=lalme_dataset,
          include_gamma=False,
          num_samples=config.num_samples_save,
          num_samples_chunk=config.num_samples_chunk_plot,
      )
      lalme_az_.to_netcdf(workdir + f'/lalme_az_eta_{float(eta_i):.3f}.nc')
    logging.info("...done!")

  # Last plot of posteriors
  if config.log_img_at_end:
    logging.info("Plotting results...")
    log_images(
        state_list=state_list,
        prng_key=next(prng_seq),
        config=config,
        lalme_dataset=lalme_dataset,
        batch=train_ds,
        cond_hparams_names=config.cond_hparams_names,
        # show_mu=True,
        # show_zeta=True, 
        show_basis_fields=False,
        # show_W_items=lalme_dataset['items'],
        # show_a_items=lalme_dataset['items'],
        # lp_floating=lalme_dataset['LP'][lalme_dataset['num_profiles_anchor']:],
        # lp_floating_traces=config.lp_floating_grid10,
        lp_floating_grid10=config.lp_floating_grid10,
        # lp_anchor_val_grid30=config.lp_anchor_val_grid30,
        # lp_anchor_val_grid21=config.lp_anchor_val_grid21,
        # lp_anchor_val_grid28=config.lp_anchor_val_grid28,
        # lp_anchor_val_grid10=config.lp_anchor_val_grid10,
        # lp_random_anchor=(
        #     lalme_dataset['LP'][:lalme_dataset['num_profiles_anchor']]
        #     if config.include_random_anchor else None),
        # lp_random_anchor_grid10=config.lp_random_anchor_10,
        # show_lp_anchor_val=(
        #     True if lalme_dataset['num_profiles_split'].lp_anchor_val > 0 else
        #     False),
        # show_lp_anchor_test=(
        #     True if lalme_dataset['num_profiles_split'].lp_anchor_test > 0 else
        #     False),
        mcmc_img=(config.path_mcmc_img if config.dataset_id=='coarsen_8_items' else None),
        loc_inducing=train_ds['loc_inducing'],
        show_location_priorhp_compare=False,
        show_eval_metric=False, 
        eta_eval_grid=jnp.linspace(0, 1, 21),
        num_samples_chunk=config.num_samples_chunk_plot,
        summary_writer=summary_writer,
        workdir_png=workdir,
    )
    jax.clear_caches()
    logging.info("...done!")
  
  wandb.finish()


#####################################################################################
  # Find best eta ###

  #################################
  # TODO: repeated block, can probably be deleted
  if config.tune_vmp_hparams:
    logging.info('Finding best hyperparameters...')

  def tune_vmp_hparams(cond_hparams_names,
                       eta_i:float = None):
      num_profiles_split = train_ds['num_profiles_split']
      LPs = train_ds['LP']
      LPs_split = np.split(
      LPs,
      np.cumsum(num_profiles_split),
      )[:-1]

      floating_anchor_copies = config.floating_anchor_copies
      if floating_anchor_copies:
        train_idxs = jnp.where(jnp.isin(LPs_split[3], LPs_split[0]*1000))[0]
      else:
        train_idxs = None

      ad_hoc_val_profiles = config.ad_hoc_val_profiles
      if ad_hoc_val_profiles:
        val_idxs = jnp.where(jnp.isin(LPs_split[1], jnp.array(config.ad_hoc_val_list)))[0]
      else:
        val_idxs = None


      init_eta_vals = {'default': 1.0,
                'mixed':0.75,
                'low':0.,
                'high':0.5,}

      prior_hparams_default = jnp.stack(PriorHparams()) # check
      hp_star_default = jnp.hstack([prior_hparams_default, init_eta_vals['default']]) 

      init_vals = {'default':hp_star_default,
                  'mixed':jnp.hstack([jnp.array([4., 1., 1., 0.5, 1., 1., 0.2, 0.1]), init_eta_vals['mixed']]),
                  'low':jnp.hstack([jnp.array([1., 4., 1., 0.5, 1., 1., 0.1, 0.2]), init_eta_vals['low']]),
                  'high':jnp.hstack([jnp.array([8., 14., 1., 0.5, 1., 1., 0.4, 0.5]), init_eta_vals['high']]),}

      # optim_mask = jnp.array([1, 1, 0, 0, 0, 0, 1, 1, 1])
      optim_mask = jnp.array([1 if i in cond_hparams_names else 0 for i in PriorHparams()._fields ] + ([1] if 'eta' in cond_hparams_names else [0]))
      print('optim mask:', optim_mask)
      optim_mask_indices = (tuple(i for i, x in enumerate(optim_mask) if x == 0),tuple(i for i, x in enumerate(optim_mask) if x == 1))

      if not os.path.exists(workdir + f'/hparam_tuning'):
        os.makedirs(workdir + f'/hparam_tuning', exist_ok=True)


      # optim_kwargs_old = config.optim_kwargs.to_dict().copy()
      # optim_kwargs_old['lr_schedule_kwargs']['decay_rate'] = 0.6
      # optim_kwargs_old['lr_schedule_kwargs']['peak_value'] = 3e-4
      # 'elbo_opt_old':make_optimizer(**optim_kwargs_old), 
      all_optimisers = {'elbo_opt':make_optimizer(**config.optim_kwargs), #note you are using optim kwargs of VMP training
                      'plain_lr1':make_optimizer_hparams(**config.optim_kwargs_hp),  
                      'plain_lr2':make_optimizer_hparams(**config.optim_kwargs_hp_alternative)}

        
      for optimiser_name, my_optimiser in all_optimisers.items():  
        for init_type, init_val in init_vals.items():
          
        
          hp_star_init = init_val[optim_mask==1]
          hp_fixed = init_val[optim_mask==0]
          print(f"optimiser: {optimiser_name}, init type: {init_type}")

          if jax.process_index() == 0:
              summary_writer_hp = tensorboard.SummaryWriter(workdir + f'/hparam_tuning{"_FIXED_ETA" if eta_i is not None else ""}/{init_type}_{optimiser_name}')
              summary_writer_hp.hparams({init_type:hp_star_init, optimiser_name:my_optimiser})

          error_locations_estimate_jit = lambda locations_sample, loc: error_locations_estimate(
            locations_sample=locations_sample,
            num_profiles_split=num_profiles_split,
            loc=loc,
            floating_anchor_copies=floating_anchor_copies,
            train_idxs=train_idxs,
            ad_hoc_val_profiles=ad_hoc_val_profiles, 
            val_idxs=val_idxs,
          )
          error_locations_estimate_jit = jax.jit(error_locations_estimate_jit)

          def mse_fixedhp(
            hp_params:Array,
            hp_optim_mask_indices:Tuple,
            hp_fixed_values:Array,
            state_list: List[TrainState],
            batch: Optional[Batch],
            prng_key: PRNGKey,
            flow_name: str,
            flow_kwargs: Dict[str, Any],
            include_random_anchor:bool,
            num_samples: int,
            profile_is_anchor:Array,
            num_profiles_split:int,
            config:ConfigDict,
            eta_fixed:float = None,
        ) -> Dict[str, Array]:

            # # Sample eta values
            # hp_fixed_values = jnp.array(hp_fixed_values)
            # hp_params_all = jnp.zeros(len(hp_optim_mask_indices[0])+ len(hp_optim_mask_indices[1]))#jnp.zeros(optim_mask.shape)
            # hp_params_all = hp_params_all.at[(hp_optim_mask_indices[1],)].set(hp_params)
            # hp_params_all = hp_params_all.at[(hp_optim_mask_indices[0],)].set(hp_fixed_values)

            
            # logprobs_rho_dict = logprob_rho(hparams=hp_params_all,
            #                   w_sampling_scale_alpha=config.prior_hparams_hparams.w_sampling_scale_alpha,
            #                   w_sampling_scale_beta=config.prior_hparams_hparams.w_sampling_scale_beta,
            #                   a_sampling_scale_alpha=config.prior_hparams_hparams.a_sampling_scale_alpha,
            #                   a_sampling_scale_beta=config.prior_hparams_hparams.a_sampling_scale_beta,
            #                   kernel_sampling_amplitude_alpha=config.prior_hparams_hparams.kernel_sampling_amplitude_alpha,
            #                   kernel_sampling_amplitude_beta=config.prior_hparams_hparams.kernel_sampling_amplitude_beta,
            #                   kernel_sampling_lengthscale_alpha=config.prior_hparams_hparams.kernel_sampling_lengthscale_alpha,
            #                   kernel_sampling_lengthscale_beta=config.prior_hparams_hparams.kernel_sampling_lengthscale_beta,
            #                   eta_sampling_a=config.eta_sampling_a,
            #                   eta_sampling_b=config.eta_sampling_b)
            # logprobs_rho = jnp.array(list(logprobs_rho_dict.values()))
            # logprobs_rho = logprobs_rho.at[(hp_optim_mask_indices[0],)].set(0.)

            # eta_profiles = jnp.where(
            #             profile_is_anchor,1.,hp_params_all[-1])#(367)
            # eta_items = jnp.ones(len(batch['num_forms_tuple'])) #(71)

            # cond_values = jnp.hstack([hp_params_all[:-1],
            #                           eta_profiles, eta_items,
            #                           ]) #(n_samples, n_hps+367+71)
            if eta_fixed is not None:
              cond_values = jnp.hstack([hp_params, eta_fixed])
            else:
              cond_values = hp_params

            q_distr_out = sample_all_flows(
                params_tuple=[state.params for state in state_list],
                prng_key=prng_key,
                flow_name=flow_name,
                flow_kwargs=flow_kwargs,
                cond_values=jnp.broadcast_to(cond_values, (num_samples, len(cond_values))),
                # smi_eta=smi_eta_,
                include_random_anchor=include_random_anchor,
                num_samples=num_samples,
            )

            error_loc_dict = error_locations_estimate_jit(
                    locations_sample=q_distr_out['locations_sample'],
                    loc=batch['loc'],
                )
            return error_loc_dict['mean_dist_anchor_val'] #- logprobs_rho.sum()
          
          # Jit optimization of hparams 
            
          mse_jit = lambda hp_params, batch, prng_key, state_list_vmp,  hp_optim_mask_indices, hp_fixed_values: mse_fixedhp(
              hp_params=hp_params,
              eta_fixed=eta_i,
              hp_optim_mask_indices=hp_optim_mask_indices,
              hp_fixed_values=hp_fixed_values,
              state_list=state_list_vmp,
              batch=batch,
              prng_key=prng_key,
              num_samples=config.num_samples_hparams_optim,
              flow_name=config.flow_name,
              flow_kwargs=config.flow_kwargs,
              include_random_anchor=config.include_random_anchor,
              profile_is_anchor=profile_is_anchor,
              num_profiles_split=num_profiles_split,
              config=config,
          )

          mse_jit = jax.jit(mse_jit, static_argnames=('hp_optim_mask_indices'))

          update_hp_star_state = lambda hp_star_state, batch, prng_key: update_state(
                state=hp_star_state,
                batch=batch,
                prng_key=prng_key,
                optimizer=my_optimiser, #make_optimizer(**config.optim_kwargs), 
                loss_fn=mse_jit,
                loss_fn_kwargs={
                    'state_list_vmp': state_list,
                    'hp_optim_mask_indices':optim_mask_indices,
                    'hp_fixed_values':hp_fixed,
                },
            )
          update_hp_star_state = jax.jit(update_hp_star_state)

        
          # Reset random key sequence
          prng_seq = hk.PRNGSequence(config.seed)


          info_dict = {'init':hp_star_init, 'init_type':init_type, 'hp_names':cond_hparams_names,
          'loss':[], 'params':[], 'step':[]}

          # key_search = next(prng_seq)

          # # SGD over elpd for all hparams 
          hp_star_state = TrainState(
              params=hp_star_init,
              opt_state=my_optimiser.init(hp_star_init),
              step=0,
          )
          for _ in range(config.hp_star_steps):
            hp_star_state, mse = update_hp_star_state(
                hp_star_state,
                batch=train_ds,
                prng_key=next(prng_seq),
            )
            
            if ((hp_star_state.step == 0) or (hp_star_state.step == 1) or (hp_star_state.step % 100 == 0)):
              labs = "STEP: %5d; training loss: %.6f " + ' '.join([hp + ':%.3f' for hp in cond_hparams_names])
              logging.info(labs,
                float(hp_star_state.step),
              float(mse["train_loss"]), *[float(hp_star_state.params[i]) for i in range(len(cond_hparams_names))])

            # Clip eta_star to [0,1] hypercube and hp_star to [0.000001,..]
            hp_star_state = TrainState(
                params=jnp.hstack([jnp.clip(hp_star_state.params[cond_hparams_names.index('w_prior_scale')],0., 10.) if 'w_prior_scale' in cond_hparams_names else [],# w_prior_scale
                                  jnp.clip(hp_star_state.params[cond_hparams_names.index('a_prior_scale')],3., 19.) if 'a_prior_scale' in cond_hparams_names else [],# a_prior_scale
                                  jnp.clip(hp_star_state.params[cond_hparams_names.index('kernel_amplitude')],0.1, 0.4) if 'kernel_amplitude' in cond_hparams_names else [],# kernel_amplitude
                                  jnp.clip(hp_star_state.params[cond_hparams_names.index('kernel_length_scale')],0.2, 0.5) if 'kernel_length_scale' in cond_hparams_names else [],# kernel_length_scale
                                  jnp.clip(hp_star_state.params[cond_hparams_names.index('eta')], 0., 1.) if 'eta' in cond_hparams_names else []]),
                opt_state=hp_star_state.opt_state,
                step=hp_star_state.step,
            )

            info_dict['loss'].append(mse["train_loss"])
            info_dict['params'].append(hp_star_state.params)
            info_dict['step'].append(hp_star_state.step)

            summary_writer_hp.scalar(
                tag=f'hp_loss_mse{f"_eta{eta_i}" if eta_i is not None else ""}',
                value=mse['train_loss'],
                step=hp_star_state.step - 1,
            )
            for hp_star_name, hp_star_i in zip(cond_hparams_names, hp_star_state.params):
              summary_writer_hp.scalar(
                  tag=f'hp_opt_{hp_star_name}{f"_eta{eta_i}" if eta_i is not None else ""}',
                  value=hp_star_i,
                  step=hp_star_state.step - 1,
              )

          with open(workdir + f"/hp_info_{'eta' if 'eta' in cond_hparams_names else 'only'}priorhps_{init_type}_{optimiser_name}_{f'fixed_eta{eta_i}' if eta_i is not None else 'new'}" + ".sav", 'wb') as f:
            pickle.dump(info_dict, f)

  if config.tune_vmp_hparams_fix_eta:
    for eta_i in config.eta_plot:
      logging.info(f'Finding best hyperparameters for eta={eta_i}...')
      tune_vmp_hparams(eta_i=eta_i, 
                       cond_hparams_names=[i for i in config.cond_hparams_names if i != 'eta'])
  else:
    tune_vmp_hparams(cond_hparams_names=config.cond_hparams_names)

#####################################################################################
# # For debugging
# config = get_config()
# workdir = str(pathlib.Path.home() / 'spatial-smi-output-exp/all_items/nsf/vmp_flow')
# # train_and_evaluate(config, workdir)

# config.checkpoint_steps = -1
# config.eval_steps = 5000
# config.kernel_kwargs.amplitude = 0.515
# config.kernel_kwargs.length_scale = 0.515
# config.log_img_at_end = False
# config.log_img_steps = -1
# config.optim_kwargs.lr_schedule_kwargs.decay_rate = 0.55
# config.optim_kwargs.lr_schedule_kwargs.peak_value = 0.00031622776601683783
# config.optim_kwargs.lr_schedule_kwargs.transition_steps = 10000
# config.synetune_metric = "mean_dist_anchor_val_min"
# config.training_steps = 30000