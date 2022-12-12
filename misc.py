"""Miscelaneous auxiliary functions."""

from typing import Any, Dict

import jax
import jax.numpy as jnp

import arviz as az
from arviz import InferenceData

from modularbayes._src.typing import Array


def as_lower_chol(x: Array) -> Array:
  """Create a matrix that could be used as Lower Cholesky.

    Args:
      x: Square matrix.

    Returns:
      Lower triangular matrix with positive diagonal.

    Note:
      The function simply masks for the upper triangular part of the matrix
      followed by an exponential transformation to the diagonal, in order to
      make it positive. It does not calculate the cholesky decomposition.
  """
  return jnp.tril(x, -1) + jnp.diag(jax.nn.softplus(jnp.diag(x)))


def issymmetric(a, rtol=1e-05, atol=1e-08):
  return jnp.allclose(a, a.T, rtol=rtol, atol=atol)


def force_symmetric(x, lower=True):
  """Create a symmetric matrix by copying lower/upper diagonal"""
  x_tri = jnp.tril(x) if lower else jnp.triu(x)
  return x_tri + x_tri.T - jnp.diag(jnp.diag(x_tri))


def log1mexpm(x):
  """Accurately Computes log(1 - exp(-x)).

  Source:
    https://cran.r-project.org/web/packages/Rmpfr/
  """

  return jnp.log(-jnp.expm1(-x))


def lalme_az_from_dict(
    samples_dict: Dict[str, Any],
    lalme_dataset: Dict[str, Any],
) -> InferenceData:
  """Converts a posterior dictionary to an ArviZ InferenceData object.

  Args:
    posterior_dict: Dictionary of posterior samples.

  Returns:
    ArviZ InferenceData object.
  """

  samples_dict.update(
      {f"W_{i}": x for i, x in enumerate(samples_dict['mixing_weights_list'])})
  del samples_dict['mixing_weights_list']
  samples_dict.update(
      {f"a_{i}": x for i, x in enumerate(samples_dict['mixing_offset_list'])})
  del samples_dict['mixing_offset_list']

  num_basis_gps, num_inducing_points = samples_dict['gamma_inducing'].shape[-2:]

  coords_lalme = {
      "mu_items":
          lalme_dataset['items'],
      "zeta_items":
          lalme_dataset['items'],
      "gamma_basis":
          range(num_basis_gps),
      "gamma_loc_inducing":
          range(num_inducing_points),
      "loc_floating_profiles":
          lalme_dataset['LP'][-lalme_dataset['num_profiles_floating']:],
      "loc_floating_coords": ["x", "y"],
  }
  dims_lalme = {
      "mu": ["mu_items"],
      "zeta": ["zeta_items"],
      "gamma_inducing": ["gamma_basis", "gamma_loc_inducing"],
      "loc_floating": ["loc_floating_profiles", "loc_floating_coords"],
  }

  if "loc_floating_aux" in samples_dict:
    coords_lalme.update({
        "loc_floating_aux_profiles":
            lalme_dataset['LP'][-lalme_dataset['num_profiles_floating']:],
        "loc_floating_aux_coords": ["x", "y"],
    })
    dims_lalme.update({
        "loc_floating_aux": [
            "loc_floating_aux_profiles", "loc_floating_aux_coords"
        ]
    })

  for i, forms_i in enumerate(lalme_dataset['forms']):
    coords_lalme.update({f"W_{i}_basis": range(num_basis_gps)})
    coords_lalme.update({f"W_{i}_forms": forms_i})
    coords_lalme.update({f"a_{i}_forms": forms_i})
    dims_lalme.update({f"W_{i}": [f"W_{i}_basis", f"W_{i}_forms"]})
    dims_lalme.update({f"a_{i}": [f"a_{i}_forms"]})

  lalme_az = az.convert_to_inference_data(
      samples_dict,
      coords=coords_lalme,
      dims=dims_lalme,
  )

  return lalme_az