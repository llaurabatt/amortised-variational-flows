"""Miscelaneous auxiliary functions."""

import jax
import jax.numpy as jnp

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


def force_symmetric(A, lower=True):
  """Create a symmetric matrix by copying lower/upper diagonal"""
  A_tri = jnp.tril(A) if lower else jnp.triu(A)
  return A_tri + A_tri.T - jnp.diag(jnp.diag(A_tri))


def log1mexpm(x):
  """Accurately Computes log(1 - exp(-x)).

  Source:
    https://cran.r-project.org/web/packages/Rmpfr/
  """

  return jnp.log(-jnp.expm1(-x))
