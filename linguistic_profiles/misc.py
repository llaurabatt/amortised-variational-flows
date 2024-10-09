"""Miscelaneous auxiliary functions."""
import unicodedata
import string

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


def clean_filename(filename,
                   whitelist="-_.() %s%s" %
                   (string.ascii_letters, string.digits),
                   replace=' ',
                   char_limit=255):
  """Clean a string so it can be used for a filename.
  Source:
    https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
  """
  # replace spaces
  for r in replace:
    filename = filename.replace(r, '_')

  # keep only valid ascii chars
  cleaned_filename = unicodedata.normalize('NFKD',
                                           filename).encode('ASCII',
                                                            'ignore').decode()

  # keep only whitelisted chars
  cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
  if len(cleaned_filename) > char_limit:
    print(
        "Warning, filename truncated because it was over {}. Filenames may no longer be unique"
        .format(char_limit))
  return cleaned_filename[:char_limit]
