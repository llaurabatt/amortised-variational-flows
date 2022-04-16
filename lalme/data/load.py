import pkg_resources

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from modularbayes._src.typing import Any, Dict, Optional

kernels = tfp.math.psd_kernels
tfd = tfp.distributions

Array = jnp.ndarray
PRNGKey = jnp.ndarray
Kernel = kernels.PositiveSemidefiniteKernel


def load_lalme(dataset_id: str = 'coarsen_8_items') -> Dict[str, Any]:
  """Load data from the Linguistic Atlas of Late Mediaeval English.

    Args:
        dataset_id : String describing the type of data to be read.

    Return:
        data: Dictionary with the data used in the LALME model.

    Reference:
        M. Benskin, M. Laing, V. Karaiskos and K. Williamson. An Electronic Version of A Linguistic
        Atlas of Late Mediaeval English [http://www.lel.ed.ac.uk/ihd/elalme/elalme.html]
    """

  data = {}

  files = [
      f'{dataset_id}_loc.csv', f'{dataset_id}_m.csv', f'{dataset_id}_items.txt',
      f'{dataset_id}_forms.txt', 'floating_profiles_id.txt'
  ]

  stream = [pkg_resources.resource_stream(__name__, file) for file in files]

  ### Loading Data
  loc_df = pd.read_csv(stream[0])
  y_concat_df = pd.read_csv(stream[1], index_col=0)
  items_all = np.genfromtxt(stream[2], dtype='S', delimiter=',')
  forms_all = np.genfromtxt(stream[3], dtype='S', delimiter=',')
  floating_id = np.genfromtxt(stream[4], dtype=int, delimiter=',')

  # Identify anchor/floating profiles
  loc_df['floating'] = loc_df['LP'].isin(floating_id)

  # Rearange indices to put anchor profiles first
  anchor_first_indices = np.concatenate(
      [np.where(~loc_df['floating'])[0],
       np.where(loc_df['floating'])[0]])
  loc_df = loc_df.iloc[anchor_first_indices, :]
  y_concat_df = y_concat_df.iloc[:, anchor_first_indices]

  assert all(
      y_concat_df.columns.astype(int).to_numpy() == loc_df['LP'].to_numpy())

  # Number of profiles
  data['num_profiles'] = loc_df.shape[0]
  data['num_profiles_floating'] = int(loc_df['floating'].sum())
  data['num_profiles_anchor'] = data['num_profiles'] - data[
      'num_profiles_floating']

  ### Profile locations ###
  data['loc'] = loc_df[['Easting', 'Northing']].to_numpy()

  # Get unique items from data (avoid sorting alphabetically)
  items, _index = np.unique(items_all, return_index=True)
  items = items[_index.argsort()]
  data['items'] = items
  data['num_items'] = len(items)

  # Get forms
  data['num_forms_tuple'] = ()
  for item in items:
    data['num_forms_tuple'] = data['num_forms_tuple'] + (np.sum(
        items_all == item),)
  forms = np.split(forms_all, np.cumsum(data['num_forms_tuple'])[:-1])
  data['forms'] = forms
  assert all(forms[i].shape[0] == data['num_forms_tuple'][i]
             for i in range(data['num_items']))

  # Get Y
  # data['y'] will be a list, with one element per item
  # every element is a matrix, with rows equal to the number of forms in that item,
  # and columns equal to the number of profiles
  data['y'] = np.split(
      y_concat_df.to_numpy(), np.cumsum(data['num_forms_tuple'])[:-1], axis=0)
  assert all(data['y'][i].shape[0] == data['num_forms_tuple'][i]
             for i in range(data['num_items']))
  data['num_forms_tuple'] = tuple(data['num_forms_tuple'])

  return data


def process_lalme(
    data_raw: Dict,
    num_profiles_anchor_keep: Optional[int] = None,
    num_profiles_floating_keep: Optional[int] = None,
    num_items_keep: Optional[int] = None,
    loc_bounds: Optional[np.ndarray] = 5. * np.array([[-1., 1.], [-1., 1.]]),
    remove_empty_forms: bool = True,
) -> Dict[str, Any]:
  """Data processing for the LALME dataset.

    Args:
        data : Dictionary with LALME data, as produced by load_lalme.
        loc_bounds : Array indicating the bounded area to map profile locations
            from their original Easting/Northing coordinates.
        num_profiles_anchor_keep : Number of anchor profiles to keep. If None, keep all profiles.
        num_profiles_floating_keep : Number of floating profiles to keep. If None, keep all profiles.
        remove_empty_forms : If True, eliminates forms that have not a single profile using it.

    Return:
        data: Dictionary with the data used in the LALME model.

    Reference:
        M. Benskin, M. Laing, V. Karaiskos and K. Williamson. An Electronic Version of A Linguistic
        Atlas of Late Mediaeval English [http://www.lel.ed.ac.uk/ihd/elalme/elalme.html]
    """

  data = data_raw.copy()

  # Map locations to the square defined by loc_bounds
  if loc_bounds is not None:
    scaler = MinMaxScaler()
    data['loc'] = scaler.fit_transform(data['loc'])
    data['loc'] *= loc_bounds[:, 1] - loc_bounds[:, 0]
    data['loc'] += loc_bounds[:, 0]

  ### Subsetting data ###

  # Profiles subsetting #
  num_profiles_anchor = data['num_profiles_anchor']
  num_profiles_floating = data['num_profiles_floating']
  num_profiles_anchor_keep = int(
      num_profiles_anchor_keep
  ) if num_profiles_anchor_keep is not None else num_profiles_anchor
  num_profiles_floating_keep = int(
      num_profiles_floating_keep
  ) if num_profiles_floating_keep is not None else num_profiles_floating
  assert 0 < num_profiles_anchor_keep <= num_profiles_anchor
  assert 0 < num_profiles_floating_keep <= num_profiles_floating
  if (num_profiles_anchor_keep < num_profiles_anchor) or (
      num_profiles_floating_keep < num_profiles_floating):
    # profiles to be kept
    index_profiles_anchor_keep = np.arange(num_profiles_anchor_keep)

    index_profiles_floating_keep = np.arange(
        1 + num_profiles_anchor, num_profiles_anchor +
        num_profiles_floating)[:num_profiles_floating_keep]

    # Select these profiles only
    index_profiles_keep = np.concatenate(
        [index_profiles_anchor_keep, index_profiles_floating_keep])

    # Subset elements in data dictionary
    data['loc'] = data['loc'][index_profiles_keep, :]
    data['y'] = [y_item[:, index_profiles_keep] for y_item in data['y']]
    data['num_profiles'] = num_profiles_anchor_keep + num_profiles_floating_keep
    data['num_profiles_floating'] = num_profiles_floating_keep

    assert all(
        [y_item.shape[1] == data['num_profiles'] for y_item in data['y']])
    assert data['loc'].shape[0] == data['num_profiles']

  # Items subsetting
  num_items = data['num_items']
  num_items_keep = (
      int(num_items_keep) if num_items_keep is not None else num_items)
  assert 0 < num_items_keep <= num_items
  if num_items_keep < num_items:
    # Items to keep
    index_items_keep = np.arange(num_items_keep)
    data['items'] = data['items'][index_items_keep]
    data['num_items'] = num_items_keep
    data['num_forms_tuple'] = tuple(
        data['num_forms_tuple'][i] for i in index_items_keep)
    data['forms'] = [data['forms'][i] for i in index_items_keep]
    data['y'] = [data['y'][i] for i in index_items_keep]

  # Forms subsetting

  if remove_empty_forms:
    for i in range(data['num_items']):
      # identifies forms that actually appear in profiles
      form_used = np.where(data['y'][i].sum(axis=1) > 0)[0]
      # Only keep those in the data
      data['num_forms_tuple'] = data['num_forms_tuple'][:i] + (
          len(form_used),) + data['num_forms_tuple'][(i + 1):]
      data['forms'][i] = data['forms'][i][form_used]
      data['y'][i] = data['y'][i][form_used, :]

  return data
