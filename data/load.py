"""LALME data loading and processing."""
from collections import namedtuple

from absl import logging
import pkg_resources

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from jax import numpy as jnp
import jax
from tensorflow_probability.substrates import jax as tfp

from modularbayes._src.typing import Any, Dict, List, Optional

kernels = tfp.math.psd_kernels
tfd = tfp.distributions

Array = jnp.ndarray
PRNGKey = jnp.ndarray
Kernel = kernels.PositiveSemidefiniteKernel

LpSplit = namedtuple("lp_split", [
    'lp_anchor_train',
    'lp_anchor_val',
    'lp_anchor_test',
    'lp_floating_train',
])


def load_lalme(dataset_id: str = 'coarsen_8_items',
               floating_anchor_copies:bool = False) -> Dict[str, Any]:
  """Load data from the Linguistic Atlas of Late Mediaeval English.

    Args:
        dataset_id : String describing the type of data to be read.

    Return:
        data: Dictionary with the data used in the LALME model.

    Reference:
        M. Benskin, M. Laing, V. Karaiskos and K. Williamson.
        An Electronic Version of A Linguistic
        Atlas of Late Mediaeval English
        [http://www.lel.ed.ac.uk/ihd/elalme/elalme.html]
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

  # Transform bytes into strings
  items_all = items_all.astype(str)
  forms_all = [x.astype(str) for x in forms_all]

  # Identify anchor/floating profiles
  loc_df['floating'] = loc_df['LP'].isin(floating_id)

  # Rearange indices to put anchor profiles first
  anchor_first_indices = np.concatenate(
      [np.where(~loc_df['floating'])[0],
       np.where(loc_df['floating'])[0]])
  loc_df = loc_df.iloc[anchor_first_indices, :]
  y_concat_df = y_concat_df.iloc[:, anchor_first_indices]
  # jax.debug.breakpoint()

  if floating_anchor_copies:
    num_profiles_anchor = np.where(~loc_df['floating'])[0].shape[0]

    loc_df = pd.concat([loc_df.iloc[ :num_profiles_anchor, :],
                        loc_df.iloc[ :num_profiles_anchor, :]], axis=0).reset_index(drop=True).copy()
    
    y_concat_df = pd.concat([y_concat_df.iloc[:, :num_profiles_anchor],
                        y_concat_df.iloc[:, :num_profiles_anchor]], axis=1).reset_index(drop=True).copy()
    
    assert all(
      y_concat_df.columns.astype(int).to_numpy() == loc_df['LP'].to_numpy())
    
    # assign anchor copy to floating
    assert int(loc_df['floating'].sum()) == 0
    loc_df['floating'] = pd.concat([loc_df['floating'].iloc[:num_profiles_anchor], 
                                    pd.Series([True]*num_profiles_anchor)]).reset_index(drop=True)
    assert  (loc_df['floating'].values == pd.concat([pd.Series([False]*num_profiles_anchor), 
                                    pd.Series([True]*num_profiles_anchor)]).reset_index(drop=True).values).sum() == num_profiles_anchor*2
    
    loc_df['LP'][loc_df.floating == True] = loc_df['LP'][loc_df.floating == True].apply(lambda x: str(x) + '_c')



  assert all(
      y_concat_df.columns.astype(int).to_numpy() == loc_df['LP'].to_numpy())

  # Number of profiles
  data['num_profiles'] = loc_df.shape[0]
  data['num_profiles_floating'] = int(loc_df['floating'].sum())
  data['num_profiles_anchor'] = data['num_profiles'] - data[
      'num_profiles_floating']

  ### Profile locations ###
  data['LP'] = loc_df['LP'].values
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
  assert all(data['forms'][i].shape[0] == data['num_forms_tuple'][i]
             for i in range(data['num_items']))

  # Get Y
  # data['y'] will be a list, with one element per item
  # every element is a matrix, with rows equal to the number of forms for it,
  # and columns equal to the number of profiles
  data['y'] = np.split(
      y_concat_df.to_numpy(), np.cumsum(data['num_forms_tuple'])[:-1], axis=0)
  assert all(data['y'][i].shape[0] == data['num_forms_tuple'][i]
             for i in range(data['num_items']))
  data['num_forms_tuple'] = tuple(data['num_forms_tuple'])

  return data


def process_lalme(
    lalme_dataset: Dict,
    lp_anchor_train: List[int],
    lp_floating_train: List[int],
    items_keep: List[str],
    loc_bounds: Optional[np.ndarray] = np.array([[-1., 1.], [-1., 1.]]),
    lp_anchor_val: Optional[List[int]] = None,
    lp_anchor_test: Optional[List[int]] = None,
    remove_empty_forms: bool = True,
) -> Dict[str, Any]:
  """Data processing for the LALME dataset.

    Args:
        data : Dictionary with LALME data, as produced by load_lalme.
        loc_bounds : Array indicating the bounded area to map profile locations
          from their original Easting/Northing coordinates.
        num_lp_anchor_train : Number of anchor profiles considered for
          training. The exact locations of these profiles are known and no
          imputation is done for them.
        num_lp_anchor_val : Number of anchor profiles considered
          for validation. The exact locations of these profiles are known, but
          we reconstruct them using the model. Hyperparameters (eg. kernel)
          are tuned on these profiles.
        num_lp_anchor_test : Number of anchor profiles considered
          for validation. The exact locations of these profiles are known, but
          we reconstruct them using the model. These are not used for tunning
          but rather to estimate the error of the model.
        num_lp_floating_train : Number of floating profiles to keep.
        remove_empty_forms : If True, eliminates forms that have not a single
            profile using it.

    Return:
        data: Dictionary with the data used in the LALME model.

    Reference:
        M. Benskin, M. Laing, V. Karaiskos and K. Williamson. An Electronic
        Version of A Linguistic Atlas of Late Mediaeval English
        [http://www.lel.ed.ac.uk/ihd/elalme/elalme.html]
    """

  # Copy data to avoid modifying the original
  lalme_dataset = lalme_dataset.copy()

  # Set default values
  if lp_anchor_val is None:
    lp_anchor_val = []
  if lp_anchor_test is None:
    lp_anchor_test = []

  # Map locations to the square defined by loc_bounds
  if loc_bounds is not None:
    scaler = MinMaxScaler()
    lalme_dataset['loc'] = scaler.fit_transform(lalme_dataset['loc'])
    lalme_dataset['loc'] *= loc_bounds[:, 1] - loc_bounds[:, 0]
    lalme_dataset['loc'] += loc_bounds[:, 0]

  ### Subsetting data ###

  ## Profiles subsetting ##
  lp_keep = np.array(lp_anchor_train + lp_anchor_val + lp_anchor_test +
                     lp_floating_train)
  lalme_dataset['num_profiles'] = len(lp_keep)
  lalme_dataset['num_profiles_anchor'] = len(lp_anchor_train)
  lalme_dataset['num_profiles_floating'] = len(lp_anchor_val + lp_anchor_test +
                                               lp_floating_train)
  lalme_dataset['num_profiles_split'] = LpSplit(
      len(lp_anchor_train), len(lp_anchor_val), len(lp_anchor_test),
      len(lp_floating_train))

  assert len(lp_keep) == len(np.unique(lp_keep))
  assert np.in1d(lp_keep, lalme_dataset['LP']).all()
  # return indices of lalme_dataset['LP'] corresponding to lp_keep
  lp_keep_idx = np.array(
      [np.where(lalme_dataset['LP'] == lp_)[0][0] for lp_ in lp_keep])
  assert (lalme_dataset['LP'][lp_keep_idx] == lp_keep).all()

  # Subset elements in data dictionary
  lalme_dataset['LP'] = lalme_dataset['LP'][lp_keep_idx]
  lalme_dataset['loc'] = lalme_dataset['loc'][lp_keep_idx, :]
  lalme_dataset['y'] = [y_item[:, lp_keep_idx] for y_item in lalme_dataset['y']]

  assert all([
      y_item.shape[1] == lalme_dataset['num_profiles']
      for y_item in lalme_dataset['y']
  ])
  assert lalme_dataset['LP'].shape[0] == lalme_dataset['num_profiles']
  assert lalme_dataset['loc'].shape[0] == lalme_dataset['num_profiles']

  ## Forms subsetting ##
  if remove_empty_forms:
    for i in range(lalme_dataset['num_items']):
      # identifies forms that actually appear in profiles
      form_used = np.where(lalme_dataset['y'][i].sum(axis=1) > 0)[0]
      # Only keep those form with at least one observation
      lalme_dataset[
          'num_forms_tuple'] = lalme_dataset['num_forms_tuple'][:i] + (
              len(form_used),) + lalme_dataset['num_forms_tuple'][(i + 1):]
      lalme_dataset['forms'][i] = lalme_dataset['forms'][i][form_used]
      lalme_dataset['y'][i] = lalme_dataset['y'][i][form_used, :]

  ## Items subsetting ##

  # Keep only items with at least two forms
  # (otherwise, the model cannot be trained because softmax over fields is undefined)
  items_ok_bool = np.array(
      [f_i >= 2 for f_i in lalme_dataset['num_forms_tuple']])
  if not all(items_ok_bool):
    logging.warning("Removing items with less than two forms: %s",
                    str(lalme_dataset['items'][~items_ok_bool]))
  items_keep = np.intersect1d(lalme_dataset['items'][items_ok_bool], items_keep)

  assert len(items_keep) > 0
  items_keep_idx = np.array(
      [np.where(lalme_dataset['items'] == i_)[0][0] for i_ in items_keep])
  assert (lalme_dataset['items'][items_keep_idx] == items_keep).all()

  lalme_dataset['items'] = lalme_dataset['items'][items_keep_idx]
  lalme_dataset['num_items'] = len(items_keep)
  lalme_dataset['num_forms_tuple'] = tuple(
      lalme_dataset['num_forms_tuple'][i] for i in items_keep_idx)
  lalme_dataset['forms'] = [lalme_dataset['forms'][i] for i in items_keep_idx]
  lalme_dataset['y'] = [lalme_dataset['y'][i] for i in items_keep_idx]

  return lalme_dataset
