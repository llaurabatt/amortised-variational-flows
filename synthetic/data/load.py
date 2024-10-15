import pkg_resources

import pandas as pd
from .gen_data import gen_data
import os
import pickle


def load_synthetic(n_groups: int, 
                   n_obs: int,
                   seed: int):
  """Synthetic data.
    """

  dir_path = os.path.dirname(__file__)
  data_filename = f'synthetic_data_SEED{seed}_{n_obs}obs_{n_groups}groups.csv'
  data_file_path = os.path.join(dir_path, data_filename)

  true_params_filename = f'true_params_SEED{seed}_{n_obs}obs_{n_groups}groups.pickle'
  true_params_file_path = os.path.join(dir_path, true_params_filename)

  sim_data_filename = f'sim_data_summary_SEED{seed}_{n_obs}obs_{n_groups}groups.png'
  sim_data_file_path = os.path.join(dir_path, sim_data_filename)
  
  fig = gen_data(data_filename=data_file_path, true_params_filename=true_params_file_path, 
           save_sim_data_filename=sim_data_file_path,
           SEED=seed, n_groups=n_groups, n_obs=n_obs)

  dataset = pd.read_csv(data_file_path)
  with open(true_params_file_path, 'rb') as f:
      true_params = pickle.load(f) 
  return dataset, true_params, fig
