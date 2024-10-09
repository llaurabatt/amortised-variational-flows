"""Datasets used to illustrate Bayesian Modular Inference."""

from .load import load_lalme, process_lalme

lalme_coarsen_8 = load_lalme(dataset_id='coarsen_8_items')
lalme_coarsen_all = load_lalme(dataset_id='coarsen_all_items')
