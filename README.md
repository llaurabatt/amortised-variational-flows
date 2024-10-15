This repo contain the implementation of the experiments in the article *Amortising Variational Bayesian Inference over prior hyperparameters with a Normalising Flow*.

# Set up the environment

Environment requirements to run the paper experiments can be found in the environment.yaml file. This file can be used to set up an environment with any environment manager e.g., venv, Conda, Mamba, Micromamba. 
For installing the environment with Micromamba, first install Micromamba following the instruction on the [Mamba website](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
You can the create and activate the environment as follows:
```
micromamba create -f environment.yaml

micromamba activate <name-environment>
```
This will automatically install the ModularBayes package together with its dependencies and all the other packages required to run the experiments in the paper.
To additionally install JAX in the environment, please follow the instructions for the chosen installation option in the official JAX website (see [JAX installation](https://jax.readthedocs.io/en/latest/installation.html)).

# Reproduce paper experiments

Clone this repository locally
```bash
git clone https://github.com/llaurabatt/my-spatial-smi-oldv.git
```
For reproducing the experiments, run the following files:

**Synthetic** experiments:
```
run_synthetic.sh
``` 
**Epidemiological** dataset experiments:
```
run_epidemiology.sh
```
**Linguistic profiles** experiments on the **small** dataset:
```
run_linguistic_profiles_small.sh
```
**Linguistic profiles** experiments on the **large** dataset:
```
run_linguistic_profiles_all.sh
```

