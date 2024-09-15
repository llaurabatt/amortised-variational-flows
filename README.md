
# Set up the environment

Environment requirements to run the paper experiments can be found in the environment.yaml file. This file can be used to set up an environment with any environment manager e.g., venv, Conda, Mamba, Micromamba. With Micromamba, you can create and activate the environment as follows:
```
micromamba create -f environment.yaml

micromamba activate <name-environment>
```


# LP small dataset



# OLDDD #
# Simultaneous Reconstruction of Spatial Frequency Fields and Sample Locations via Bayesian Semi-Modular Inference

<!-- badges: start -->
[![License:
MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/chriscarmona/spatial-smi/blob/main/LICENSE)
<!-- badges: end -->

This repo contain the implementation of variational methods described in the article *Simultaneous Reconstruction of Spatial Frequency Fields and Sample Locations via Bayesian Semi-Modular Inference*.

## Examples

We include code to replicate all the examples from our article. By executing the `run.sh` bash script one can train all variational posteriors and produce visualizations and summaries (follow [*Installation instructions*](#installation-instructions) before running the script).

```bash
bash run.sh
```

We recommend to monitor training via tensorboard

```bash
WORK_DIR=$HOME/spatial-smi-output
tensorboard --logdir=$WORK_DIR
```

## Installation instructions

1. \[Optional] Create a new virtual environment for this project (see [*Create a virtual environment*](#creating-a-virtual-environment) below).
2. Install JAX. This may vary according to your CUDA version (See [JAX installation](https://github.com/google/jax#installation)).
3. Clone this repository locally
```bash
git clone https://github.com/chriscarmona/spatial-smi.git
```
4. Install dependencies
```bash
cd spatial-smi
pip install -r requirements.txt
```

## Citation

If you find this work relevant for your scientific publication, we encourage you to add the following reference:

```bibtex
@misc{Carmona2022spatial,
    title = {Simultaneous Reconstruction of Spatial Frequency Fields and Sample Locations via Bayesian Semi-Modular Inference},
    year = {2022},
    author = {Carmona, Chris U. and Haines, Ross A. and Anderson-Loake, Max and Benskin, Michael and Nicholls, Geoff K.},
}

@misc{Carmona2022scalable,
    title = {Scalable Semi-Modular Inference with Variational Meta-Posteriors},
    year = {2022},
    author = {Carmona, Chris U. and Nicholls, Geoff K.},
    month = {4},
    url = {http://arxiv.org/abs/2204.00296},
    doi = {10.48550/arXiv.2204.00296},
    arxivId = {2204.00296},
    keywords = {Cut models, Generalized Bayes, Model misspecification, Scalable inference, Variational Bayes}
}
```

### Creating a virtual environment

For MacOS or Linux, you can use `venv` (see the [venv documentation](https://docs.python.org/3/library/venv.html)).

Create `spatial-smi` virtual environment
```bash
rm -rf ~/.virtualenvs/spatial-smi
python3 -m venv ~/.virtualenvs/spatial-smi
source ~/.virtualenvs/spatial-smi/bin/activate
pip install -U pip
pip install -U setuptools wheel
```

Feel free to modify the directory for the virtual environment by replacing `~/.virtualenvs/spatial-smi` with a path of your choice.
