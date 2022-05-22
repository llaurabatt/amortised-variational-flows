#!/bin/bash

# Assume we are located at main repo directory
REPO_DIR=$PWD

# Directory to save all outputs
WORK_DIR=$HOME/spatial-smi/output

# Create output directory and install missing dependencies
mkdir -p $WORK_DIR
pip install -Ur $REPO_DIR/requirements.txt

## Variational replication of MCMC
## Single eta, Mean field
eta_floating='(1.0,)'
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_mf_like_mcmc.py \
                                --workdir=$WORK_DIR/8_items/mf/eta_floating \
                                --config.iterate_smi_eta=$eta_floating
## Single eta, Neural Spline Flow
eta_floating='(1.0,)'
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_nsf_like_mcmc.py \
                                --workdir=$WORK_DIR/8_items/nsf/eta_floating \
                                --config.iterate_smi_eta=$eta_floating
## Variational Meta-Posterior via VMP-flow
### Neural Spline Flow
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_nsf_vmp_flow_like_mcmc.py \
                                --workdir=$WORK_DIR/8_items/nsf/vmp_flow

## One posterior for each eta
eta_floating='(0.001,0.5,1.0)'
#### Mean field Variational Inference (MFVI)
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_mf.py \
                                --workdir=$WORK_DIR/all_items/mean_field/eta_floating \
                                --config.iterate_smi_eta=$eta_floating \
                                --config.dataset_id='coarsen_all_items'
#### Neural Spline Flow
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_nsf.py \
                                --workdir=$WORK_DIR/all_items/nsf/eta_floating \
                                --config.iterate_smi_eta=$eta_floating \
                                --config.dataset_id='coarsen_all_items'
## Variational Meta-Posterior via VMP-flow
### Neural Spline Flow
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_nsf_vmp_flow.py \
                                --workdir=$WORK_DIR/all_items/nsf/vmp_flow
