#!/bin/bash

# Assume we are located at main repo directory
REPO_DIR=$PWD

# Directory to save all outputs
WORK_DIR=$HOME/spatial-smi/output

# Create output directory and install missing dependencies
mkdir -p $WORK_DIR
pip install -Ur $REPO_DIR/requirements.txt

eta_floating='(0.001,0.2,0.5,1.0)'

### 8 ITEMS ###

## MCMC
## Single eta
mkdir -p $WORK_DIR/8_items/mcmc/log
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/mcmc.py \
                                --workdir=$WORK_DIR/8_items/mcmc/eta_floating \
                                --config.iterate_smi_eta=$eta_floating \
                                --log_dir=$WORK_DIR/8_items/mcmc/log

## Variational inference, replication of MCMC
## Single eta, Mean field (MFVI)
mkdir -p $WORK_DIR/8_items/mf/log
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_mf_like_mcmc.py \
                                --workdir=$WORK_DIR/8_items/mf/eta_floating \
                                --config.iterate_smi_eta=$eta_floating \
                                --log_dir=$WORK_DIR/8_items/mf/log

## Variational inference, replication of MCMC
## Single eta, Neural Spline Flow
mkdir -p $WORK_DIR/8_items/nsf/log_single_eta
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_nsf_like_mcmc.py \
                                --workdir=$WORK_DIR/8_items/nsf/eta_floating \
                                --config.iterate_smi_eta=$eta_floating \
                                --log_dir=$WORK_DIR/8_items/nsf/log_single_eta

## Variational Meta-Posterior via VMP-flow, replication of MCMC
### Neural Spline Flow
mkdir -p $WORK_DIR/8_items/nsf/log_vmp_flow
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_nsf_vmp_flow_like_mcmc_max.py \
                                --workdir=$WORK_DIR/8_items/nsf/vmp_flow \
                                --log_dir=$WORK_DIR/8_items/nsf/log_vmp_flow

### ALL ITEMS ###

## Single eta, Mean field
mkdir -p $WORK_DIR/all_items/mf/log
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_mf.py \
                                --workdir=$WORK_DIR/all_items/mf/eta_floating \
                                --config.iterate_smi_eta=$eta_floating \
                                --config.dataset_id='coarsen_all_items' \
                                --log_dir=$WORK_DIR/all_items/mf/log

## Single eta, Neural Spline Flow
mkdir -p $WORK_DIR/all_items/nsf/log_single_eta
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_nsf.py \
                                --workdir=$WORK_DIR/all_items/nsf/eta_floating \
                                --config.iterate_smi_eta=$eta_floating \
                                --config.dataset_id='coarsen_all_items' \
                                --log_dir=$WORK_DIR/all_items/nsf/log_single_eta

## Variational Meta-Posterior via VMP-flow
### Neural Spline Flow
mkdir -p $WORK_DIR/all_items/nsf/log_vmp_flow
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/configs/flow_nsf_vmp_flow.py \
                                --workdir=$WORK_DIR/all_items/nsf/vmp_flow \
                                --log_dir=$WORK_DIR/all_items/nsf/log_vmp_flow
