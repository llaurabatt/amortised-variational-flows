#!/bin/bash

# Assume we are located at main repo directory
REPO_DIR=$PWD

# Directory to save all outputs
WORK_DIR=$HOME/spatial-smi/output

# Create output directory and install missing dependencies
mkdir -p $WORK_DIR
pip install -r $REPO_DIR/requirements.txt

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

## Variational Meta-Posterior via VMP-map
### Mean Field
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/flow_mf_vmp_map.py \
                                --workdir=$WORK_DIR/all_items/mean_field/vmp_map
### Neural Spline Flow
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/flow_nsf_vmp_map.py \
                                --workdir=$WORK_DIR/all_items/nsf/vmp_map


## Variational Meta-Posterior via VMP-flow
### Neural Spline Flow
python3 $REPO_DIR/lalme/main.py --config=$REPO_DIR/lalme/flow_nsf_vmp_flow.py \
                                --workdir=$WORK_DIR/all_items/nsf/vmp_flow
