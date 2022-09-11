#!/bin/bash
set -e
set -x

# Assume we are located at main repo directory
REPO_DIR=$PWD

# Directory to save all outputs
WORK_DIR=$HOME/spatial-smi-output

# Create output directory and install missing dependencies
mkdir -p $WORK_DIR
pip install -Ur $REPO_DIR/requirements.txt

all_eta=('0.001' '0.2' '0.5' '1.0')

### 8 ITEMS ###

## MCMC
## Single eta
for eta in "${all_eta[@]}"
do
  python3 $REPO_DIR/main.py --config=$REPO_DIR/configs/mcmc.py \
                            --workdir=$WORK_DIR/8_items/mcmc/eta_floating_$eta \
                            --config.eta_profiles_floating=$eta \
                            --log_dir=$WORK_DIR/8_items/mcmc/log \
                            --alsologtostderr
done

## Variational inference, replication of MCMC
## Single eta, Mean field (MFVI)
for eta in "${all_eta[@]}"
do
  python3 $REPO_DIR/main.py --config=$REPO_DIR/configs/flow_mf_like_mcmc.py \
                            --workdir=$WORK_DIR/8_items/mf/eta_floating_$eta \
                            --config.eta_profiles_floating=$eta \
                            --log_dir=$WORK_DIR/8_items/mf/log \
                            --alsologtostderr
done

## Variational inference, replication of MCMC
## Single eta, Neural Spline Flow
for eta in "${all_eta[@]}"
do
  python3 $REPO_DIR/main.py --config=$REPO_DIR/configs/flow_nsf_like_mcmc.py \
                            --workdir=$WORK_DIR/8_items/nsf/eta_floating_$eta \
                            --config.eta_profiles_floating=$eta \
                            --log_dir=$WORK_DIR/8_items/nsf/log_single_eta \
                            --alsologtostderr
done

## Variational Meta-Posterior via VMP-flow, replication of MCMC
### Neural Spline Flow
python3 $REPO_DIR/main.py --config=$REPO_DIR/configs/flow_nsf_vmp_flow_like_mcmc.py \
                          --workdir=$WORK_DIR/8_items/nsf/vmp_flow \
                          --log_dir=$WORK_DIR/8_items/nsf/log_vmp_flow \
                          --alsologtostderr

### ALL ITEMS ###

## Single eta, Mean field
for eta in "${all_eta[@]}"
do
  python3 $REPO_DIR/main.py --config=$REPO_DIR/configs/flow_mf.py \
                            --workdir=$WORK_DIR/all_items/mf/eta_floating_$eta \
                            --config.eta_profiles_floating=$eta \
                            --config.dataset_id='coarsen_all_items' \
                            --log_dir=$WORK_DIR/all_items/mf/log \
                            --alsologtostderr
done

## Single eta, Neural Spline Flow
for eta in "${all_eta[@]}"
do
  python3 $REPO_DIR/main.py --config=$REPO_DIR/configs/flow_nsf.py \
                            --workdir=$WORK_DIR/all_items/nsf/eta_floating_$eta \
                            --config.eta_profiles_floating=$eta \
                            --config.dataset_id='coarsen_all_items' \
                            --log_dir=$WORK_DIR/all_items/nsf/log_single_eta \
                            --alsologtostderr
done

## Variational Meta-Posterior via VMP-flow
### Neural Spline Flow
python3 $REPO_DIR/main.py --config=$REPO_DIR/configs/flow_nsf_vmp_flow.py \
                          --workdir=$WORK_DIR/all_items/nsf/vmp_flow \
                          --log_dir=$WORK_DIR/all_items/nsf/log_vmp_flow \
                          --alsologtostderr
