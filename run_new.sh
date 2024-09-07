#!/bin/bash
set -e
set -x


# Assume we are located at main repo directory
REPO_DIR=$PWD

# Directory to save all outputs

WORK_DIR=$HOME/mount/vmp-output/lp-small
# Create output directory and install missing dependencies
mkdir -p $WORK_DIR


########## 5 ITEMS

## MCMC

eta_mcmc=1.0
python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_mcmc.py \
                        --workdir $WORK_DIR/5_items/mcmc/w500_s50_000_t50_sub100_eta_floating_$eta_mcmc \
                        --config.eta_profiles_floating $eta_mcmc \
                        --log_dir $WORK_DIR/5_items/mcmc/w500_s50_000_t50_sub100_eta_floating_$eta_mcmc/log_dir \
                        --alsologtostderr