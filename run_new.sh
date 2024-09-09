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

# eta_mcmc=1.0
# python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_mcmc.py \
#                         --workdir $WORK_DIR/5_items/mcmc/w500_s50_000_t50_sub100_eta_floating_$eta_mcmc \
#                         --config.eta_profiles_floating "$eta_mcmc" \
#                         --log_dir $WORK_DIR/5_items/mcmc/w500_s50_000_t50_sub100_eta_floating_$eta_mcmc/log_dir \
#                         --alsologtostderr


MCMC_5_ITEMS_DIR=$HOME/mount/vmp-output/lp-small/5_items/mcmc/w500_s50_000_t50_sub100_eta_floating_1.0

## VMP eta only

python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_flow_nsf_vmp_flow_etaVMP.py \
                          --config.path_mcmc_img $MCMC_5_ITEMS_DIR/lalme_floating_profiles_grid_eta_floating_1.000.png \
                          --config.path_MCMC_samples $MCMC_5_ITEMS_DIR/lalme_az_w500_s50000_t50_sub100.nc \
                          --config.wandb_project_name "LP-5items-VMP-eta" \
                          --workdir $WORK_DIR/5_items/vmp_eta \
                          --log_dir $WORK_DIR/5_items/vmp_eta/log_dir \
                          --alsologtostderr

## AdditiveVMP eta only

python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_flow_nsf_vmp_flow_etaAdditiveVMP.py \
                          --config.path_mcmc_img $MCMC_5_ITEMS_DIR/lalme_floating_profiles_grid_eta_floating_1.000.png \
                          --config.path_MCMC_samples $MCMC_5_ITEMS_DIR/lalme_az_w500_s50000_t50_sub100.nc \
                          --config.wandb_project_name "LP-5items-AdditiveVMP-eta" \
                          --workdir $WORK_DIR/5_items/additive_vmp_eta \
                          --log_dir $WORK_DIR/5_items/additive_vmp_eta/log_dir \
                          --alsologtostderr

## VP eta = 1.0

python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_flow_nsf_vmp_flow_VPeta1.py \
                          --config.path_mcmc_img $MCMC_5_ITEMS_DIR/lalme_floating_profiles_grid_eta_floating_1.000.png \
                          --config.path_MCMC_samples $MCMC_5_ITEMS_DIR/lalme_az_w500_s50000_t50_sub100.nc \
                          --config.wandb_project_name "LP-5items-VP-eta1" \
                          --workdir $WORK_DIR/5_items/vp_eta1 \
                          --log_dir $WORK_DIR/5_items/vp_eta1/log_dir \
                          --alsologtostderr