#!/bin/bash
set -e
set -x


# Assume we are located at main repo directory
REPO_DIR=$PWD/linguistic_profiles

# Directory to save all outputs

WORK_DIR=$HOME/mount/vmp-output/lp-small

# Create output directory and install missing dependencies
mkdir -p $WORK_DIR


########## 5 ITEMS ##########

## MCMC

eta_mcmc=1.0
python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_mcmc.py \
                        --workdir $WORK_DIR/5_items/mcmc/w500_s50_000_t50_sub100_eta_floating_$eta_mcmc \
                        --config.eta_profiles_floating "$eta_mcmc" \
                        --log_dir $WORK_DIR/5_items/mcmc/w500_s50_000_t50_sub100_eta_floating_$eta_mcmc/log_dir \
                        --alsologtostderr


MCMC_5_ITEMS_DIR=$HOME/mount/vmp-output/lp-small/5_items/mcmc/w500_s50_000_t50_sub100_eta_floating_1.0

# Run the three scripts below if you want to do VI HPO
## HPO AdditiveVMP eta only

# python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_hpo_flow_nsf_vmp_flow_etaAdditiveVMP.py \
#                           --config.path_mcmc_img $MCMC_5_ITEMS_DIR/lalme_floating_profiles_grid_eta_floating_1.000.png \
#                           --config.path_MCMC_samples $MCMC_5_ITEMS_DIR/lalme_az_w500_s50000_t50_sub100.nc \
#                           --config.wandb_project_name "LP-5items-HPO-AdditiveVMP-eta2" \
#                           --config.n_sweep_runs 10 \
#                           --config.sweep_id '4xn6yzi5'\
#                           --workdir $WORK_DIR/5_items/additive_vmp_eta/hpo \
#                           --log_dir $WORK_DIR/5_items/additive_vmp_eta/hpo/log_dir \
#                           --alsologtostderr
# ## HPO VMP eta only

# python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_hpo_flow_nsf_vmp_flow_etaVMP.py \
#                           --config.path_mcmc_img $MCMC_5_ITEMS_DIR/lalme_floating_profiles_grid_eta_floating_1.000.png \
#                           --config.path_MCMC_samples $MCMC_5_ITEMS_DIR/lalme_az_w500_s50000_t50_sub100.nc \
#                           --config.wandb_project_name "LP-5items-HPO-VMP-eta" \
#                           --config.n_sweep_runs 10 \
#                           --workdir $WORK_DIR/5_items/vmp_eta/hpo \
#                           --log_dir $WORK_DIR/5_items/vmp_eta/hpo/log_dir \
#                           --alsologtostderr

# ## HPO VP eta = 1.0

# python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_hpo_flow_nsf_vmp_flow_VPeta1.py \
#                           --config.path_mcmc_img $MCMC_5_ITEMS_DIR/lalme_floating_profiles_grid_eta_floating_1.000.png \
#                           --config.path_MCMC_samples $MCMC_5_ITEMS_DIR/lalme_az_w500_s50000_t50_sub100.nc \
#                           --config.wandb_project_name "LP-5items-HPO-VP-eta1" \
#                           --config.n_sweep_runs 10 \
#                           --workdir $WORK_DIR/5_items/vp_eta1/hpo \
#                           --log_dir $WORK_DIR/5_items/vp_eta1/hpo/log_dir \
#                           --alsologtostderr

# ## RUN AdditiveVMP eta only

python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_flow_nsf_vmp_flow_OPTIM_etaAdditiveVMP.py \
                          --config.path_mcmc_img $MCMC_5_ITEMS_DIR/lalme_floating_profiles_grid_eta_floating_1.000.png \
                          --config.path_MCMC_samples $MCMC_5_ITEMS_DIR/lalme_az_w500_s50000_t50_sub100.nc \
                          --config.wandb_project_name "LP-5items-AdditiveVMP-eta" \
                          --workdir $WORK_DIR/5_items/additive_vmp_eta_optim \
                          --log_dir $WORK_DIR/5_items/additive_vmp_eta_optim/log_dir \
                          --alsologtostderr

# ## RUN VMP eta only

python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_flow_nsf_vmp_flow_OPTIM_etaVMP.py \
                          --config.path_mcmc_img $MCMC_5_ITEMS_DIR/lalme_floating_profiles_grid_eta_floating_1.000.png \
                          --config.path_MCMC_samples $MCMC_5_ITEMS_DIR/lalme_az_w500_s50000_t50_sub100.nc \
                          --config.wandb_project_name "LP-5items-VMP-eta" \
                          --workdir $WORK_DIR/5_items/vmp_eta_optim \
                          --log_dir $WORK_DIR/5_items/vmp_eta_optim/log_dir \
                          --alsologtostderr



# ## RUN VP eta = 1.0

python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_flow_nsf_vmp_flow_OPTIM_VPeta1.py \
                          --config.path_mcmc_img $MCMC_5_ITEMS_DIR/lalme_floating_profiles_grid_eta_floating_1.000.png \
                          --config.path_MCMC_samples $MCMC_5_ITEMS_DIR/lalme_az_w500_s50000_t50_sub100.nc \
                          --config.wandb_project_name "LP-5items-VP-eta1" \
                          --workdir $WORK_DIR/5_items/vp_eta1_optim \
                          --log_dir $WORK_DIR/5_items/vp_eta1_optim/log_dir \
                          --alsologtostderr

## MCMC WD comparisons

VI_path_dict="{'VMP':'$WORK_DIR/5_items/vmp_eta_optim/lalme_az_run_54x9xlda_step10000.nc','Additive-VMP':'$WORK_DIR/5_items/additive_vmp_eta/hpo/lalme_az_run_9n9nsvcq_step160000.nc','VP':'$WORK_DIR/5_items/vp_eta1_optim/lalme_az_eta_1.000_bhmsxeif_step200000.nc'}"
eta_mcmc=1.0
python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_mcmc.py \
                        --workdir $WORK_DIR/5_items/mcmc/w500_s50_000_t50_sub100_eta_floating_$eta_mcmc \
                        --config.path_variational_samples $VI_path_dict \
                        --config.eta_profiles_floating "$eta_mcmc" \
                        --log_dir $WORK_DIR/5_items/mcmc/w500_s50_000_t50_sub100_eta_floating_$eta_mcmc/log_dir \
                        --alsologtostderr
