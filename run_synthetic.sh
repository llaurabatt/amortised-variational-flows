#!/bin/bash
set -e
set -x

# Assume we are located at the SMI directory
WORK_DIR_MAIN=$PWD

# Directory to save all outputs
WORK_DIR_VP_true_small=$HOME/mount/vmp-output-time/synthetic/vp-true-ng10-nobs8
WORK_DIR_mcmc_true_small=$HOME/mount/vmp-output-time/synthetic/mcmc-ng10-nobs8
WORK_DIR_small=$HOME/mount/vmp-output-time/synthetic/vmp-ng10-nobs8

WORK_DIR_true_large=$HOME/mount/vmp-output-time/synthetic/vp-true-ng50-nobs50
WORK_DIR_mcmc_true_large=$HOME/mount/vmp-output-time/synthetic/mcmc-ng50-nobs50
WORK_DIR_large=$HOME/mount/vmp-output-time/synthetic/vmp-ng50-nobs50





## SYNTHETIC DATA Variational Meta-Posterior INTEGRATED via VMP-flow with beta hyperparameter tuning
## Small dataset

# # # VP at true hyperparameter values
mkdir -p $WORK_DIR_VP_true_small
python3 $WORK_DIR_MAIN/synthetic/main.py --config=$WORK_DIR_MAIN/synthetic/configs/flow_nsf_vp_flow_true_small_TIME.py \
                                               --workdir=$WORK_DIR_VP_true_small \
                                               --log_dir $WORK_DIR_VP_true_small/log_dir \
                                              --alsologtostderr

# # MCMC at true hyperparameter values

# mkdir -p $WORK_DIR_mcmc_true_small
# python3 $WORK_DIR_MAIN/synthetic/main.py --config=$WORK_DIR_MAIN/synthetic/configs/mcmc_small.py \
#                                                --workdir=$WORK_DIR_mcmc_true_small \
#                                                --log_dir $WORK_DIR_mcmc_true_small/log_dir \
#                                               --alsologtostderr


# # # VMP + comparison plots
mkdir -p $WORK_DIR_small
python3 $WORK_DIR_MAIN/synthetic/main.py --config=$WORK_DIR_MAIN/synthetic/configs/flow_nsf_vmp_flow_small_TIME.py \
                                               --workdir=$WORK_DIR_small \
                                               --log_dir $WORK_DIR_small/log_dir \
                                              --alsologtostderr

# # # Plot hyperparameter tuning results

# # python3 $WORK_DIR_MAIN/synthetic/hp_convergence_plots.py --path_results=$WORK_DIR_small

# # ## Large dataset

# # # VP at true hyperparameter values
mkdir -p $WORK_DIR_true_large
python3 $WORK_DIR_MAIN/synthetic/main.py --config=$WORK_DIR_MAIN/synthetic/configs/flow_nsf_vp_flow_true_large_TIME.py \
                                               --workdir=$WORK_DIR_true_large \
                                               --log_dir $WORK_DIR_true_large/log_dir \
                                              --alsologtostderr

# # # MCMC at true hyperparameter values

# mkdir -p $WORK_DIR_mcmc_true_large
# python3 $WORK_DIR_MAIN/synthetic/main.py --config=$WORK_DIR_MAIN/synthetic/configs/mcmc_large.py \
#                                                --workdir=$WORK_DIR_mcmc_true_large \
#                                                --log_dir $WORK_DIR_mcmc_true_large/log_dir \
#                                               --alsologtostderr

# # # VMP + comparison plots
mkdir -p $WORK_DIR_large
python3 $WORK_DIR_MAIN/synthetic/main.py --config=$WORK_DIR_MAIN/synthetic/configs/flow_nsf_vmp_flow_large_TIME.py \
                                               --workdir=$WORK_DIR_large \
                                               --log_dir $WORK_DIR_large/log_dir \
                                              --alsologtostderr 

# # # Plot hyperparameter tuning results
# # python3 $WORK_DIR_MAIN/synthetic/hp_convergence_plots.py --path_results=$WORK_DIR_large
