#!/bin/bash
set -e
set -x


# Assume we are located at main repo directory
REPO_DIR=$PWD/linguistic_profiles

# Directory to save all outputs

N_ELBO_SAMPLES=("50ELBO" "3ELBO")
N_VAL_SAMPLES=("40val")

declare -A all_etas
all_etas["3ELBO"]='0.001 0.05 0.250 0.420 0.500 0.750 1.000' 
all_etas["50ELBO"]='0.001 0.05 0.250 0.500 0.610 0.750 1.000'

for n_val in "${N_VAL_SAMPLES[@]}"; do
    for n_elbo in "${N_ELBO_SAMPLES[@]}"; do

        WORK_DIR=$HOME/mount/vmp-output/lp-all/$n_elbo/$n_val

        # Create output directory if it does not exist
        mkdir -p $WORK_DIR

        # # VMP and hypeprameter optimisation
        # python3 $REPO_DIR/main.py --config $REPO_DIR/configs/all_items_flow_nsf_vmp_flow_${n_elbo}_${n_val}.py \
        #                           --workdir $WORK_DIR \
        #                           --log_dir $WORK_DIR/log_dir \
        #                           --alsologtostderr

        # # plot hyperparameter convergence
        # python3 $REPO_DIR/hp_convergence_plots_NEW.py --path=$WORK_DIR

        # VP at different etas on optimised prior hyperparameters
        # etas="${all_etas[$n_elbo]}"
        # IFS=' ' read -r -a eta_array <<< "$etas"
        # for eta in "${eta_array[@]}"; do
        #   echo "Processing eta: $eta"
        #   python3 $REPO_DIR/main.py --config $REPO_DIR/configs/all_items_flow_nsf_vp_flow_${n_elbo}_${n_val}.py \
        #                             --workdir $WORK_DIR/VP_eta_$eta \
        #                             --config.eta_fixed $eta \
        #                             --config.optim_prior_hparams_dir_fixed_eta $WORK_DIR \
        #                             --log_dir $WORK_DIR/VP_eta_$eta/log_dir \
        #                             --alsologtostderr
        # done

        # Amortisation gap
        WORK_DIR_TEMP="/home/llaurabat/mount/old-disk/home/llaurabat/spatial-smi-output-integrated-allhps-40val-smallcondval/all_items/nsf/vmp_flow"
        python3 $REPO_DIR/amortisation_gap.py --config $REPO_DIR/configs/amortisation_plot_nsf_vmp_flow_${n_elbo}_${n_val}.py \
                                --config.workdir_VMP $WORK_DIR \
                                --config.optim_prior_hparams_dir $WORK_DIR \
                                --workdir $WORK_DIR \
                                --alsologtostderr
    done
done
