#!/bin/bash
set -e
set -x


# Assume we are located at main repo directory
REPO_DIR=$PWD/linguistic_profiles

# Directory to save all outputs

WORK_DIR=$HOME/mount/vmp-output/lp-all/3ELBO/40val

# Create output directory and install missing dependencies
mkdir -p $WORK_DIR



# ### ALL ITEMS ###

## Variational Meta-Posterior via VMP-flow

# python3 $REPO_DIR/main.py --config $REPO_DIR/configs/all_items_flow_nsf_vmp_flow_3ELBO.py \
#                           --workdir $WORK_DIR \
#                           --log_dir $WORK_DIR/log_dir \
#                           --alsologtostderr

# plot hyperparameter convergence
# python3 $REPO_DIR/hp_convergence_plots_NEW.py --path=$WORK_DIR

# VP at different etas on optimised prior hyperparameters

all_eta=('0.250' '0.05' '0.500' '0.750' '1.000' '0.420' '0.001')
for eta in "${all_eta[@]}"
do
python3 $REPO_DIR/main.py --config $REPO_DIR/configs/all_items_flow_nsf_vp_flow_3ELBO.py \
                          --workdir $WORK_DIR/VP_eta_$eta \
                          --config.eta_fixed $eta \
                          --log_dir $WORK_DIR/VP_eta_$eta/log_dir \
                          --alsologtostderr
done


# # Amortisation gap
# python3 $REPO_DIR/amortisation_gap.py --config $REPO_DIR/configs/amortisation_plot_nsf_vmp_flow.py \
#                           --workdir $WORK_DIR/all_items/nsf/vmp_flow \
#                           --alsologtostderr
