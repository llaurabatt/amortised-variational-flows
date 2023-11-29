#!/bin/bash
set -e
set -x

# Assume we are located at main repo directory
REPO_DIR=$PWD

# Directory to save all outputs
WORK_DIR=$HOME/spatial-smi-output-integrated-allhps-40val-smallcondval-unifrhoeta


# Create output directory and install missing dependencies
mkdir -p $WORK_DIR
# pip install -Ur $REPO_DIR/requirements.txt

# all_eta=('0.001' '0.250' '0.500' '0.750' '1.000')
all_eta=('0.001' '1.000')
# ### 8 ITEMS ###

# ## Variational inference

# ## Single eta, Mean field (MFVI)
# for eta in "${all_eta[@]}"
# do
#   python3 $REPO_DIR/main.py --config $REPO_DIR/configs/8_items_flow_mf.py \
#                             --workdir $WORK_DIR/8_items/mf/eta_floating_$eta \
#                             --config.eta_profiles_floating $eta \
#                             --log_dir $WORK_DIR/8_items/mf/eta_floating_$eta/log_dir \
#                             --alsologtostderr
# done

# ## Variational inference
# ## Single eta, Neural Spline Flow
# for eta in "${all_eta[@]}"
# do
#   python3 $REPO_DIR/main.py --config $REPO_DIR/configs/8_items_flow_nsf.py \
#                             --workdir $WORK_DIR/8_items/nsf/eta_floating_$eta \
#                             --config.eta_profiles_floating $eta \
#                             --log_dir $WORK_DIR/8_items/nsf/eta_floating_$eta/log_dir \
#                             --alsologtostderr
# done

# ## Variational Meta-Posterior via VMP-flow
# ### Neural Spline Flow
# python3 $REPO_DIR/main.py --config $REPO_DIR/configs/8_items_flow_nsf_vmp_flow.py \
#                           --config.path_mcmc_img $HOME/my-spatial-smi-oldv/data/8_items_mcmc_floating.png \
#                           --workdir $WORK_DIR/8_items/nsf/vmp_flow \
#                           --log_dir $WORK_DIR/8_items/nsf/vmp_flow/log_dir \
#                           --alsologtostderr

# ## MCMC
## Single eta
# for eta in "${all_eta[@]}"
# do
#   python3 $REPO_DIR/main.py --config $REPO_DIR/configs/8_items_mcmc.py \
#                             --workdir $WORK_DIR/8_items/mcmc/eta_floating_$eta \
#                             --config.eta_profiles_floating $eta \
#                             --config.path_variational_samples $WORK_DIR/8_items/nsf/eta_floating_${eta}/posterior_sample_dict.npz \
#                             --log_dir $WORK_DIR/8_items/mcmc/eta_floating_$eta/log_dir \
#                             --alsologtostderr
# done

# ### ALL ITEMS ###

# ## Single eta, Mean field
# for eta in "${all_eta[@]}"
# do
#   python3 $REPO_DIR/main.py --config $REPO_DIR/configs/all_items_flow_mf.py \
#                             --workdir $WORK_DIR/all_items/mf/eta_floating_$eta \
#                             --config.eta_profiles_floating $eta \
#                             --log_dir $WORK_DIR/all_items/mf/eta_floating_$eta/log_dir \
#                             --alsologtostderr
# done

# ## Single eta, Neural Spline Flow
# for eta in "${all_eta[@]}"
# do
#   python3 $REPO_DIR/main.py --config $REPO_DIR/configs/all_items_flow_nsf.py \
#                             --workdir $WORK_DIR/all_items/nsf/eta_floating_$eta \
#                             --config.eta_profiles_floating $eta \
#                             --log_dir $WORK_DIR/all_items/nsf/eta_floating_$eta/log_dir \
#                             --alsologtostderr
# done

## Variational Meta-Posterior via VMP-flow
# Neural Spline Flow
python3 $REPO_DIR/main.py --config $REPO_DIR/configs/all_items_flow_nsf_vmp_flow.py \
                          --workdir $WORK_DIR/all_items/nsf/vmp_flow \
                          --log_dir $WORK_DIR/all_items/nsf/vmp_flow/log_dir \
                          --alsologtostderr
