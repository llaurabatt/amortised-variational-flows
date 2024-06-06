#!/bin/bash
set -e
set -x

# Assume we are located at main repo directory
REPO_DIR=$PWD

# Directory to save all outputs
# WORK_DIR=$HOME/spatial-smi-output-original
# WORK_DIR=$HOME/HPOPT200k-spatial-smi-output-VPeta1-withintegratedsmallcondval # it is with NOKERNEL
# WORK_DIR=$HOME/HPOPT200k-NEW-spatial-smi-output-integrated-ONLYeta-smallcondval-SALVACHECKPOINT # it is with NOKERNEL
# WORK_DIR=$HOME/HPOPT200k-spatial-smi-output-integrated-ONLYeta-smallcondval-SALVACHECKPOINT # it is with NOKERNEL
# WORK_DIR=$HOME/HPOPT200k-spatial-smi-output-VPeta1-withintegratedsmallcondval-SALVACHECKPOINT # it is with NOKERNEL
# WORK_DIR=$HOME/spatial-smi-output-VPeta1-withintegratedsmallcondval-WITHBESTOPTIM
# WORK_DIR=$HOME/spatial-smi-output-integrated-ONLYeta-smallcondval-WITHBESTOPTIM
WORK_DIR=$HOME/spatial-smi-output-integrated-allhps-40val-smallcondval-MOREELBOSAMPLES
# WORK_DIR=$HOME/HPOPT-spatial-smi-output-integrated-allhps-40val-smallcondval

# Create output directory and install missing dependencies
mkdir -p $WORK_DIR
# pip install -Ur $REPO_DIR/requirements.txt

all_eta=('0.05' '0.500' '0.750' '1.000' '0.001' '0.610') #('0.001' '0.250' '0.420' '0.500' '0.750' '1.000')
# all_eta=('1.000')
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
#

# --config.path_MCMC_samples $HOME/spatial-smi-output-original/5_items/mcmc/eta_floating_w500_s15_000_t10_sub100_1.000/lalme_az_10_000s_thinning10.nc \
# python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_hpo_flow_nsf_vmp_flow.py \
#                           --config.path_mcmc_img $HOME/my-spatial-smi-oldv/data/5_items_mcmc_floating_eta1.000.png \
#                           --config.path_MCMC_samples $HOME/spatial-smi-output-original/5_items/mcmc/eta_floating_w500_s50_000_t50_sub100_1.000/lalme_az_10_000s_thinning50.nc \
#                           --workdir $WORK_DIR/5_items/nsf/vmp_flow \
#                           --log_dir $WORK_DIR/5_items/nsf/vmp_flow/log_dir \
#                           --alsologtostderr
# python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_flow_nsf_vmp_flow.py \
#                           --config.path_mcmc_img $HOME/my-spatial-smi-oldv/data/5_items_mcmc_floating_eta1.000.png \
#                           --config.path_MCMC_samples $HOME/spatial-smi-output-original/5_items/mcmc/eta_floating_w500_s15_000_t10_sub100_1.000/lalme_az_10_000s_thinning10.nc \
#                           --workdir $WORK_DIR/5_items/nsf/vmp_flow \
#                           --log_dir $WORK_DIR/5_items/nsf/vmp_flow/log_dir \
#                           --alsologtostderr
# ## MCMC
## Single eta
# multiple VI paths

# VI_path_dict="{'VMP':'/home/llaurabat/spatial-smi-output-integrated-ONLYeta/8_items/nsf/vmp_flow/lalme_az_eta_1.000.nc','ADDITIVE-VMP':'/home/llaurabat/spatial-smi-output-original-smallcondval-LASTTRY/8_items/nsf/vmp_flow/lalme_az_eta_1.000.nc','VP':'/home/llaurabat/spatial-smi-output-VPeta1-withintegratedsmallcondval/8_items/nsf/vmp_flow/lalme_az_eta_1.000.nc'}"
# for eta in "${all_eta[@]}"
# do
#   python3 $REPO_DIR/main.py --config $REPO_DIR/configs/8_items_mcmc.py \
#                             --workdir $WORK_DIR/8_items/mcmc/eta_floating_w500_s15_000_t10_sub100_$eta \
#                             --config.eta_profiles_floating $eta \
#                             --config.path_variational_samples $VI_path_dict\
#                             --log_dir $WORK_DIR/8_items/mcmc/eta_floating_w500_s15_000_t10_sub100_$eta/log_dir \
#                             --alsologtostderr
# done

# VI_path_dict="{'VMP':'/home/llaurabat/spatial-smi-output-integrated-ONLYeta/5_items/nsf/vmp_flow/lalme_az_eta_1.000.nc','ADDITIVE-VMP':'/home/llaurabat/spatial-smi-output-original-smallcondval-LASTTRY/5_items/nsf/vmp_flow/lalme_az_eta_1.000.nc','VP':'/home/llaurabat/spatial-smi-output-VPeta1-withintegratedsmallcondval/5_items/nsf/vmp_flow/lalme_az_eta_1.000.nc'}"
# VI_path_dict="{'VMP':'/home/llaurabat/spatial-smi-output-integrated-ONLYeta/5_items/nsf/vmp_flow/lalme_az_eta_1.000.nc','ADDITIVE-VMP':'/home/llaurabat/spatial-smi-output-original-smallcondval-LASTTRY/5_items/nsf/vmp_flow/lalme_az_eta_1.000.nc','VP':'/home/llaurabat/spatial-smi-output-VPeta1-withintegratedsmallcondval-WITHBESTOPTIM/5_items/nsf/vmp_flow/lalme_az_eta_1.000.nc'}"

# for eta in "${all_eta[@]}"
# do
#   python3 $REPO_DIR/main.py --config $REPO_DIR/configs/5_items_mcmc.py \
#                             --workdir $WORK_DIR/5_items/mcmc/eta_floating_w500_s50_000_t50_sub100_$eta \
#                             --config.eta_profiles_floating $eta \
#                             --config.path_variational_samples $VI_path_dict\
#                             --log_dir $WORK_DIR/5_items/mcmc/eta_floating_w500_s50_000_t50_sub100_$eta/log_dir \
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
# # # Neural Spline Flow
for eta in "${all_eta[@]}"
do
# export JAX_ENABLE_X64=1
python3 $REPO_DIR/main.py --config $REPO_DIR/configs/all_items_flow_nsf_vmp_flow.py \
                          --workdir $WORK_DIR/all_items/nsf/vmp_flow/VP_eta_$eta \
                          --config.eta_fixed $eta \
                          --log_dir $WORK_DIR/all_items/nsf/vmp_flow/VP_eta_$eta/log_dir \
                          --alsologtostderr
done

# python3 $REPO_DIR/main.py --config $REPO_DIR/configs/all_items_flow_nsf_vmp_flow.py \
#                           --workdir $WORK_DIR/all_items/nsf/vmp_flow \
#                           --log_dir $WORK_DIR/all_items/nsf/vmp_flow/log_dir \
#                           --alsologtostderr

# python3 $REPO_DIR/main.py --config $REPO_DIR/configs/all_items_hpo_flow_nsf_vmp_flow.py \
#                           --workdir $WORK_DIR/all_items/nsf/vmp_flow \
#                           --log_dir $WORK_DIR/all_items/nsf/vmp_flow/log_dir \
# #                           --alsologtostderr

# python3 $REPO_DIR/amortisation_gap.py --config $REPO_DIR/configs/amortisation_plot_nsf_vmp_flow.py \
#                           --workdir $WORK_DIR/all_items/nsf/vmp_flow \
#                           --alsologtostderr