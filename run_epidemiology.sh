#!/bin/bash
set -e
set -x

# Assume we are located at the SMI directory
WORK_DIR_MAIN=$PWD

# Directory to save all outputs
WORK_DIR_mcmc=$HOME/mount/vmp-output/epidemiology/mcmc
CONFIG_DIR_mcmc=$HOME/my-spatial-smi-oldv/epidemiology/configs/mcmc 
WORK_DIR_bayes=$HOME/mount/vmp-output/epidemiology/vmp-bayes
WORK_DIR_smi=$HOME/mount/vmp-output/epidemiology/vmp-smi

# MCMC
# for CONFIG_FILE in "$CONFIG_DIR_mcmc"/*.py; do
#   CONFIG_BASENAME=$(basename "$CONFIG_FILE" .py)
#   echo "Running $PYTHON_SCRIPT with configuration $CONFIG_BASENAME"
#   python3 $WORK_DIR_MAIN/epidemiology/main.py --config=$CONFIG_FILE \
#                                                  --workdir=$WORK_DIR_mcmc/$CONFIG_BASENAME \
#                                                  --log_dir $WORK_DIR_mcmc/$CONFIG_BASENAME/log_dir \
#                                                  --alsologtostderr
# done

# # # ## LOOCV Variational Meta-Posterior INTEGRATED via VMP-flow with beta hyperparameter tuning
hpv_no_obs=13

# LOOCV BAYES

# set +e
# for ((i=0;i<hpv_no_obs;i++))

# do
#     mask_Y=$(python -c "print('(' + ', '.join(['0' if j==$i else '1' for j in range($hpv_no_obs)]) + ')')") 
#     python3 $WORK_DIR_MAIN/epidemiology/main.py --config=$WORK_DIR_MAIN/epidemiology/configs/flow_nsf_vmp_flow_bayes_loocv.py \
#                                                 --workdir="$WORK_DIR_bayes/loocv_y/dropped_$(($i))"\
#                                                 --config.mask_Y "$mask_Y" \
#                                                 --log_dir "$WORK_DIR_bayes/loocv_y/dropped_$(($i))" \
#                                                 --alsologtostderr 
#     status=$?
#     if [ $status -ne 0 ]; then
#         echo "Error detected in iteration $i, continuing..."
#         # Optionally, add any error handling here
#     else
#         echo "Iteration $i completed successfully."
#     fi
# done
# set -e

# set +e
# for ((i=0;i<hpv_no_obs;i++))

# do
#     mask_Z=$(python -c "print('(' + ', '.join(['0' if j==$i else '1' for j in range($hpv_no_obs)]) + ')')") 
#     python3 $WORK_DIR_MAIN/epidemiology/main.py --config=$WORK_DIR_MAIN/epidemiology/configs/flow_nsf_vmp_flow_bayes_loocv.py \
#                                                 --workdir="$WORK_DIR_bayes/loocv_z/dropped_$(($i))"\
#                                                 --config.mask_Z "$mask_Z" \
#                                                 --log_dir "$WORK_DIR_bayes/loocv_z/dropped_$(($i))" \
#                                                 --alsologtostderr 
#     status=$?
#     if [ $status -ne 0 ]; then
#         echo "Error detected in iteration $i, continuing..."
#         # Optionally, add any error handling here
#     else
#         echo "Iteration $i completed successfully."
#     fi
# done
# set -e

# # LOOCV SMI

# set +e
# for ((i=0;i<hpv_no_obs;i++))

# do
#     mask_Y=$(python -c "print('(' + ', '.join(['0' if j==$i else '1' for j in range($hpv_no_obs)]) + ')')") 
#     python3 $WORK_DIR_MAIN/epidemiology/main.py --config=$WORK_DIR_MAIN/epidemiology/configs/flow_nsf_vmp_flow_smi_loocv.py \
#                                                 --workdir="$WORK_DIR_smi/loocv_y/dropped_$(($i))"\
#                                                 --config.mask_Y "$mask_Y" \
#                                                 --log_dir "$WORK_DIR_smi/loocv_y/dropped_$(($i))" \
#                                                 --alsologtostderr 
#     status=$?
#     if [ $status -ne 0 ]; then
#         echo "Error detected in iteration $i, continuing..."
#         # Optionally, add any error handling here
#     else
#         echo "Iteration $i completed successfully."
#     fi
# done
# set -e

# set +e
# for ((i=0;i<hpv_no_obs;i++))

# do
#     mask_Z=$(python -c "print('(' + ', '.join(['0' if j==$i else '1' for j in range($hpv_no_obs)]) + ')')") 
#     python3 $WORK_DIR_MAIN/epidemiology/main.py --config=$WORK_DIR_MAIN/epidemiology/configs/flow_nsf_vmp_flow_smi_loocv.py \
#                                                 --workdir="$WORK_DIR_smi/loocv_z/dropped_$(($i))"\
#                                                 --config.mask_Z "$mask_Z" \
#                                                 --log_dir "$WORK_DIR_smi/loocv_z/dropped_$(($i))" \
#                                                 --alsologtostderr 
#     status=$?
#     if [ $status -ne 0 ]; then
#         echo "Error detected in iteration $i, continuing..."
#         # Optionally, add any error handling here
#     else
#         echo "Iteration $i completed successfully."
#     fi
# done
# set -e



# Full VMP Bayes with hyperparameter tuning

python3 $WORK_DIR_MAIN/epidemiology/main.py --config=$WORK_DIR_MAIN/epidemiology/configs/flow_nsf_vmp_flow_bayes.py \
                                               --workdir=$WORK_DIR_bayes \
                                               --workdir_mcmc $WORK_DIR_mcmc \
                                               --log_dir $WORK_DIR_bayes/log_dir \
                                              --alsologtostderr 

# Full VMP SMI with hyperparameter tuning

# python3 $WORK_DIR_MAIN/epidemiology/main.py --config=$WORK_DIR_MAIN/epidemiology/configs/flow_nsf_vmp_flow_smi.py \
#                                                --workdir=$WORK_DIR_smi \
#                                                --workdir_mcmc $WORK_DIR_mcmc \
#                                                --log_dir $WORK_DIR_smi/log_dir \
#                                               --alsologtostderr 

# Plot hyperparameter tuning results

python3 $WORK_DIR_MAIN/epidemiology/hp_convergence_new.py --path_smi=$WORK_DIR_smi \
                                                   --path_bayes=$WORK_DIR_bayes

