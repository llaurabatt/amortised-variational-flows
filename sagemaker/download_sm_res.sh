#!/bin/bash
set -e
set -x

# If this script is run in a EC2, then copy all png files locally doing:
# rsync -av --progress --include='*/' --include='*.png' --exclude='*' chrcarm-gpu:"~/spatial-smi-output" .

# Directory to save all outputs
WORK_DIR=$HOME/spatial-smi-output

mkdir -p $WORK_DIR
cd $WORK_DIR

export all_eta=('0.001' '0.250' '0.500' '0.750' '1.000')

# Download MCMC results on 8 items dataset
all_files=( \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-mcmc-eta0p001-2022-11-20-15-49-01-863/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-mcmc-eta0p250-2022-11-20-15-49-04-252/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-mcmc-eta0p500-2022-11-20-15-49-06-263/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-mcmc-eta0p750-2022-11-20-15-49-07-634/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-mcmc-eta1p000-2022-11-20-15-49-09-033/output/model.tar.gz' \
)
for i in {0..4}
do
  echo ${all_files[i]}
  out_dir=$WORK_DIR/8_items/mcmc/eta_floating_${all_eta[i]}
  mkdir -p $out_dir
  aws s3 cp ${all_files[i]} $out_dir/.
  tar -xf $out_dir/model.tar.gz -C $out_dir
done

# Download MF results on 8 items dataset
all_files=( \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-mf-eta0p001-2022-11-20-15-49-10-998/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-mf-eta0p250-2022-11-20-15-49-12-530/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-mf-eta0p500-2022-11-20-15-49-13-824/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-mf-eta0p750-2022-11-20-15-49-15-208/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-mf-eta1p000-2022-11-20-15-49-16-598/output/model.tar.gz' \
)
for i in {0..4}
do
  echo ${all_files[i]}
  out_dir=$WORK_DIR/8_items/mf/eta_floating_${all_eta[i]}
  mkdir -p $out_dir
  aws s3 cp ${all_files[i]} $out_dir/.
  tar -xf $out_dir/model.tar.gz -C $out_dir
done

# Download NSF results on 8 items dataset
all_files=( \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-nsf-eta0p001-2022-11-20-15-49-17-931/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-nsf-eta0p250-2022-11-20-15-49-19-396/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-nsf-eta0p500-2022-11-20-15-49-20-693/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-nsf-eta0p750-2022-11-20-15-49-22-007/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-nsf-eta1p000-2022-11-20-15-49-23-350/output/model.tar.gz' \
)
for i in {0..4}
do
  echo ${all_files[i]}
  out_dir=$WORK_DIR/8_items/nsf/eta_floating_${all_eta[i]}
  mkdir -p $out_dir
  aws s3 cp ${all_files[i]} $out_dir/.
  tar -xf $out_dir/model.tar.gz -C $out_dir
done

# Download VMP-flow results on 8 items dataset
all_files=( \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-nsf-vmp-flow-2022-11-20-15-49-39-391/output/model.tar.gz' \
)
for i in {0..0}
do
  echo ${all_files[i]}
  out_dir=$WORK_DIR/8_items/nsf/vmp_flow
  mkdir -p $out_dir
  aws s3 cp ${all_files[i]} $out_dir/.
  tar -xf $out_dir/model.tar.gz -C $out_dir
done

# Download Mean-Field results on all items dataset
all_files=( \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-mf-eta0p001-2022-11-20-15-49-25-008/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-mf-eta0p250-2022-11-20-15-49-26-398/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-mf-eta0p500-2022-11-20-15-49-27-742/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-mf-eta0p750-2022-11-20-15-49-29-084/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-mf-eta1p000-2022-11-20-15-49-30-458/output/model.tar.gz' \
)
for i in {0..4}
do
  echo ${all_files[i]}
  out_dir=$WORK_DIR/all_items/mf/eta_floating_${all_eta[i]}
  mkdir -p $out_dir
  aws s3 cp ${all_files[i]} $out_dir/.
  tar -xf $out_dir/model.tar.gz -C $out_dir
done

# Download NSF results on all items dataset
all_files=( \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-nsf-eta0p001-2022-11-20-15-49-32-275/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-nsf-eta0p250-2022-11-20-15-49-33-761/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-nsf-eta0p500-2022-11-20-15-49-35-132/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-nsf-eta0p750-2022-11-20-15-49-36-615/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-nsf-eta1p000-2022-11-20-15-49-37-905/output/model.tar.gz' \
)
for i in {0..4}
do
  echo ${all_files[i]}
  out_dir=$WORK_DIR/all_items/nsf/eta_floating_${all_eta[i]}
  mkdir -p $out_dir
  aws s3 cp ${all_files[i]} $out_dir/.
  tar -xf $out_dir/model.tar.gz -C $out_dir
done

# # Download VMP-flow results on all items dataset
# all_files=( \
#   's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-nsf-vmp-flow-2022-11-20-15-49-40-799/output/model.tar.gz' \
# )
# for i in {0..0}
# do
#   echo ${all_files[i]}
#   out_dir=$WORK_DIR/all_items/nsf/vmp_flow
#   mkdir -p $out_dir
#   aws s3 cp ${all_files[i]} $out_dir/.
#   tar -xf $out_dir/model.tar.gz -C $out_dir
# done
