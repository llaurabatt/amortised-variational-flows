#!/bin/bash
set -e
set -x

# If this script is run in a EC2, then copy all png files locally doing:
# rsync -av --progress --include='*/' --include='*.png' --exclude='*' chrcarm-gpu:"~/spatial-smi-output" .

# Directory to save all outputs
WORK_DIR=$HOME/spatial-smi-output

mkdir -p $WORK_DIR
cd $WORK_DIR

all_eta=('0.001' '0.250' '0.500' '0.750' '1.000')

# Download MCMC results on 8 items dataset
all_files=( \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-mcmc-eta0p001-2022-09-25-21-53-11-448/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-mcmc-eta0p250-2022-09-25-21-53-18-812/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-mcmc-eta0p500-2022-09-25-21-53-25-633/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-mcmc-eta0p750-2022-09-25-21-53-32-757/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-mcmc-eta1p000-2022-09-25-21-53-37-108/output/model.tar.gz' \
)
for i in {0..4}
do
  echo ${all_files[i]}
  out_dir=$WORK_DIR/8_items/mcmc/eta_floating_${all_eta[i]}
  mkdir -p $out_dir
  aws s3 cp ${all_files[i]} $out_dir/.
  tar -xf $out_dir/model.tar.gz -C $out_dir
done

# Download NSF results on 8 items dataset
all_files=( \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-flow-nsf-like-mcmc-eta0p001-2022-09-25-21-53-44-882/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-flow-nsf-like-mcmc-eta0p250-2022-09-25-21-53-51-503/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-flow-nsf-like-mcmc-eta0p500-2022-09-25-21-53-58-144/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-flow-nsf-like-mcmc-eta0p750-2022-09-25-21-54-05-230/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-flow-nsf-like-mcmc-eta1p000-2022-09-25-21-54-14-321/output/model.tar.gz' \
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
  's3://sagemaker-us-east-1-467525936083/spatial-smi-8-items-flow-nsf-vmp-flow-2022-10-02-10-05-21-650/output/model.tar.gz' \
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
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-mf-eta0p001-2022-10-01-18-54-32-619/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-mf-eta0p250-2022-10-01-18-54-40-309/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-mf-eta0p500-2022-10-01-18-54-47-473/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-mf-eta0p750-2022-10-01-18-54-54-505/output/model.tar.gz' \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-all-items-flow-mf-eta1p000-2022-10-01-18-55-01-618/output/model.tar.gz' \
)
for i in {0..4}
do
  echo ${all_files[i]}
  out_dir=$WORK_DIR/all_items/mf/eta_floating_${all_eta[i]}
  mkdir -p $out_dir
  aws s3 cp ${all_files[i]} $out_dir/.
  tar -xf $out_dir/model.tar.gz -C $out_dir
done

# Download VMP-flow results on all items dataset
all_files=( \
  's3://sagemaker-us-east-1-467525936083/spatial-smi-flow-nsf-vmp-flow-2022-09-25-22-33-01-060/output/model.tar.gz' \
)
for i in {0..0}
do
  echo ${all_files[i]}
  out_dir=$WORK_DIR/all_items/nsf/vmp_flow
  mkdir -p $out_dir
  aws s3 cp ${all_files[i]} $out_dir/.
  tar -xf $out_dir/model.tar.gz -C $out_dir
done
