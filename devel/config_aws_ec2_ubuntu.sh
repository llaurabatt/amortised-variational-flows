#!/bin/bash

# This script configures an EC2 instance.
# We use the following instance cnfiguration:
#   EC2 type: p3.2xlarge
#   AMI name: Deep Learning AMI GPU Tensorflow 2.9.1 (Ubuntu 20.04) 20220628
#   AMI ID: ami-00f5894e701989c0e

# Send this script to the ec2 and execute it:
# rsync -v devel/config_aws_ec2_gpu.sh ec2-1:/home/ec2-user/.
# ssh ec2-1
# ./config_ec2_ubuntu.sh

# upgrade packages
sudo apt update
sudo apt -y upgrade
sudo apt -y autoremove

# Update python version
sudo apt install -y python3-venv

# Install some libraries
sudo apt install -y tree awscli

# Set-up git
git config --global user.name "Chris Carmona"
git config --global user.email carmona@stats.ox.ac.uk
git config --global alias.hist "log --pretty=format:'%C(yellow)[%ad]%C(reset) %C(green)[%h]%C(reset) | %C(red)%s %C(bold red){{%an}}%C(reset) %C(blue)%d%C(reset)' --graph --date=short"

# Clone ncad
git clone https://chriscarmona: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@github.com/chriscarmona/spatial-smi.git ~/spatial-smi
# Install ncad in a new virtual environment 
rm -rf ~/.virtualenvs/spatial-smi
python3 -m venv ~/.virtualenvs/spatial-smi
source ~/.virtualenvs/spatial-smi/bin/activate
pip install -U pip
pip install -U wheel setuptools
pip install -U "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r ./spatial-smi/requirements/requirements.txt
pip install -r ~/spatial-smi/requirements/requirements-sagemaker.txt