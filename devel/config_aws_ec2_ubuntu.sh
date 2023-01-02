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

# Create virtual environment 
rm -rf ~/.virtualenvs/spatial-smi
python3 -m venv ~/.virtualenvs/spatial-smi
source ~/.virtualenvs/spatial-smi/bin/activate
pip install -U pip
pip install -U wheel setuptools
pip install -U "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Clone repo
git clone https://chriscarmona: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@github.com/chriscarmona/spatial-smi.git ~/spatial-smi
cd ~/spatial-smi
pip install -r requirements.txt
pip install -r devel/requirements.txt
pip install -r sagemaker/requirements.txt
