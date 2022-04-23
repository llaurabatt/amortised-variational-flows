#!/bin/bash

# Send this script to the ec2
# rsync -v devel/config_aws_ec2_gpu.sh ec2-us-east-1:/home/ec2-user/.
# ssh ec2-us-east-1
# sudo yum upgrade
# ./config_aws_ec2_gpu.sh

# Update cuda version
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.1 /usr/local/cuda

# Update python version
sudo yum install python3.9

# Set-up git
git config --global user.name "Chris Carmona"
git config --global user.email carmona@stats.ox.ac.uk

# Clone and install modularbayes
mkdir -p smi/01_code
cd ~/smi/01_code
git clone https://chriscarmona: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@github.com/chriscarmona/modularbayes.git
pip3 install --upgrade pip

rm -rf ~/.virtualenvs/modularbayes
python3.8 -m venv ~/.virtualenvs/modularbayes
source ~/.virtualenvs/modularbayes/bin/activate
pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
pip install -e ./modularbayes
