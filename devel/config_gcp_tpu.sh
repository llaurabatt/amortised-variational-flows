#!/bin/bash

# (From https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)

### ON LOCAL MACHINE ###

# # # Configure the gcloud command
# gcloud config set account christianu7@gmail.com
# gcloud config set project modularbayes
# # Enable the Cloud TPU API
# gcloud services enable tpu.googleapis.com
# gcloud beta services identity create --service tpu.googleapis.com

# # Characteristics of the TPU VM
# export VM=gcp-tpu

# export GCP_ZONE=europe-west4-a
# export TPU_TYPE=v3-8

# export GCP_ZONE=us-central1-f
# export TPU_TYPE=v2-8

# export GCP_ZONE=us-central1-a
# export TPU_TYPE=v3-8

# # Set zone as config variable
# gcloud config set compute/zone ${GCP_ZONE}

# # Create a Cloud TPU VM
# gcloud alpha compute tpus tpu-vm create $VM \
#     --zone $GCP_ZONE \
#     --accelerator-type $TPU_TYPE \
#     --version v2-alpha
# #    --preemptible

# # Connect to your Cloud TPU VM
# gcloud alpha compute tpus tpu-vm ssh $VM --zone $GCP_ZONE

# # You can ssh connect without the gcloud command:
# #   1. add a ssh key to the project metadata, follow:
# #     https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys
# #   2. Get the External IP for the TPU VM to which you want to connect
# #     gcloud alpha compute tpus tpu-vm describe $VM --zone=$GCP_ZONE
# #   3. Configure your ~/.ssh/config file, adding something similar to this:
# #   The HostName field corresponds to the external IP address of the VM
# #
# #     Host gcp-tpu
# #       AddKeysToAgent yes
# #       UseKeychain yes
# #       HostName 34.132.197.251
# #       IdentityFile ~/.ssh/google_compute_engine
# #       User carmona

# # Now you can connect using VSCODE or command line
# # ssh gcp-tpu

# Send this script to the TPU VM
# rsync -v devel/config_gcp_tpu.sh $VM:/home/carmona/.
# ssh $VM
# ./config_gcp_tpu.sh

### ON TPU MACHINE ###

# upgrade VM packages
sudo apt-get update
sudo apt-get -y upgrade
sudo apt -y upgrade
sudo apt -y autoremove

# Create virtual enviromnent
sudo apt -y install python3.8-venv
rm -rf ~/.virtualenvs/modularbayes
python3 -m venv ~/.virtualenvs/modularbayes
source ~/.virtualenvs/modularbayes/bin/activate
pip install --upgrade pip
pip install --upgrade wheel
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Set-up git
git config --global user.name "Chris Carmona"
git config --global user.email carmona@stats.ox.ac.uk

# Clone and install modularbayes
mkdir -p smi/01_code
cd smi/01_code
git clone https://chriscarmona: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@github.com/chriscarmona/modularbayes.git
pip install -e ./modularbayes
# python3 examples/lalme/main.py --workdir=examples/lalme/output --config=examples/lalme/configs/iaf.py
# python3 examples/lalme/main.py --workdir=examples/lalme/output --config=examples/lalme/configs/meanfield.py
