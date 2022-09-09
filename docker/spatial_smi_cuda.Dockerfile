FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

LABEL maintainer="carmona@stats.ox.ac.uk"

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Update packages
RUN apt-get update
RUN apt-get upgrade -y
# RUN apt-get install -y nvidia-cuda-toolkit

# Install some libraries
RUN apt-get install -y git wget unzip curl awscli tree

# Install python3
RUN apt-get install -y python3-pip
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip --no-cache-dir install -U pip
# Install useful python modules
RUN pip install -U wheel setuptools pylint yapf

# Add a non-root user
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  #
  # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# For Sagemaker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN pip install -U sagemaker-training

# Install Jax with CUDA support
RUN pip --no-cache-dir install -U "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Dependencies
RUN git clone https://chriscarmona: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@github.com/chriscarmona/spatial-smi.git
RUN pip install -r ~/spatial-smi/requirements/requirements.txt
RUN chmod +x ~/spatial-smi/examples/run.sh

CMD ["/bin/bash"]
