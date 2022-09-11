#!/bin/bash
set -e
set -x

# IMG_NAME="jax-cuda"
IMG_NAME="jax-tf"
# IMG_NAME="sagemaker-jax"

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-east-1 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${IMG_NAME}"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --region "${region}" --repository-names "${IMG_NAME}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --region "${region}" --repository-name "${IMG_NAME}" > /dev/null
fi

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build  -t ${IMG_NAME} -f docker/$IMG_NAME.Dockerfile .
docker tag ${IMG_NAME} ${fullname}

# Authenticate Docker client to the Amazon ECR registry
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${fullname}

docker push ${fullname}