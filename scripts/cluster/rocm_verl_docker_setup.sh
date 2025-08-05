#!/bin/bash
set -e
set -x

VERL_ROCM_IMAGE="yushengsuthu/verl:verl-0.4.1_ubuntu-22.04_rocm6.3.4-numa-patch_vllm0.8.5_sglang0.4.6.post4"
CURRENT_ACR="debuggymacr"
ACR_LOGIN_SERVER="${CURRENT_ACR}.azurecr.io"
IMAGE_NAME="rocm-verl-v2"
IMAGE_TAG="latest"

# Pull the image from DockerHub
docker pull ${VERL_ROCM_IMAGE}

# Tag the image for ACR
docker tag ${VERL_ROCM_IMAGE} ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}

# Login to ACR
az acr login -n ${CURRENT_ACR}

# Push the tagged image to ACR
docker push ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}