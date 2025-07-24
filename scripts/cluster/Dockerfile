# syntax=docker/dockerfile:1
FROM singularitybase.azurecr.io/base/job/pytorch/acpt-torch2.7.0-py3.10-cuda12.6-ubuntu22.04:20250609T145333123


ARG DEBIAN_FRONTEND=noninteractive

# ---------- OS deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https ca-certificates curl gnupg lsb-release unzip \
    python3 python3-pip python3-venv git build-essential \
 && rm -rf /var/lib/apt/lists/*

# ---------- Azure CLI repo & install ----------

WORKDIR /work

RUN mkdir -p /etc/apt/keyrings \
 && curl -sLS https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --dearmor > /etc/apt/keyrings/microsoft.gpg \
 && chmod go+r /etc/apt/keyrings/microsoft.gpg \
 && AZ_DIST="$(lsb_release -cs)" \
 && printf "Types: deb\nURIs: https://packages.microsoft.com/repos/azure-cli/\nSuites: %s\nComponents: main\nArchitectures: %s\nSigned-by: /etc/apt/keyrings/microsoft.gpg\n" \
        "$AZ_DIST" "$(dpkg --print-architecture)" \
    > /etc/apt/sources.list.d/azure-cli.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends azure-cli \
 && rm -rf /var/lib/apt/lists/*

# ---------- kubectl ----------

WORKDIR /work

RUN KUBECTL_VER="$(curl -sL https://dl.k8s.io/release/stable.txt)" \
 && curl -LO "https://dl.k8s.io/release/${KUBECTL_VER}/bin/linux/amd64/kubectl" \
 && install -m 0755 kubectl /usr/local/bin/kubectl \
 && rm kubectl

# ---------- kubelogin ----------

WORKDIR /work

RUN curl -LO https://github.com/Azure/kubelogin/releases/latest/download/kubelogin-linux-amd64.zip \
 && unzip kubelogin-linux-amd64.zip \
 && install -m 0755 bin/linux_amd64/kubelogin /usr/local/bin/kubelogin \
 && rm -rf kubelogin-linux-amd64.zip bin

# ---------- Python deps / repos ----------

WORKDIR /work

# R2E-Gym
RUN git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
WORKDIR /work/R2E-Gym
RUN git checkout rllm2 \
 && pip install --no-cache-dir -e .

WORKDIR /work

# rllm (with submodules)
RUN git clone --recurse-submodules https://github.com/threewisemonkeys-as/rllm.git
WORKDIR /work/rllm
RUN git checkout rllm2 \
 && pip install --no-cache-dir -e ./verl[vllm] \
 && pip install --no-cache-dir -e .

# ---------- PATH tweak (optional) ----------
ENV PATH="${PATH}:/root/.local/bin"