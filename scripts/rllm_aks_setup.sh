#!/usr/bin/env bash

set -euo pipefail

LOG_FILE="$HOME/setup_$(date +%F_%T).log"
exec > >(tee -a "$LOG_FILE") 2>&1   # mirror stdout/stderr to both console & log

# apt in non-interactive mode
export DEBIAN_FRONTEND=noninteractive

# ---- helper functions -------------------------------------------------------
log()   { printf "\n[%(%F %T)T] %s\n" -1 "$*"; }
abort() { log "ERROR: $*"; exit 1; }
trap 'abort "Script failed at line $LINENO (command: $BASH_COMMAND)"' ERR



###############################################################################
log "1. Install and setup azure-cli"
###############################################################################

sudo apt-get update
sudo apt-get -y install apt-transport-https ca-certificates curl gnupg lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -sLS https://packages.microsoft.com/keys/microsoft.asc |
  gpg --dearmor | sudo tee /etc/apt/keyrings/microsoft.gpg > /dev/null
sudo chmod go+r /etc/apt/keyrings/microsoft.gpg

AZ_DIST=$(lsb_release -cs)
echo "Types: deb
URIs: https://packages.microsoft.com/repos/azure-cli/
Suites: ${AZ_DIST}
Components: main
Architectures: $(dpkg --print-architecture)
Signed-by: /etc/apt/keyrings/microsoft.gpg" | sudo tee /etc/apt/sources.list.d/azure-cli.sources

sudo apt-get update
sudo apt-get -y install azure-cli 

az login --identity --client-id 7020352e-2535-4532-99b8-18e99901af1b

az aks get-credentials --resource-group debug-gym --name debug-gym --overwrite-existing


###############################################################################
log "3. Install Kubernetes CLI (kubectl)"
###############################################################################

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
mkdir -p ~/.local/bin
mv ./kubectl ~/.local/bin/kubectl
export PATH="$PATH:$HOME/.local/bin"
log "PATH is now $PATH"


curl -LO https://github.com/Azure/kubelogin/releases/latest/download/kubelogin-linux-amd64.zip
unzip kubelogin-linux-amd64.zip
sudo mv bin/linux_amd64/kubelogin /usr/local/bin/
rm -rf kubelogin-linux-amd64.zip bin/


export KUBECONFIG="$HOME/.kube/config"
log "KUBECONFIG set to $KUBECONFIG"

kubelogin convert-kubeconfig -l azurecli

kubectl get --raw='/readyz?verbose'
kubectl get --raw='/healthz?verbose'
kubectl get nodes

###############################################################################
log "4. Install additional dependencies"
###############################################################################
git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
cd R2E-Gym
git checkout rllm2
pip install -e .
cd ..
git clone --recurse-submodules https://github.com/threewisemonkeys-as/rllm.git
cd rllm
git checkout rllm2
pip install -e ./verl[vllm]
pip install -e .
python examples/swe/prepare_swe_data.py
cd ..

###############################################################################
log "5. Run script"
###############################################################################

cd rllm/examples/swe
bash train_deepswe_8b_8h100.sh

log "All steps completed successfully!"

