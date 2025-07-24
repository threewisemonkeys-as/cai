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


# ---- variables ---------------------------------------------------------------

# AZ_CLIENT_ID="7020352e-2535-4532-99b8-18e99901af1b"
AZ_CLIENT_ID="7b009a27-5912-4556-8f17-0d3d707778ec"
AZ_RESOURCE_GROUP="debug-gym"
AZ_CLUSTER_NAME="debug-gym"


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

az login --identity --client-id "$AZ_CLIENT_ID"

az aks get-credentials --resource-group "$AZ_RESOURCE_GROUP" --name "$AZ_CLUSTER_NAME" --overwrite-existing


###############################################################################
log "2. Install Kubernetes CLI (kubectl)"
###############################################################################

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
mkdir -p ~/.local/bin
mv ./kubectl ~/.local/bin/kubectl
export PATH="$PATH:$HOME/.local/bin"
log "PATH is now $PATH"

sudo apt-get update 
sudo apt-get install -y unzip

curl -LO https://github.com/Azure/kubelogin/releases/latest/download/kubelogin-linux-amd64.zip
unzip kubelogin-linux-amd64.zip
sudo mv bin/linux_amd64/kubelogin /usr/local/bin/
rm -rf kubelogin-linux-amd64.zip bin/

export KUBECONFIG="$HOME/.kube/config"
log "KUBECONFIG set to $KUBECONFIG"

kubelogin convert-kubeconfig -l azurecli

kubectl get --raw='/readyz?verbose'
kubectl get --raw='/healthz?verbose'


# ---- check RBAC permissions ------------------------------------------------


errors=()

ok()   { echo "✅ $1"; }
bad()  { echo "❌ $1"; errors+=("$1"); }

can_i() {
  local verb="$1" res="$2"
  local out rc
  out="$(kubectl auth can-i "$verb" "$res" 2>&1 >/dev/null)"
  rc=$?       # 0=yes, 1=no, >1 real error (kubectl failure)

  if (( rc == 0 )); then
    ok  "$verb $res"
  elif (( rc == 1 )); then
    bad "$verb $res"
  else
    bad "$verb $res (kubectl error: $out)"
  fi
}

for v in get list watch create delete; do
  can_i "$v" pods
done

can_i create pods/exec
can_i create pods/attach
can_i get    pods/log
can_i create pods/portforward

echo
if ((${#errors[@]})); then
  echo "Missing permissions:"
  printf '  - %s\n' "${errors[@]}"
  exit 1
else
  echo "🎉 All required RBAC permissions are present."
fi

kubectl -n default create rolebinding default-sa-edit \
  --clusterrole=edit \
  --serviceaccount=default:default

kubectl auth can-i create pods/exec -n default \
  --as=system:serviceaccount:default:default 

###############################################################################
log "3. Install additional dependencies"
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
cd ..

