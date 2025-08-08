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


# ---- Install and setup azure-cli --------------------------------------------

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az login --use-device-code

# ---- Install Kubernetes CLI (kubectl) ---------------------------------------

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
mkdir -p ~/.local/bin
mv ./kubectl ~/.local/bin/kubectl
export PATH="$PATH:$HOME/.local/bin"
log "PATH is now $PATH"

sudo apt-get update 
sudo apt-get install -y unzip

# ---- Install kubelogin ------------------------------------------------------

curl -LO https://github.com/Azure/kubelogin/releases/latest/download/kubelogin-linux-amd64.zip
unzip kubelogin-linux-amd64.zip
sudo mv bin/linux_amd64/kubelogin /usr/local/bin/
rm -rf kubelogin-linux-amd64.zip bin/


# ---- Configure ------------------------------------------------------------

export KUBECONFIG="$HOME/.kube/config"
log "KUBECONFIG set to $KUBECONFIG"

AZ_RESOURCE_GROUP="debug-gym"
AZ_CLUSTER_NAME="debug-gym"
az aks get-credentials --resource-group "$AZ_RESOURCE_GROUP" --name "$AZ_CLUSTER_NAME" --overwrite-existing

kubelogin convert-kubeconfig -l azurecli



# ---- Check permissions ------------------------------------------------------


kubectl get --raw='/readyz?verbose'
kubectl get --raw='/healthz?verbose'


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
