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


# AZ_CLIENT_ID="7020352e-2535-4532-99b8-18e99901af1b"
AZ_CLIENT_ID="7b009a27-5912-4556-8f17-0d3d707778ec"
az login --identity --client-id "$AZ_CLIENT_ID"

export KUBECONFIG="$HOME/.kube/config"
log "KUBECONFIG set to $KUBECONFIG"

AZ_RESOURCE_GROUP="debug-gym"
AZ_CLUSTER_NAME="debug-gym"
az aks get-credentials --resource-group "$AZ_RESOURCE_GROUP" --name "$AZ_CLUSTER_NAME" --overwrite-existing

kubelogin convert-kubeconfig -l azurecli

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
