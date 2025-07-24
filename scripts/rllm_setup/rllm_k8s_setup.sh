#!/usr/bin/env bash
###############################################################################
# Automated setup for remote Docker daemon + Conda env + testing script
###############################################################################

set -euo pipefail

LOG_FILE="$HOME/setup_$(date +%F_%T).log"
exec > >(tee -a "$LOG_FILE") 2>&1   # mirror stdout/stderr to both console & log

# apt in non-interactive mode
export DEBIAN_FRONTEND=noninteractive


# ---- helper functions -------------------------------------------------------
log()   { printf "\n[%(%F %T)T] %s\n" -1 "$*"; }
abort() { log "ERROR: $*"; exit 1; }
trap 'abort "Script failed at line $LINENO (command: $BASH_COMMAND)"' ERR

# ---- variables --------------------------------------------------------------
CPU_REMOTE_HOST="arrival-similarly-kerry-recognize.trycloudflare.com"


###############################################################################
log "1. Install Cloudflared CLI and accept connection"
###############################################################################
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg |
  sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null

echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] \
https://pkg.cloudflare.com/cloudflared any main" |
  sudo tee /etc/apt/sources.list.d/cloudflared.list

sudo apt-get update -y
sudo apt-get install -y cloudflared

cloudflared access tcp --hostname $CPU_REMOTE_HOST

###############################################################################
log "2. Install Kubernetes CLI (kubectl)"
###############################################################################

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
mkdir -p ~/.local/bin
mv ./kubectl ~/.local/bin/kubectl
export PATH="$PATH:$HOME/.local/bin"
log "PATH is now $PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUBECONFIG_FILE="$SCRIPT_DIR/k3s.yaml"
if [[ ! -f "$KUBECONFIG_FILE" ]]; then
    echo "ERROR: $KUBECONFIG_FILE missing – copy it from the cluster first." >&2
    exit 1
fi
export KUBECONFIG="$KUBECONFIG_FILE"
log "KUBECONFIG set to $KUBECONFIG"

kubectl get --raw='/readyz?verbose'
kubectl get --raw='/healthz?verbose'
kubectl get nodes

###############################################################################
log "5. Install additional dependencies"
###############################################################################
git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
cd R2E-Gym
git checkout cai/rllm2
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
log "6. Run script"
###############################################################################

cd rllm/examples/swe
bash train_deepswe_8b_8h100.sh

log "All steps completed successfully!"
