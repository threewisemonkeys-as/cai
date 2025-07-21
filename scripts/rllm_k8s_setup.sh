#!/usr/bin/env bash
###############################################################################
# Automated setup for remote Docker daemon + Conda env + testing script
###############################################################################

set -euo pipefail

LOG_FILE="$HOME/setup_$(date +%F_%T).log"
exec > >(tee -a "$LOG_FILE") 2>&1   # mirror stdout/stderr to both console & log

# ---- helper functions -------------------------------------------------------
log()   { printf "\n[%(%F %T)T] %s\n" -1 "$*"; }
abort() { log "ERROR: $*"; exit 1; }
trap 'abort "Script failed at line $LINENO (command: $BASH_COMMAND)"' ERR

# ---- variables --------------------------------------------------------------
CPU_REMOTE_HOST="arrival-similarly-kerry-recognize.trycloudflare.com"
CPU_REMOTE_USER="t-atsonwane@microsoft.com"
SSH_KEY_PRIVATE="gcr"
SSH_KEY_PUBLIC="gcr.pub"
CONDA_DIR="$HOME/miniconda"
CONDA_ENV_NAME="rllm"

# apt in non-interactive mode
export DEBIAN_FRONTEND=noninteractive


###############################################################################
log "1. Configure SSH"
###############################################################################
mkdir -p "$HOME/.ssh"
install -m 0600 "$SSH_KEY_PRIVATE" "$HOME/.ssh/gcr"
install -m 0644 "$SSH_KEY_PUBLIC"  "$HOME/.ssh/gcr.pub"

SSH_CONFIG="$HOME/.ssh/config"
SSH_BACKUP="$SSH_CONFIG.$(date +%F_%T).bak"

SSH_HOST_BLOCK="$(cat <<EOF
Host kube-remote
    HostName ${CPU_REMOTE_HOST}
    User ${CPU_REMOTE_USER}
    IdentityFile ~/.ssh/gcr
    ControlMaster auto
    ControlPath ~/.ssh/control-%C
    ControlPersist 600
    ProxyCommand cloudflared access ssh --hostname %h
EOF
)"

# If ~/.ssh/config exists, back it up and prepend the block only if missing
if [[ -f "$SSH_CONFIG" ]]; then
  if ! grep -q "^Host kube-remote\b" "$SSH_CONFIG"; then
    log "▶ Backing up existing ~/.ssh/config to $SSH_BACKUP"
    cp "$SSH_CONFIG" "$SSH_BACKUP"
    { printf "%s\n\n" "$SSH_HOST_BLOCK"; cat "$SSH_BACKUP"; } > "$SSH_CONFIG"
  else
    log "▶ Host kube-remote already present in ~/.ssh/config – skipping insert"
  fi
else
  # No config yet; just write the new block
  printf "%s\n" "$SSH_HOST_BLOCK" > "$SSH_CONFIG"
fi
chmod 600 "$SSH_CONFIG"


###############################################################################
log "2. Install Cloudflared CLI"
###############################################################################
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg |
  sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null

echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] \
https://pkg.cloudflare.com/cloudflared any main" |
  sudo tee /etc/apt/sources.list.d/cloudflared.list

sudo apt-get update -y
sudo apt-get install -y cloudflared

###############################################################################
log "3. Create SSH tunnel"
###############################################################################

# Ensure autossh is available
if ! command -v autossh &>/dev/null; then
  sudo apt-get install -y autossh
fi

# Start tunnel:
#  -M 0    : disable status port, rely on SSH keep‑alives
#  -f      : background only after tunnel is up
#  -N      : no remote command
#  -L      : local→remote forward
autossh -M 0 -f -N \
  -L 6443:localhost:6443 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  kube-remote

SSH_PID=$(pgrep -f "autossh.*6443:localhost:6443.*kube-remote")
trap 'kill "$SSH_PID"' EXIT

# Quick health‑check (optional, but makes failures obvious)
# install nc if not available
if ! command -v nc &>/dev/null; then
  sudo apt-get install -y netcat
fi
for attempt in {1..10}; do
  if nc -z localhost 6443; then
    log "▶ Tunnel is up"
    break
  fi
  sleep 1
done || abort "SSH tunnel failed to come up"

###############################################################################
log "4. Install Kubernetes CLI (kubectl)"
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

kubectl get nodes

###############################################################################
log "5. Install additional dependencies"
###############################################################################
git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
cd R2E-Gym
# git checkout cai/rllm2
pip install -e .
cd ..
git clone --recurse-submodules https://github.com/threewisemonkeys-as/rllm.git
cd rllm
git checkout rllm2
pip install -e ./verl[vllm]
pip install -e .
python examples/swe/prepare_swe_data.py
cd ..
pip install docker[ssh]



###############################################################################
log "6. Run script"
###############################################################################

cd rllm/examples/swe
bash train_deepswe_8b_8h100.sh

log "All steps completed successfully!"

