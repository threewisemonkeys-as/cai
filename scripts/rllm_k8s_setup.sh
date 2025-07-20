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
DOCKERD_REMOTE_HOST="boots-liked-substantial-plus.trycloudflare.com"
DOCKERD_REMOTE_USER="t-atsonwane@microsoft.com"
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

SSH_HOST_BLOCK="$(cat <<'EOF'
Host kube-remote
    HostName arrival-similarly-kerry-recognize.trycloudflare.com
    User t-atsonwane@microsoft.com
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
log "3. Install Kubernetes CLI (kubectl)"
###############################################################################

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
mkdir -p ~/.local/bin
mv ./kubectl ~/.local/bin/kubectl
export PATH="$PATH:$HOME/.local/bin"
if ! grep -q 'export PATH="\$PATH:\$HOME/.local/bin"' "$HOME/.bashrc"; then
  echo 'export PATH="$PATH:$HOME/.local/bin"' >> "$HOME/.bashrc"
fi

KUBECONFIG_FILE="$(dirname "$0")/k3s.yaml"
if ! grep -q 'export KUBECONFIG=' "$HOME/.bashrc";
then
  echo "export KUBECONFIG=\"$KUBECONFIG_FILE\"" >> "$HOME/.bashrc"
fi

source "$HOME/.bashrc"

kubectl get nodes

###############################################################################
log "4. Create ssh tunnel in the background"
###############################################################################

ssh -vvv -N -L 6443:localhost:6443 -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 kube-remote &
SSH_PID=$!
trap "kill $SSH_PID" EXIT


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


bash rllm/examples/swe/train_deepswe_4b.sh

log "All steps completed successfully!"

