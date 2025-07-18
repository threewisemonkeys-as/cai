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
log "1. Configure SSH for Cloudflare-proxied remote Docker daemon"
###############################################################################
mkdir -p "$HOME/.ssh"
install -m 0600 "$SSH_KEY_PRIVATE" "$HOME/.ssh/gcr"
install -m 0644 "$SSH_KEY_PUBLIC"  "$HOME/.ssh/gcr.pub"

SSH_CONFIG="$HOME/.ssh/config"
SSH_BACKUP="$SSH_CONFIG.$(date +%F_%T).bak"

DOCKER_SSH_BLOCK="$(cat <<'EOF'
Host dockerd-remote
    HostName boots-liked-substantial-plus.trycloudflare.com
    IdentityFile ~/.ssh/gcr
    User t-atsonwane@microsoft.com
    ProxyCommand cloudflared access ssh --hostname %h
EOF
)"

# If ~/.ssh/config exists, back it up and prepend the block only if missing
if [[ -f "$SSH_CONFIG" ]]; then
  if ! grep -q "^Host dockerd-remote\b" "$SSH_CONFIG"; then
    log "▶ Backing up existing ~/.ssh/config to $SSH_BACKUP"
    cp "$SSH_CONFIG" "$SSH_BACKUP"
    { printf "%s\n\n" "$DOCKER_SSH_BLOCK"; cat "$SSH_BACKUP"; } > "$SSH_CONFIG"
  else
    log "▶ Host dockerd-remote already present in ~/.ssh/config – skipping insert"
  fi
else
  # No config yet; just write the new block
  printf "%s\n" "$DOCKER_SSH_BLOCK" > "$SSH_CONFIG"
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
log "3. Install Docker Engine & plugins"
###############################################################################
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg |
  sudo tee /etc/apt/keyrings/docker.asc >/dev/null
sudo chmod a+r /etc/apt/keyrings/docker.asc

CODENAME=$(lsb_release -cs)
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $CODENAME stable" |
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
                        docker-buildx-plugin docker-compose-plugin



###############################################################################
log "7. Install additional dependencies"
###############################################################################
git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
git checkout cai/rllm
cd R2E-Gym
pip install -e .
cd ..
git clone https://github.com/threewisemonkeys-as/rllm.git
git checkout cai
cd rllm
pip install -e ./verl[vllm]
pip install -e .
cd ..
pip install docker[ssh]

###############################################################################
log "8. Run testing script"
###############################################################################
python conn_test.py

log "All steps completed successfully!"