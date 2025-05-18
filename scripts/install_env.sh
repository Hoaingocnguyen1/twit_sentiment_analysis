# !/usr/bin/env bash
# install_env.sh – Provision host with Docker (v20.10+), Compose v2 plugin, Miniconda
# Usage: sudo bash scripts/install_env.sh

set -euo pipefail
IFS=$'\n\t'

#── Helper Functions ────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

#── Pre-req Check ────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  error "This script must be run as root (sudo)."
fi

# Detect distro
if   grep -qiE 'ubuntu|debian' /etc/os-release; then DISTRO='debian'
elif grep -qiE 'rhel|centos|fedora'  /etc/os-release; then DISTRO='rhel'
else
  error "Unsupported distro. Please use Debian/Ubuntu or RHEL/CentOS."
fi

#── Update & Install APT Prereqs ────────────────────────────────────────────────
# install_prereqs_debian() {
#   info "Updating apt and installing prerequisites..."
#   apt-get update -y
#   apt-get install -y --no-install-recommends \
#     ca-certificates curl gnupg lsb-release
# }

# install_prereqs_rhel() {
#   info "Installing prerequisites via dnf..."
#   dnf install -y \
#     ca-certificates curl gnupg2
# }

# #── Docker & Compose v2 ─────────────────────────────────────────────────────────
# install_docker() {
#   if command -v docker &> /dev/null; then
#     warn "Docker already installed; skipping."
#     return
#   fi

#   info "Installing Docker Engine & Compose v2 plugin..."
#   case $DISTRO in
#     debian)
#       # Add Docker GPG key & repo
#       mkdir -p /etc/apt/keyrings
#       curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg \
#         | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

#       echo \
#       "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
#       https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
#       $(lsb_release -cs) stable" \
#       > /etc/apt/sources.list.d/docker.list

#       apt-get update -y
#       apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
#       ;;

#     rhel)
#       dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
#       dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
#       ;;
#   esac

#   # Enable & start
#   systemctl enable docker
#   systemctl start docker

#   # Add users to docker group
#   if [[ -n "${SUDO_USER-}" ]]; then
#     info "Adding $SUDO_USER to docker group."
#     usermod -aG docker "$SUDO_USER" || warn "Couldn't add $SUDO_USER to docker group."
#   fi

#   info "Docker installed: $(docker --version)"
#   info "Docker Compose plugin: $(docker compose version)"
# }

# #── Firewall (UFW) ──────────────────────────────────────────────────────────────
# configure_firewall() {
#   if command -v ufw &> /dev/null; then
#     info "Configuring UFW rules..."
#     ufw allow ssh
#     ufw allow 80,443/tcp    # HTTP/HTTPS
#     ufw allow 5000/tcp      # MLflow
#     ufw allow 8080/tcp      # Airflow
#     ufw allow 9090/tcp      # Prometheus
#     ufw allow 3000/tcp      # Grafana
#     ufw reload || warn "UFW reload failed."
#   else
#     warn "UFW not installed; skipping firewall config."
#   fi
# }

#── Miniconda ───────────────────────────────────────────────────────────────────
install_miniconda() {
  if command -v conda &> /dev/null; then
    warn "Conda already installed; skipping."
    return
  fi

  info "Installing Miniconda to /opt/miniconda3..."
  local installer=/tmp/Miniconda3-latest.sh
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$installer"
  bash "$installer" -b -p /opt/miniconda3
  /opt/miniconda3/bin/conda init --system bash

  info "Miniconda installed. Run 'source /etc/profile.d/conda.sh' to start using conda."
}

#── Main ────────────────────────────────────────────────────────────────────────
main() {
  case $DISTRO in
    debian) install_prereqs_debian ;;
    rhel)   install_prereqs_rhel   ;;
  esac

  install_docker
  configure_firewall
  install_miniconda

  info "All done! Please log out and log back in (or run 'newgrp docker') to apply group changes."
}

main "$@"
