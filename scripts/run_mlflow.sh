?!/usr/bin/env bash
# scripts/run_mlflow_project.sh – run MLflow project with reusable conda env
# Usage:
#   ./scripts/run_mlflow_project.sh [--param1 value1 ...]
# Example:
#   ./scripts/run_mlflow_project.sh -P model_name=bert-base-uncased

set -euo pipefail
IFS=$'\n\t'

info(){ echo -e "\033[1;34m[INFO]\033[0m $*"; }
error(){ echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# 1. Change to MLflow project directory
dir=$(dirname "${BASH_SOURCE[0]}")
PROJECT_DIR="$(cd "$dir/../mlflow" && pwd)"
info "Switching to MLflow project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# 2. Ensure conda exists and source it
command -v conda &>/dev/null || error "conda not found. Please install Miniconda/Anaconda first."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# 3. Env name matches conda.yaml
ENV_NAME="twit-sentiment-gpu"
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  info "Creating Conda env '${ENV_NAME}' from conda.yaml..."
  conda env create -f conda.yaml
else
  info "Conda env '${ENV_NAME}' already exists; skipping creation."
fi

# 4. Activate environment
info "Activating Conda env '${ENV_NAME}'..."
conda activate "$ENV_NAME"

# 5. Ensure mlflow CLI is available
command -v mlflow &>/dev/null || error "mlflow CLI not found in env '${ENV_NAME}'."

# 6. Run MLflow project, forwarding all parameters
info "Running MLflow project 'train' with args: $*"
mlflow run . -e train "$@"

#Run full file xong goi 2 cai duoi de train 
#./scripts/run_mlflow_project.sh -P model_name="bert-base-uncased"
#./scripts/run_mlflow_project.sh -P model_name="answerdotai/ModernBERT-base"