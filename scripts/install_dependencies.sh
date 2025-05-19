#!/usr/bin/env bash
# =============================================================================
# install_dependencies.sh
#
# Script to create a Python 3.12 virtual environment and install dependencies
# for the MLflow server. It locates requirements.txt in the script directory
# or its parent, sets up a venv in the parent directory of the script, then installs packages within it.
#
# Usage:
#   chmod +x install_dependencies.sh
#   ./install_dependencies.sh [--python PYTHON_EXECUTABLE] [--venv PATH]
#
# Options:
#   --python PYTHON_EXECUTABLE  Specify Python interpreter (default: python3.12)
#   --venv PATH                Path to virtual environment
#                              (default: parent_dir_of_script/venv)
# =============================================================================

set -euo pipefail

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Default settings
PYTHON=python3.12
VENV_PATH="$PARENT_DIR/venv"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --python)
      [[ -n ${2:-} ]] || { echo "Error: --python requires an argument." >&2; exit 1; }
      PYTHON=$2; shift 2;
      ;;
    --venv)
      [[ -n ${2:-} ]] || { echo "Error: --venv requires an argument." >&2; exit 1; }
      VENV_PATH=$2; shift 2;
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      echo "Usage: $0 [--python PYTHON_EXECUTABLE] [--venv PATH]" >&2
      exit 1;
      ;;
  esac
done

# Ensure Python exists
if ! command -v "$PYTHON" &> /dev/null; then
  echo "Error: $PYTHON not found. Please install Python 3.12 or specify correct interpreter." >&2
  exit 1
fi
VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$VERSION" != "3.12" ]]; then
  echo "Error: $PYTHON is version $VERSION, but Python 3.12 required." >&2
  exit 1
fi

echo "Using interpreter: $PYTHON (version $VERSION)"

# Locate requirements.txt
REQ_FILE=""
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  REQ_FILE="$SCRIPT_DIR/requirements.txt"
elif [ -f "$SCRIPT_DIR/../requirements.txt" ]; then
  REQ_FILE="$SCRIPT_DIR/../requirements.txt"
fi
if [ -z "$REQ_FILE" ]; then
  echo "Error: requirements.txt not found in $SCRIPT_DIR or its parent directory." >&2
  exit 1
fi

echo "Found requirements file: $REQ_FILE"

# Setup virtual environment
echo "Setting up virtual environment at $VENV_PATH"
$PYTHON -m venv "$VENV_PATH"
# Activate venv
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

# Ensure pip is up-to-date
pip install --upgrade pip

echo "Installing dependencies from $REQ_FILE"
pip install -r "$REQ_FILE"

echo "Dependencies installed successfully in venv at $VENV_PATH"