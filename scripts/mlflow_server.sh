#!/usr/bin/env bash

# =============================================================================
# start_mlflow.sh
#
# Script to launch an MLflow Tracking Server with Azure Blob artifact store.
# Loads environment variables from .env (in script dir or parent),
# configures backend store and Azure artifact store, and starts the server.
# Excludes any test routines.
#
# Usage:
#   chmod +x start_mlflow.sh
#   ./start_mlflow.sh [--host HOST] [--port PORT] \
#                    [--backend-store-uri URI] \
#                    [--azure-account NAME] [--azure-container NAME]
#
# Example:
#   ./start_mlflow.sh --host 0.0.0.0 --port 5000 \
#                    --backend-store-uri sqlite:///mlflow.db \
#                    --azure-account twitlakehouse \
#                    --azure-container artifact
# =============================================================================

set -euo pipefail

# --- Locate and load .env (script directory or parent directory) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE=""
if [ -f "$SCRIPT_DIR/.env" ]; then
  ENV_FILE="$SCRIPT_DIR/.env"
elif [ -f "$SCRIPT_DIR/../.env" ]; then
  ENV_FILE="$SCRIPT_DIR/../.env"
fi

if [ -n "$ENV_FILE" ]; then
  echo "Loading environment variables from $ENV_FILE"
  export $(grep -v '^#' "$ENV_FILE" | xargs)
else
  echo "Warning: .env file not found in script or parent directory"
fi

# Mirror LAKE_STORAGE_CONN_STR to AZURE_STORAGE_CONNECTION_STRING if needed
if [ -n "${LAKE_STORAGE_CONN_STR:-}" ] && [ -z "${AZURE_STORAGE_CONNECTION_STRING:-}" ]; then
  export AZURE_STORAGE_CONNECTION_STRING="$LAKE_STORAGE_CONN_STR"
fi

# --- Default configuration ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}"
AZURE_ACCOUNT_NAME="${AZURE_ACCOUNT_NAME:-twitlakehouse}"
AZURE_BLOB_CONTAINER="${AZURE_BLOB_CONTAINER:-artifact}"
ARTIFACT_ROOT="wasbs://${AZURE_BLOB_CONTAINER}@${AZURE_ACCOUNT_NAME}.blob.core.windows.net/mlflow-artifacts"
LOG_FILE="${LOG_FILE:-mlflow_server.log}"

# --- Display configuration ---
echo "Starting MLflow Tracking Server with the following settings:"
echo "  HOST:               $HOST"
echo "  PORT:               $PORT"
echo "  BACKEND_STORE_URI:  $BACKEND_STORE_URI"
echo "  ARTIFACT_ROOT:      $ARTIFACT_ROOT"
echo "  LOG_FILE:           $LOG_FILE"

# --- Launch MLflow server ---
ohup mlflow server \
  --host "$HOST" \
  --port "$PORT" \
  --backend-store-uri "$BACKEND_STORE_URI" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  > "$LOG_FILE" 2>&1 &

echo "MLflow server started (PID: $!)"