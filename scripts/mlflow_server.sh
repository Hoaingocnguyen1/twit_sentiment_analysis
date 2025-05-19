#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# start_mlflow.sh
#
# Script to launch an MLflow Tracking Server with Azure Blob artifact store
# and PostgreSQL backend store on Azure.
# Loads environment variables from .env (script/parent), configures backend
# store and Azure artifact store, activates virtual environment if present,
# then starts the server.
# =============================================================================

# --- Parse command line arguments ---
FOREGROUND=0
CHECK_SERVER=0

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --host) HOST="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --foreground) FOREGROUND=1 ;;
    --check) CHECK_SERVER=1 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

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
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
else
  echo "Warning: .env file not found in script or parent directory"
fi

# Mirror LAKE_STORAGE_CONN_STR to AZURE_STORAGE_CONNECTION_STRING if needed
if [ -n "${LAKE_STORAGE_CONN_STR:-}" ] && [ -z "${AZURE_STORAGE_CONNECTION_STRING:-}" ]; then
  export AZURE_STORAGE_CONNECTION_STRING="$LAKE_STORAGE_CONN_STR"
fi

# --- Check for required environment variables ---
REQUIRED_VARS=("PG_HOST" "PG_PORT" "PG_DB2" "PG_USER" "PG_PASSWORD" "AZURE_ACCOUNT_NAME" "AZURE_BLOB_CONTAINER")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!var:-}" ]; then
    MISSING_VARS+=("$var")
  fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
  echo "Error: Missing required environment variables: ${MISSING_VARS[*]}"
  echo "Please set these variables in your .env file or export them before running this script."
  exit 1
fi

# --- Activate virtual environment if available ---
VENV_DIR="$(dirname "$SCRIPT_DIR")/twit_mlops"
if [ -d "$VENV_DIR" ]; then
  echo "Activating virtual environment at $VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "Virtual environment not found at $VENV_DIR, proceeding without activation"
fi 

# --- Check MLflow installation ---
if ! command -v mlflow &> /dev/null; then
  echo "Error: MLflow not found. Please install MLflow using: pip install mlflow"
  exit 1
fi

# --- Build backend store URI ---
if [ -n "${MLFLOW_BACKEND_STORE_URI:-}" ]; then
  BACKEND_STORE_URI="$MLFLOW_BACKEND_STORE_URI"
else
  # Construct PostgreSQL URI: postgresql://user:password@host:port/dbname
  BACKEND_STORE_URI="postgresql://${PG_USER}:${PG_PASSWORD}@${PG_HOST}:${PG_PORT}/${PG_DB2}"
fi

# --- Default configuration ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
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
if [ $FOREGROUND -eq 1 ]; then
  echo "Starting MLflow server in foreground mode"
  mlflow server \
    --host "$HOST" \
    --port "$PORT" \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "$ARTIFACT_ROOT"
else
  echo "Starting MLflow server in background mode"
  nohup mlflow server \
    --host "$HOST" \
    --port "$PORT" \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    > "$LOG_FILE" 2>&1 &
  
  SERVER_PID=$!
  echo "MLflow server started (PID: $SERVER_PID)"
  
  # Give the server a moment to start
  sleep 2
  
  # Check if the server is running
  if [ $CHECK_SERVER -eq 1 ]; then
    echo "Checking server status..."
    if ps -p $SERVER_PID > /dev/null; then
      echo "Server is running with PID: $SERVER_PID"
      # Test if the server is responding
      if command -v curl &> /dev/null; then
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://$HOST:$PORT/health || echo "Failed")
        if [ "$HTTP_CODE" = "200" ]; then
          echo "Server is responding correctly (HTTP 200)"
          
          # Test backend store connection
          echo "Checking PostgreSQL backend store connection..."
          if command -v psql &> /dev/null; then
            if PGPASSWORD="$PG_PASSWORD" psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB2" -c "SELECT 1" &> /dev/null; then
              echo "✓ Successfully connected to PostgreSQL backend store"
            else
              echo "✗ Failed to connect to PostgreSQL backend store"
              echo "  Please check your PostgreSQL credentials and connection settings"
            fi
          else
            echo "psql command not found. Cannot verify PostgreSQL connection."
          fi
          
          # Test Azure Blob Storage artifact store
          echo "Checking Azure Blob Storage artifact store connection..."
          if command -v az &> /dev/null; then
            if [ -n "${AZURE_STORAGE_CONNECTION_STRING:-}" ]; then
              # Test using az storage command
              if az storage container exists --name "$AZURE_BLOB_CONTAINER" --connection-string "$AZURE_STORAGE_CONNECTION_STRING" &> /dev/null; then
                echo "✓ Successfully connected to Azure Blob Storage artifact store"
              else
                echo "✗ Failed to connect to Azure Blob Storage. Container may not exist: $AZURE_BLOB_CONTAINER"
              fi
            else
              echo "✗ AZURE_STORAGE_CONNECTION_STRING not set. Cannot verify Azure Blob Storage connection."
            fi
          elif command -v python3 &> /dev/null; then
            echo "Checking Azure connection using Python Azure SDK..."
            PYTHON_CHECK=$(python3 -c "
import sys
try:
    from azure.storage.blob import BlobServiceClient
    conn_str = '$AZURE_STORAGE_CONNECTION_STRING'
    if not conn_str:
        print('AZURE_STORAGE_CONNECTION_STRING not set')
        sys.exit(1)
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service_client.get_container_client('$AZURE_BLOB_CONTAINER')
    if container_client.exists():
        print('success')
    else:
        print('Container does not exist: $AZURE_BLOB_CONTAINER')
        sys.exit(1)
except Exception as e:
    print(f'Error: {str(e)}')
    sys.exit(1)
" 2>&1)
            if [ "$PYTHON_CHECK" = "success" ]; then
              echo "✓ Successfully connected to Azure Blob Storage artifact store"
            else
              echo "✗ Failed to connect to Azure Blob Storage: $PYTHON_CHECK"
            fi
          else
            echo "Neither az CLI nor Python with Azure SDK found. Cannot verify Azure Blob Storage connection."
          fi
        else
          echo "Warning: Server might not be responding correctly. HTTP code: $HTTP_CODE"
          echo "Check $LOG_FILE for details"
        fi
      else
        echo "curl not found. Cannot check server response."
      fi
    else
      echo "Error: Server failed to start. Check $LOG_FILE for details"
      tail -n 20 "$LOG_FILE"
      exit 1
    fi
  fi
fi