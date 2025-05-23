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
#
# IMPORTANT: This script prevents multiple MLflow servers from running on the
# same port. To use MLflow, instead of running "mlflow ui", use this script,
# and then access the UI at the URL shown on completion.
# =============================================================================

# --- Parse command line arguments ---
FOREGROUND=0
CHECK_SERVER=0
RESTART=0

if [[ $EUID -ne 0 ]]; then
  echo "⚠️  Please re-run this script with sudo: sudo $0 $*"
  exec sudo "$0" "$@"
fi

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --host) HOST="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --foreground) FOREGROUND=1 ;;
    --check) CHECK_SERVER=1 ;;
    --restart) RESTART=1 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

# --- Default configuration ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
LOG_FILE="${LOG_FILE:-mlflow_server.log}"
PID_FILE="/var/run/mlflow_server.pid"

# --- Function to check if MLflow is already running ---
check_mlflow_running() {
  # Check if process is running by PID file
  if [ -f "$PID_FILE" ]; then
    local pid
    pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
      return 0  # Process is running
    fi
  fi
  
  # Double-check by port
  if command -v lsof &> /dev/null; then
    if lsof -i :"$PORT" > /dev/null 2>&1; then
      return 0  # Port is in use
    fi
  elif command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ":$PORT "; then
      return 0  # Port is in use
    fi
  elif command -v ss &> /dev/null; then
    if ss -tuln | grep -q ":$PORT "; then
      return 0  # Port is in use
    fi
  fi
  
  return 1  # Not running
}

# --- Function to stop MLflow server ---
stop_mlflow_server() {
  # First, check if any process is using the port
  local port_pids=""
  
  if command -v lsof &> /dev/null; then
    port_pids=$(lsof -t -i:"$PORT" 2>/dev/null || echo "")
  elif command -v netstat &> /dev/null && command -v grep &> /dev/null && command -v awk &> /dev/null; then
    port_pids=$(netstat -tlnp 2>/dev/null | grep ":$PORT " | awk '{print $7}' | cut -d'/' -f1 || echo "")
  elif command -v ss &> /dev/null && command -v grep &> /dev/null && command -v awk &> /dev/null; then
    port_pids=$(ss -tlnp 2>/dev/null | grep ":$PORT " | awk '{print $6}' | cut -d',' -f2 | cut -d'=' -f2 || echo "")
  fi

  # If we found any PIDs using the port, try to kill them
  if [ -n "$port_pids" ]; then
    echo "Found processes using port $PORT: $port_pids"
    for pid in $port_pids; do
      if [ -n "$pid" ] && [ "$pid" != "" ]; then
        echo "Stopping process $pid using port $PORT..."
        kill "$pid" 2>/dev/null || true
      fi
    done
    
    # Give processes time to terminate
    sleep 2
    
    # Check again and force kill if necessary
    for pid in $port_pids; do
      if [ -n "$pid" ] && [ "$pid" != "" ] && ps -p "$pid" > /dev/null 2>&1; then
        echo "Force stopping process $pid..."
        kill -9 "$pid" 2>/dev/null || true
      fi
    done
  fi

  # Now handle the PID from our PID file
  if [ -f "$PID_FILE" ]; then
    local pid
    pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
      echo "Stopping MLflow server (PID: $pid)..."
      kill "$pid" 2>/dev/null || true
      
      # Wait for process to terminate
      local count=0
      while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
        sleep 1
        ((count++))
      done
      
      # Force kill if still running
      if ps -p "$pid" > /dev/null 2>&1; then
        echo "Force stopping MLflow server..."
        kill -9 "$pid" 2>/dev/null || true
      fi
      
      echo "MLflow server stopped"
    else
      echo "PID file exists but process is not running"
    fi
    rm -f "$PID_FILE"
  fi
}

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

# --- Check if MLflow is already running ---
if check_mlflow_running; then
  if [ $RESTART -eq 1 ]; then
    echo "MLflow server is already running. Stopping it before restart..."
    stop_mlflow_server
  else
    echo "MLflow server is already running on port $PORT."
    echo "Use --restart to stop the current server and start a new one."
    exit 0
  fi
fi

# --- Activate conda environment ---
if command -v conda &> /dev/null; then
  echo "Activating conda environment: mlflow-env"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate mlflow-env
else
  echo "Error: Conda not found. Please install Anaconda or Miniconda and create the 'mlflow-env' environment."
  exit 1
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

# --- Set Azure storage details ---
AZURE_ACCOUNT_NAME="${AZURE_ACCOUNT_NAME:-twitlakehouse}"
AZURE_BLOB_CONTAINER="${AZURE_BLOB_CONTAINER:-testartifact}"
ARTIFACT_ROOT="wasbs://${AZURE_BLOB_CONTAINER}@${AZURE_ACCOUNT_NAME}.blob.core.windows.net/mlflow-artifacts"

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
  exec mlflow server \
    --host "$HOST" \
    --port "$PORT" \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "$ARTIFACT_ROOT"
else
  echo "Starting MLflow server in background mode"
  # Clean up old log file
  [ -f "$LOG_FILE" ] && mv "$LOG_FILE" "${LOG_FILE}.old"
  
  # Start server with proper process management
  mlflow server \
    --host "$HOST" \
    --port "$PORT" \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --serve-artifacts \
    > "$LOG_FILE" 2>&1 &
  
  SERVER_PID=$!
  echo "$SERVER_PID" > "$PID_FILE"
  echo "MLflow server started (PID: $SERVER_PID)"
  
  # Give the server a moment to start
  sleep 2
  
  # Check if the server process is still running
  if ! ps -p "$SERVER_PID" > /dev/null; then
    echo "Error: Server failed to start or exited immediately"
    echo "Last few lines of log file:"
    tail -n 20 "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
  fi
  
  # Create a blocking file to prevent direct mlflow ui calls
  MLFLOW_LOCKFILE="/tmp/mlflow_server_running_on_port_${PORT}"
  echo "PORT=$PORT" > "$MLFLOW_LOCKFILE"
  echo "PID=$SERVER_PID" >> "$MLFLOW_LOCKFILE"
  echo "STARTED=$(date)" >> "$MLFLOW_LOCKFILE"
  
  # Make the lock file readable by all users
  chmod 644 "$MLFLOW_LOCKFILE"
  
  # Check if the server is running
  if [ $CHECK_SERVER -eq 1 ]; then
    echo "Checking server status..."
    # Test if the server is responding
    if command -v curl &> /dev/null; then
      local retry_count=0
      local max_retries=5
      local success=0
      
      while [ $retry_count -lt $max_retries ]; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/health" 2>/dev/null || echo "Failed")
        if [ "$HTTP_CODE" = "200" ]; then
          success=1
          break
        fi
        echo "Waiting for server to become available (attempt $((retry_count+1))/$max_retries)..."
        sleep 2
        ((retry_count++))
      done
      
      if [ $success -eq 1 ]; then
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
  fi
fi

# --- Open port via ufw ---
if command -v ufw &>/dev/null; then
  echo "Ensuring UFW is enabled and port $PORT is allowed..."
  ufw enable || true
  ufw allow "${PORT}/tcp"
  ufw reload
  echo "Port $PORT opened in firewall"
else
  echo "ufw not found; please open port $PORT in your firewall manually"
fi

echo "Done. Access MLflow UI at http://$HOST:$PORT"
echo "To stop the server, run: sudo $SCRIPT_DIR/$(basename "$0") --restart"
echo ""
echo "IMPORTANT: Do NOT run 'mlflow ui' directly as it will conflict with this server."
echo "           Always use this script to manage the MLflow server."