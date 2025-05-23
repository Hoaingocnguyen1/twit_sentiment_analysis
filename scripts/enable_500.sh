#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# enable_port_5000.sh
#
# Script to enable port 5000 (or specified port) through UFW firewall
# for MLflow server access.
# This script can be used to open any port by passing --port parameter.
# =============================================================================

# --- Parse command line arguments ---
PORT=5000
DISABLE=0

# Check if running as root
if [[ $EUID -ne 0 ]]; then
  echo "⚠️  Please re-run this script with sudo: sudo $0 $*"
  exec sudo "$0" "$@"
fi

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift ;;
    --disable) DISABLE=1 ;;
    --help) 
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --port PORT     Port to enable/disable (default: 5000)"
      echo "  --disable       Disable the port instead of enabling it"
      echo "  --help          Show this help message"
      echo ""
      echo "Examples:"
      echo "  sudo $0                    # Enable port 5000"
      echo "  sudo $0 --port 8080        # Enable port 8080"
      echo "  sudo $0 --disable          # Disable port 5000"
      echo "  sudo $0 --port 8080 --disable  # Disable port 8080"
      exit 0
      ;;
    *) echo "Unknown parameter: $1. Use --help for usage information."; exit 1 ;;
  esac
  shift
done

# --- Validate port number ---
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "Error: Invalid port number: $PORT"
  echo "Port must be a number between 1 and 65535"
  exit 1
fi

# --- Function to check if UFW is installed ---
check_ufw_installed() {
  if ! command -v ufw &>/dev/null; then
    echo "Error: UFW (Uncomplicated Firewall) is not installed."
    echo "Please install UFW first:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install ufw"
    echo "  CentOS/RHEL:   sudo yum install ufw"
    exit 1
  fi
}

# --- Function to check UFW status ---
check_ufw_status() {
  local status
  status=$(ufw status | head -n 1)
  if [[ "$status" == *"inactive"* ]]; then
    return 1  # UFW is inactive
  else
    return 0  # UFW is active
  fi
}

# --- Function to enable UFW ---
enable_ufw() {
  echo "UFW is currently inactive. Enabling UFW..."
  echo "This will activate the firewall with default policies:"
  echo "  - Deny all incoming connections"
  echo "  - Allow all outgoing connections"
  echo ""
  read -p "Do you want to continue? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Set default policies before enabling
    ufw --force default deny incoming
    ufw --force default allow outgoing
    
    # Allow SSH to prevent lockout
    echo "Adding SSH rule to prevent lockout..."
    ufw allow ssh
    
    # Enable UFW
    ufw --force enable
    echo "UFW has been enabled with default policies and SSH access allowed."
  else
    echo "UFW activation cancelled. Port will not be opened."
    exit 1
  fi
}

# --- Function to check if port is already configured ---
check_port_status() {
  local port_status
  port_status=$(ufw status numbered | grep -E ":${PORT}(/tcp)?(\s|$)" | head -n 1 || echo "")
  
  if [ -n "$port_status" ]; then
    if [[ "$port_status" == *"ALLOW"* ]]; then
      return 0  # Port is allowed
    else
      return 2  # Port has other rules
    fi
  else
    return 1  # Port not configured
  fi
}

# --- Main execution ---
echo "=== UFW Port Management Script ==="
echo "Port: $PORT"
if [ $DISABLE -eq 1 ]; then
  echo "Action: DISABLE"
else
  echo "Action: ENABLE"
fi
echo ""

# Check if UFW is installed
check_ufw_installed

# Check UFW status
if ! check_ufw_status; then
  if [ $DISABLE -eq 1 ]; then
    echo "UFW is inactive. No need to disable port $PORT."
    exit 0
  else
    enable_ufw
  fi
fi

# Check current port status
check_port_status
port_status_result=$?

if [ $DISABLE -eq 1 ]; then
  # --- DISABLE PORT ---
  if [ $port_status_result -eq 1 ]; then
    echo "Port $PORT is not currently configured in UFW. Nothing to disable."
    exit 0
  else
    echo "Disabling port $PORT in UFW..."
    ufw delete allow "${PORT}/tcp" 2>/dev/null || true
    ufw delete allow "$PORT" 2>/dev/null || true
    
    # Reload UFW to apply changes
    ufw --force reload
    
    echo "✓ Port $PORT has been disabled in UFW"
    
    # Show current status
    echo ""
    echo "Current UFW status:"
    ufw status numbered | grep -E "(Status|^$|${PORT})" || echo "Port $PORT is no longer in UFW rules"
  fi
else
  # --- ENABLE PORT ---
  if [ $port_status_result -eq 0 ]; then
    echo "Port $PORT is already enabled in UFW."
    echo ""
    echo "Current UFW status:"
    ufw status numbered | grep -E "(Status|${PORT})"
    exit 0
  else
    echo "Enabling port $PORT in UFW..."
    
    # Add the rule
    ufw allow "${PORT}/tcp"
    
    # Reload UFW to apply changes
    ufw --force reload
    
    echo "✓ Port $PORT has been enabled in UFW"
    
    # Verify the rule was added
    echo ""
    echo "Verifying port configuration:"
    if ufw status | grep -q ":$PORT"; then
      echo "✓ Port $PORT is now allowed through UFW"
    else
      echo "⚠️  Warning: Port $PORT may not be properly configured"
    fi
    
    # Show current status
    echo ""
    echo "Current UFW status:"
    ufw status numbered | grep -E "(Status|${PORT})" || echo "No specific rule found for port $PORT"
  fi
fi

echo ""
echo "=== UFW Management Complete ==="

# Additional information for MLflow
if [ "$PORT" -eq 5000 ] && [ $DISABLE -eq 0 ]; then
  echo ""
  echo "ℹ️  MLflow Server Information:"
  echo "   Port 5000 is now open for MLflow server access"
  echo "   You can now start the MLflow server using: ./start_mlflow.sh"
  echo "   Access MLflow UI at: http://your-server-ip:5000"
fi