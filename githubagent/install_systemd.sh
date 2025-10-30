#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Error handler
error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit "${2:-1}"
}

# Success message
success_msg() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Check if running with sudo
if [[ $EUID -ne 0 ]]; then
   error_exit "This script must be run with sudo"
fi

# Check for systemd
if ! command -v systemctl >/dev/null 2>&1; then
    error_exit "systemd is required but not found"
fi

usage() {
    cat <<EOF
Usage: sudo $0 --name <runner-name> [--user <username>]

Options:
  --name NAME    Runner name (required, must match registered runner)
  --user USER    User to run the service as (default: current sudo user)
  --help         Show this help message
EOF
}

NAME=""
SERVICE_USER="${SUDO_USER:-$USER}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --user) SERVICE_USER="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) error_exit "Unknown argument: $1";;
  esac
done

if [[ -z "$NAME" ]]; then
    error_exit "--name is required. Specify the runner name used during registration"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER_DIR="$SCRIPT_DIR/actions-runner"

# Verify runner directory exists
if [[ ! -d "$RUNNER_DIR" ]]; then
    error_exit "Runner directory not found at $RUNNER_DIR. Please run register_runner.sh first"
fi

# Verify runner is configured
if [[ ! -f "$RUNNER_DIR/.runner" ]]; then
    error_exit "Runner not configured. Please run register_runner.sh first"
fi

# Verify user exists
if ! id "$SERVICE_USER" &>/dev/null; then
    error_exit "User '$SERVICE_USER' does not exist"
fi

UNIT_DIR="/etc/systemd/system"
UNIT_FILE="$UNIT_DIR/actions-runner-${NAME}.service"

# Check if service already exists
if [[ -f "$UNIT_FILE" ]]; then
    echo -e "${YELLOW}Service already exists at $UNIT_FILE${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting without changes"
        exit 0
    fi
    # Stop existing service if running
    systemctl stop "actions-runner-${NAME}" 2>/dev/null || true
fi

mkdir -p "$UNIT_DIR" || error_exit "Failed to create systemd directory"

echo "Creating systemd service file..."
cat <<SERVICE > "$UNIT_FILE"
[Unit]
Description=GitHub Actions Runner - $NAME
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$RUNNER_DIR
ExecStart=$RUNNER_DIR/run.sh
KillMode=process
KillSignal=SIGTERM
TimeoutStopSec=5min
Restart=always
RestartSec=10s

# Security settings
PrivateTmp=true
NoNewPrivileges=true
RestrictSUIDSGID=true

# Environment
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="RUNNER_ALLOW_RUNASROOT=0"

# Resource limits (optional, adjust as needed)
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
SERVICE

if [[ ! -f "$UNIT_FILE" ]]; then
    error_exit "Failed to create service file"
fi

# Set proper permissions
chown root:root "$UNIT_FILE"
chmod 644 "$UNIT_FILE"

# Ensure runner directory is owned by service user
chown -R "$SERVICE_USER:$SERVICE_USER" "$RUNNER_DIR"

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

success_msg "Systemd service installed successfully!"
echo ""
echo "Service name: actions-runner-${NAME}"
echo "Service user: $SERVICE_USER"
echo "Runner directory: $RUNNER_DIR"
echo ""
echo "Next steps:"
echo "  1. Start the service:   sudo systemctl start actions-runner-${NAME}"
echo "  2. Enable on boot:      sudo systemctl enable actions-runner-${NAME}"
echo "  3. Check status:        sudo systemctl status actions-runner-${NAME}"
echo "  4. View logs:           sudo journalctl -u actions-runner-${NAME} -f"

