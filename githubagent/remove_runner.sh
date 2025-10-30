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

# Warning message
warning_msg() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

usage() {
    cat <<EOF
Usage: $0 [--token <removal-token>] [--url <repo-url>] [--force]

Removes a GitHub Actions runner registration.

Options:
  --token TOKEN   Removal token (can also use GITHUB_RUNNER_TOKEN env)
  --url URL       Repository/org URL (can also use GITHUB_RUNNER_URL env) 
  --force         Force removal without systemd cleanup prompts
  --help          Show this help message

Note: You can get a removal token from GitHub Settings > Actions > Runners > ... > Remove
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER_DIR="$SCRIPT_DIR/actions-runner"
FORCE="false"

TOKEN="${GITHUB_RUNNER_TOKEN:-}"
URL="${GITHUB_RUNNER_URL:-}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --token) TOKEN="$2"; shift 2;;
    --url) URL="$2"; shift 2;;
    --force) FORCE="true"; shift;;
    -h|--help) usage; exit 0;;
    *) error_exit "Unknown argument: $1";;
  esac
done

# Try to load token from file if not provided
if [[ -z "$TOKEN" ]]; then
  if [[ -f "$SCRIPT_DIR/../secrets/github_runner_token" ]]; then
    TOKEN=$(cat "$SCRIPT_DIR/../secrets/github_runner_token")
    success_msg "Token loaded from secrets file"
  elif [[ -f "$SCRIPT_DIR/secrets/github_runner_token" ]]; then
    TOKEN=$(cat "$SCRIPT_DIR/secrets/github_runner_token")
    success_msg "Token loaded from secrets file"
  fi
fi

if [[ -z "$URL" ]]; then
  warning_msg "No URL provided. Will attempt interactive removal if runner is configured"
fi

# Check if runner exists
if [[ ! -d "$RUNNER_DIR" ]]; then
  warning_msg "Runner directory not found at $RUNNER_DIR"
  echo "Nothing to remove"
  exit 0
fi

if [[ ! -x "$RUNNER_DIR/config.sh" ]]; then
  warning_msg "Runner not properly installed at $RUNNER_DIR"
  echo "Cleaning up directory..."
  rm -rf "$RUNNER_DIR"
  success_msg "Directory removed"
  exit 0
fi

# Check for running systemd services
if [[ "$FORCE" != "true" ]]; then
  SERVICES=$(systemctl list-units --all --no-legend 'actions-runner-*' 2>/dev/null | awk '{print $1}' || true)
  if [[ -n "$SERVICES" ]]; then
    echo -e "${YELLOW}Found systemd service(s):${NC}"
    echo "$SERVICES"
    echo ""
    echo "To stop and remove these services, run:"
    for service in $SERVICES; do
      name=${service%.service}
      name=${name#actions-runner-}
      echo "  sudo systemctl stop $service"
      echo "  sudo systemctl disable $service"
      echo "  sudo rm /etc/systemd/system/$service"
    done
    echo "  sudo systemctl daemon-reload"
    echo ""
    read -p "Continue with runner removal anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Aborted"
      exit 0
    fi
  fi
fi

cd "$RUNNER_DIR" || error_exit "Failed to change to runner directory"

# Get runner name for display
RUNNER_NAME="unknown"
if [[ -f .runner ]]; then
  RUNNER_NAME=$(grep -Po '"name"\s*:\s*"\K[^"]*' .runner 2>/dev/null || echo "unknown")
fi

echo "Removing runner: $RUNNER_NAME"

# Attempt removal
if [[ -n "$TOKEN" ]]; then
  echo "Using provided token for removal..."
  if ! ./config.sh remove --unattended --token "$TOKEN" 2>/dev/null; then
    error_exit "Failed to remove runner. Token may be invalid or expired\nGet a new removal token from GitHub Settings > Actions > Runners"
  fi
else
  warning_msg "No token provided. Attempting interactive removal..."
  echo "You will need to provide a removal token manually."
  echo "Get it from: GitHub Settings > Actions > Runners > ... > Remove"
  echo ""
  if ! ./config.sh remove; then
    error_exit "Failed to remove runner"
  fi
fi

success_msg "Runner '$RUNNER_NAME' removed successfully!"

# Clean up directory
read -p "Do you want to delete the runner directory? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  cd ..
  rm -rf "$RUNNER_DIR"
  success_msg "Runner directory deleted"
fi

echo ""
echo "Runner removal complete. Don't forget to:"
echo "  1. Remove any systemd services if they exist"
echo "  2. Clean up any secrets/tokens files"
echo "  3. Update your workflows if they reference this runner"

