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

# Check for required commands
command -v curl >/dev/null 2>&1 || error_exit "curl is required but not installed"
command -v tar >/dev/null 2>&1 || error_exit "tar is required but not installed"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER_DIR="$SCRIPT_DIR/actions-runner"
mkdir -p "$RUNNER_DIR" || error_exit "Failed to create runner directory"

RUNNER_VERSION="2.317.0"

usage() {
  cat <<EOF
Usage: $0 --url <repo-or-org-url> [--name <runner-name>] [--labels <labels>] [--work <work-dir>] [--ephemeral]

Env:
  GITHUB_RUNNER_TOKEN   Runner registration token (preferred over --token)

Options:
  --url URL             GitHub repo/org URL, e.g. https://github.com/OWNER/REPO or https://github.com/OWNER
  --token TOKEN         Runner registration token (avoid: prefer GITHUB_RUNNER_TOKEN env or secrets/github_runner_token)
  --name NAME           Runner name (default: hostname + -stock)
  --labels LABELS       Comma-separated labels (default: stock-ci,linux,gpu)
  --work DIR            Work directory (default: _work)
  --ephemeral           Ephemeral mode (auto remove after one job)
EOF
}

URL=""
NAME="$(hostname)-stock"
LABELS="stock-ci,linux,gpu"
WORK="_work"
EPHEMERAL="false"
TOKEN="${GITHUB_RUNNER_TOKEN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2;;
    --token) TOKEN="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --labels) LABELS="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --ephemeral) EPHEMERAL="true"; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ -z "$URL" ]]; then
  error_exit "--url is required. Please specify the GitHub repository or organization URL"
fi

# Validate URL format
if [[ ! "$URL" =~ ^https://github\.com/[^/]+(/[^/]+)?$ ]]; then
  error_exit "Invalid URL format. Expected: https://github.com/OWNER/REPO or https://github.com/OWNER"
fi

# Try to find token from multiple sources
if [[ -z "$TOKEN" ]]; then
  # Try secrets directory in parent
  if [[ -f "$SCRIPT_DIR/../secrets/github_runner_token" ]]; then
    TOKEN=$(cat "$SCRIPT_DIR/../secrets/github_runner_token")
    success_msg "Token loaded from secrets/github_runner_token"
  # Try secrets directory in current
  elif [[ -f "$SCRIPT_DIR/secrets/github_runner_token" ]]; then
    TOKEN=$(cat "$SCRIPT_DIR/secrets/github_runner_token")
    success_msg "Token loaded from githubagent/secrets/github_runner_token"
  fi
fi

if [[ -z "$TOKEN" ]]; then
  error_exit "Runner token not provided. Please either:\n  1. Set GITHUB_RUNNER_TOKEN environment variable\n  2. Create secrets/github_runner_token file\n  3. Use --token parameter (not recommended)"
fi

# Validate token format (basic check)
if [[ ! "$TOKEN" =~ ^[A-Z0-9]{20,}$ ]]; then
  warning_msg "Token format looks unusual. GitHub runner tokens typically start with 'A' and are 29+ characters"
fi

cd "$RUNNER_DIR" || error_exit "Failed to change to runner directory"

# Check if runner already configured
if [[ -f .runner ]] && [[ -f ./config.sh ]]; then
  warning_msg "Runner already configured in this directory"
  read -p "Do you want to reconfigure? This will remove the existing configuration. (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting without changes"
    exit 0
  fi
  # Remove existing configuration
  if [[ -x ./config.sh ]]; then
    ./config.sh remove --token "$TOKEN" 2>/dev/null || true
  fi
fi

if [[ ! -f ./config.sh ]]; then
  echo "Downloading GitHub Actions runner v${RUNNER_VERSION}..."
  ARCHIVE="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
  
  # Download with progress bar and error handling
  if ! curl -L --progress-bar -o "$ARCHIVE" "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/$ARCHIVE"; then
    error_exit "Failed to download runner. Check your internet connection and GitHub access"
  fi
  
  echo "Extracting runner archive..."
  if ! tar xzf "$ARCHIVE"; then
    rm -f "$ARCHIVE"
    error_exit "Failed to extract runner archive"
  fi
  
  rm -f "$ARCHIVE"
  success_msg "Runner downloaded and extracted successfully"
fi

echo "Configuring runner..."
echo "  Name: $NAME"
echo "  URL: $URL"
echo "  Labels: $LABELS"
echo "  Work directory: $WORK"
if [[ "$EPHEMERAL" == "true" ]]; then
  echo "  Mode: Ephemeral (will be removed after one job)"
fi

EPH_ARG=()
if [[ "$EPHEMERAL" == "true" ]]; then
  EPH_ARG+=("--ephemeral")
fi

# Configure runner with error handling
if ! ./config.sh \
  --url "$URL" \
  --token "$TOKEN" \
  --name "$NAME" \
  --labels "$LABELS" \
  --work "$WORK" \
  --unattended \
  "${EPH_ARG[@]}"; then
  error_exit "Failed to configure runner. Check token validity and permissions"
fi

success_msg "Runner configured successfully!"
echo ""
echo "Next steps:"
echo "  1. Test runner:     cd $RUNNER_DIR && ./run.sh"
echo "  2. Install service: sudo bash $SCRIPT_DIR/install_systemd.sh --name $NAME"
echo "  3. Start service:   sudo systemctl start actions-runner@$NAME"
