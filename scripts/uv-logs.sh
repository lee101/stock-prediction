#!/usr/bin/env bash
set -euo pipefail
# Usage: scripts/uv-logs.sh sync
RUST_LOG=uv=debug uv -v "$@"
