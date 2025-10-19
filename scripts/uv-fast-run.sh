#!/usr/bin/env bash
set -euo pipefail
# Usage: scripts/uv-fast-run.sh --package <pkg> python -m <module>
uv run --frozen --no-sync "$@"
