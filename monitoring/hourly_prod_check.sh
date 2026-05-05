#!/usr/bin/env bash
# Back-compat cron entrypoint for hourly production checks.
#
# The historical implementation used Claude and an XGB-only prompt. The current
# production monitor is Codex-based and reads alpacaprod.md for the canonical
# writer path, so delegate here rather than keeping two audit systems alive.
set -euo pipefail

REPO="${REPO:-/nvme0n1-disk/code/stock-prediction}"
cd "$REPO"

echo "=== Hourly prod audit delegating to codex_prod_check $(date -u -Iseconds) ==="
exec "$REPO/monitoring/codex_prod_check.sh"
