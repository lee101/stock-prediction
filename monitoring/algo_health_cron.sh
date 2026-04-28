#!/bin/bash
# Lightweight deterministic algo-health check (no LLM).
# Runs the algo_health_report.py script and writes its stdout to a timestamped
# log + appends a one-line OK/FAIL summary to algo_health_cron.log.
#
# Designed for a fast cadence (every 15 min during market hours). Holds a
# flock so overlapping invocations skip silently.
set -euo pipefail

REPO="${REPO:-/nvme0n1-disk/code/stock-prediction}"
cd "$REPO"

LOG_DIR="$REPO/monitoring/logs"
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/algo_health_cron_${TS}.log"
SUMMARY="$LOG_DIR/algo_health_cron.log"

LOCK="$LOG_DIR/.algo_health_cron.lock"
exec 9> "$LOCK"
if ! flock -n 9; then
  echo "$(date -u -Iseconds) SKIPPED_LOCK" >> "$SUMMARY"
  exit 0
fi

set +e
timeout 240 "$REPO/.venv/bin/python" "$REPO/monitoring/algo_health_report.py" \
  > "$LOG" 2>&1
rc=$?
set -e

# Pull the overall verdict line (first line of report)
verdict=$(head -2 "$LOG" 2>/dev/null | tail -1 || true)
echo "$(date -u -Iseconds) rc=$rc $verdict" >> "$SUMMARY"

# Keep only last 200 detail logs to bound disk
ls -1t "$LOG_DIR"/algo_health_cron_*.log 2>/dev/null | tail -n +201 | xargs -r rm -f || true

exit "$rc"
