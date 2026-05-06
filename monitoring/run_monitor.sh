#!/usr/bin/env bash
# Trading system health monitor - runs Claude to check production systems.
# Designed to be called by cron every 6 hours during market hours.
#
# Usage:
#   ./monitoring/run_monitor.sh          # one-shot health check
#   crontab: 0 0,6,12,18 * * * /home/lee/code/stock/monitoring/run_monitor.sh
#
# Output logged to monitoring/logs/

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/monitoring/logs"
PROMPT_FILE="$ROOT/monitoring/PROMPT.md"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/monitor_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Clean old logs (keep last 7 days)
find "$LOG_DIR" -name "monitor_*.log" -mtime +7 -delete 2>/dev/null || true

cd "$ROOT"

echo "=== Trading Monitor Run: $(date -u) ===" | tee "$LOG_FILE"

# Run Claude with the monitoring prompt
claude --dangerously-skip-permissions \
  -p "$(cat "$PROMPT_FILE")" \
  --max-turns 30 \
  >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "" >> "$LOG_FILE"
echo "=== Monitor completed: $(date -u), exit=$EXIT_CODE ===" >> "$LOG_FILE"

# Print summary
tail -5 "$LOG_FILE"

exit $EXIT_CODE
