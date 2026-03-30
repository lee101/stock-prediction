#!/bin/bash
# Alpaca trading monitor agent — runs Claude to check health and fix issues.
# Invoked by systemd timer during market hours.
set -euo pipefail

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

LOG_DIR="monitoring/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%S)
LOG_FILE="$LOG_DIR/monitor_${TIMESTAMP}.log"

# First: run the health check
echo "=== Health Check $(date -u -Iseconds) ===" | tee -a "$LOG_FILE"
python monitoring/health_check.py --json 2>&1 | tee -a "$LOG_FILE"
HEALTH_EXIT=$?

# If unhealthy, invoke Claude to diagnose and fix
if [ "$HEALTH_EXIT" -ne 0 ]; then
    echo "=== Unhealthy — spawning Claude agent ===" | tee -a "$LOG_FILE"

    PROMPT="You are monitoring the Alpaca stock trading system. The health check found issues.
Run: python monitoring/health_check.py --json
Review the output, then:
1. Check service logs for root causes (journalctl, supervisorctl)
2. Fix any issues you can (restart services, clean state, etc)
3. If the Alpaca API key is expired, warn the user — you cannot renew it
4. Re-run the health check to verify fixes
5. Update alpacaprod.md with current status
Keep it brief. Only fix what you can actually fix."

    timeout 1800 claude --dangerously-skip-permissions -p "$PROMPT" 2>&1 | tee -a "$LOG_FILE" || true
else
    echo "=== All checks passed ===" | tee -a "$LOG_FILE"
fi

echo "=== Monitor complete $(date -u -Iseconds) ===" | tee -a "$LOG_FILE"
