#!/bin/bash
# Alpaca trading monitor agent — runs Claude to check health and fix issues.
# Invoked by systemd timer during market hours.
set -euo pipefail

REPO="${REPO:-/nvme0n1-disk/code/stock-prediction}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"

cd "$REPO"

LOG_DIR="$REPO/monitoring/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%S)
LOG_FILE="$LOG_DIR/monitor_${TIMESTAMP}.log"
CURRENT_LOG="$LOG_DIR/monitor_current.log"
HEALTH_EXIT="NA"
FINAL_EXIT=2
AGENT_EXIT="NA"

file_sha256() {
    if [ -s "$1" ]; then
        sha256sum "$1" | awk '{print $1}'
    else
        printf 'NA'
    fi
}

write_current_status() {
    local status="$1"
    local rc="$2"
    local tmp
    tmp=$(mktemp "${CURRENT_LOG}.tmp.XXXXXX")
    printf '%s status=%s rc=%s initial_rc=%s final_rc=%s agent_rc=%s log=%s log_sha256=%s\n' \
        "$(date -u -Iseconds)" "$status" "$rc" "$HEALTH_EXIT" "$FINAL_EXIT" "$AGENT_EXIT" "$LOG_FILE" "$(file_sha256 "$LOG_FILE")" \
        > "$tmp"
    mv "$tmp" "$CURRENT_LOG"
}

if [ ! -f .venv313/bin/activate ]; then
    echo "=== Monitor setup failed: missing .venv313/bin/activate ===" | tee -a "$LOG_FILE"
    echo "=== Monitor complete $(date -u -Iseconds) ===" | tee -a "$LOG_FILE"
    write_current_status SETUP_FAILED "$FINAL_EXIT"
    exit "$FINAL_EXIT"
fi
if ! source .venv313/bin/activate; then
    echo "=== Monitor setup failed: could not activate .venv313 ===" | tee -a "$LOG_FILE"
    echo "=== Monitor complete $(date -u -Iseconds) ===" | tee -a "$LOG_FILE"
    write_current_status SETUP_FAILED "$FINAL_EXIT"
    exit "$FINAL_EXIT"
fi

run_health_check() {
    set +e
    HEALTH_CHECK_SKIP_MONITOR_CURRENT=1 python monitoring/health_check.py --json 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    set -e
    return "$exit_code"
}

# First: run the health check
echo "=== Health Check $(date -u -Iseconds) ===" | tee -a "$LOG_FILE"
if run_health_check; then
    HEALTH_EXIT=0
else
    HEALTH_EXIT=$?
fi
FINAL_EXIT=$HEALTH_EXIT

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

    set +e
    timeout 1800 "$CLAUDE_BIN" --dangerously-skip-permissions -p "$PROMPT" 2>&1 | tee -a "$LOG_FILE"
    AGENT_EXIT=${PIPESTATUS[0]}
    set -e
    echo "=== Agent exited with code $AGENT_EXIT ===" | tee -a "$LOG_FILE"

    echo "=== Post-agent health check $(date -u -Iseconds) ===" | tee -a "$LOG_FILE"
    if run_health_check; then
        FINAL_EXIT=0
    else
        FINAL_EXIT=$?
    fi
    if [ "$FINAL_EXIT" -ne 0 ]; then
        echo "=== Still unhealthy after agent ===" | tee -a "$LOG_FILE"
    else
        echo "=== Recovered after agent ===" | tee -a "$LOG_FILE"
    fi
else
    echo "=== All checks passed ===" | tee -a "$LOG_FILE"
fi

echo "=== Monitor complete $(date -u -Iseconds) ===" | tee -a "$LOG_FILE"
if [ "$HEALTH_EXIT" -eq 0 ]; then
    MONITOR_STATUS=OK
elif [ "$FINAL_EXIT" -eq 0 ]; then
    MONITOR_STATUS=RECOVERED
else
    MONITOR_STATUS=STILL_UNHEALTHY
fi
write_current_status "$MONITOR_STATUS" "$FINAL_EXIT"
exit "$FINAL_EXIT"
