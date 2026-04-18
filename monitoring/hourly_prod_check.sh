#!/bin/bash
# Hourly Claude-driven prod trading audit.
# Invoked by cron during trading day (Mon-Fri 13:00-20:00 UTC / 9:00-16:00 ET).
#
# Settings (in ~/.claude/settings.json) already provide:
#   model: opus (→ latest = claude-opus-4-7)
#   effortLevel: xhigh
#   skipDangerousModePermissionPrompt: true
# We still pass --dangerously-skip-permissions and --model opus explicitly
# so behaviour stays pinned if the settings file is edited.
set -euo pipefail

REPO=/nvme0n1-disk/code/stock-prediction
cd "$REPO"

LOG_DIR="$REPO/monitoring/logs"
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/hourly_prod_${TS}.log"

PROMPT_FILE="$REPO/monitoring/hourly_prod_check_prompt.md"
if [ ! -f "$PROMPT_FILE" ]; then
  echo "missing prompt file: $PROMPT_FILE" >&2
  exit 2
fi

# Prevent overlapping runs if a previous invocation is still alive.
LOCK="$LOG_DIR/.hourly_prod_check.lock"
exec 9> "$LOCK"
if ! flock -n 9; then
  echo "[hourly] previous run still holding lock ($LOCK), skipping" >&2
  exit 0
fi

PROMPT=$(cat "$PROMPT_FILE")

echo "=== Hourly prod audit $(date -u -Iseconds) ===" | tee -a "$LOG"
echo "=== PROMPT sent to Claude (opus, xhigh effort, --dangerously-skip-permissions) ===" | tee -a "$LOG"
echo "$PROMPT_FILE ($(wc -c < "$PROMPT_FILE") bytes)" | tee -a "$LOG"

# 50-min hard cap per run. Claude settings pick up opus + xhigh effort.
# Stream output so partial progress is captured even if claude is killed.
# Raw stream-json lands in .raw.jsonl; a filtered human-readable projection is
# appended to $LOG via a Python filter so we see progress mid-run.
RAW_LOG="${LOG%.log}.raw.jsonl"

set +e
timeout 3000 /home/administrator/.bun/bin/claude \
    -p "$PROMPT" \
    --dangerously-skip-permissions \
    --model opus \
    --permission-mode bypassPermissions \
    --add-dir "$REPO" \
    --verbose \
    --output-format stream-json \
    --include-partial-messages \
    2> >(tee -a "$LOG" >&2) \
    | tee "$RAW_LOG" \
    | python3 -u "$REPO/monitoring/filter_stream.py" \
    | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[hourly] claude exited with code $rc" | tee -a "$LOG"

echo "=== Hourly prod audit complete $(date -u -Iseconds) ===" | tee -a "$LOG"
