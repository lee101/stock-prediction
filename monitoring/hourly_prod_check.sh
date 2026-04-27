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

REPO="${REPO:-/nvme0n1-disk/code/stock-prediction}"
cd "$REPO"

LOG_DIR="$REPO/monitoring/logs"
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/hourly_prod_${TS}.log"
RAW_LOG="${LOG%.log}.raw.jsonl"
CURRENT_LOG="$LOG_DIR/hourly_current.log"
CLAUDE_BIN="${CLAUDE_BIN:-/home/administrator/.bun/bin/claude}"

PROMPT_FILE="$REPO/monitoring/hourly_prod_check_prompt.md"
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
  printf '%s status=%s rc=%s log=%s raw_log=%s log_sha256=%s raw_log_sha256=%s\n' \
    "$(date -u -Iseconds)" "$status" "$rc" "$LOG" "$RAW_LOG" \
    "$(file_sha256 "$LOG")" "$(file_sha256 "$RAW_LOG")" \
    > "$tmp"
  mv "$tmp" "$CURRENT_LOG"
}

if [ ! -f "$PROMPT_FILE" ]; then
  echo "missing prompt file: $PROMPT_FILE" | tee -a "$LOG" >&2
  write_current_status SETUP_FAILED 2
  exit 2
fi
if [ ! -x "$CLAUDE_BIN" ]; then
  echo "missing claude binary: $CLAUDE_BIN" | tee -a "$LOG" >&2
  write_current_status SETUP_FAILED 2
  exit 2
fi

# Prevent overlapping runs if a previous invocation is still alive.
LOCK="$LOG_DIR/.hourly_prod_check.lock"
exec 9> "$LOCK"
if ! flock -n 9; then
  echo "[hourly] previous run still holding lock ($LOCK), skipping" | tee -a "$LOG" >&2
  write_current_status SKIPPED_LOCK 0
  exit 0
fi

PROMPT=$(cat "$PROMPT_FILE")

echo "=== Hourly prod audit $(date -u -Iseconds) ===" | tee -a "$LOG"
echo "=== PROMPT sent to Claude (opus, xhigh effort, --dangerously-skip-permissions) ===" | tee -a "$LOG"
echo "$PROMPT_FILE ($(wc -c < "$PROMPT_FILE") bytes)" | tee -a "$LOG"

# 50-min hard cap per run. Claude settings pick up opus + xhigh effort.
# Stream output so partial progress is captured even if claude is killed.
set +e
timeout 3000 "$CLAUDE_BIN" \
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
final_rc="$rc"
if [ "$final_rc" -eq 0 ] && [ ! -s "$RAW_LOG" ]; then
  echo "[hourly] missing non-empty raw stream artifact despite rc=0" | tee -a "$LOG"
  final_rc=70
fi
if [ "$final_rc" -eq 0 ]; then
  status=OK
else
  status=FAILED
fi

echo "=== Hourly prod audit complete $(date -u -Iseconds) ===" | tee -a "$LOG"
write_current_status "$status" "$final_rc"
exit "$final_rc"
