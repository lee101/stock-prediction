#!/usr/bin/env bash
# Scheduled Codex-driven production health check.
set -euo pipefail

REPO="${REPO:-/nvme0n1-disk/code/stock-prediction}"
cd "$REPO"

LOG_DIR="$REPO/monitoring/logs"
mkdir -p "$LOG_DIR"

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/codex_prod_${TS}.log"
RAW_LOG="${LOG%.log}.raw.jsonl"
LAST_MSG="${LOG%.log}.last.txt"
CURRENT_LOG="$LOG_DIR/codex_current.log"
PROMPT_FILE="$REPO/monitoring/codex_prod_check_prompt.md"
LOCK="$LOG_DIR/.codex_prod_check.lock"
CODEX_BIN="${CODEX_BIN:-/home/administrator/.bun/bin/codex}"

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
  printf '%s status=%s rc=%s log=%s raw_log=%s last_msg=%s log_sha256=%s raw_log_sha256=%s last_msg_sha256=%s\n' \
    "$(date -u -Iseconds)" "$status" "$rc" "$LOG" "$RAW_LOG" "$LAST_MSG" \
    "$(file_sha256 "$LOG")" "$(file_sha256 "$RAW_LOG")" "$(file_sha256 "$LAST_MSG")" \
    > "$tmp"
  mv "$tmp" "$CURRENT_LOG"
}

if [ ! -f "$PROMPT_FILE" ]; then
  echo "missing prompt file: $PROMPT_FILE" | tee -a "$LOG" >&2
  write_current_status SETUP_FAILED 2
  exit 2
fi
if [ ! -x "$CODEX_BIN" ]; then
  echo "missing codex binary: $CODEX_BIN" | tee -a "$LOG" >&2
  write_current_status SETUP_FAILED 2
  exit 2
fi

exec 9> "$LOCK"
if ! flock -n 9; then
  echo "[codex-prod] previous run still holding lock ($LOCK), skipping" | tee -a "$LOG"
  write_current_status SKIPPED_LOCK 0
  exit 0
fi

echo "=== Codex prod audit $(date -u -Iseconds) ===" | tee -a "$LOG"
echo "prompt: $PROMPT_FILE ($(wc -c < "$PROMPT_FILE") bytes)" | tee -a "$LOG"
echo "codex: $CODEX_BIN" | tee -a "$LOG"

if [ "${CODEX_PROD_CHECK_DRY_RUN:-0}" = "1" ]; then
  echo "dry run: would execute codex prod check" | tee -a "$LOG"
  write_current_status DRY_RUN 0
  exit 0
fi

PROMPT="$(cat "$PROMPT_FILE")"

set +e
timeout 2400 "$CODEX_BIN" exec \
  --cd "$REPO" \
  --dangerously-bypass-approvals-and-sandbox \
  --json \
  --output-last-message "$LAST_MSG" \
  "$PROMPT" \
  2> >(tee -a "$LOG" >&2) \
  | tee "$RAW_LOG" \
  | python3 -u "$REPO/monitoring/filter_stream.py" \
  | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "[codex-prod] codex exited with code $rc" | tee -a "$LOG"
final_rc="$rc"
if [ "$final_rc" -eq 0 ] && { [ ! -s "$RAW_LOG" ] || [ ! -s "$LAST_MSG" ]; }; then
  echo "[codex-prod] missing non-empty output artifact despite rc=0" | tee -a "$LOG"
  final_rc=70
fi
if [ "$final_rc" -eq 0 ]; then
  status=OK
else
  status=FAILED
fi
if [ -s "$LAST_MSG" ]; then
  {
    echo "--- last message ---"
    cat "$LAST_MSG"
  } | tee -a "$LOG"
fi
echo "=== Codex prod audit complete $(date -u -Iseconds) ===" | tee -a "$LOG"
write_current_status "$status" "$final_rc"
exit "$final_rc"
