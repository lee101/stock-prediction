#!/usr/bin/env bash
# Scheduled Codex-driven production health check.
set -euo pipefail

REPO=/nvme0n1-disk/code/stock-prediction
cd "$REPO"

LOG_DIR="$REPO/monitoring/logs"
mkdir -p "$LOG_DIR"

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/codex_prod_${TS}.log"
RAW_LOG="${LOG%.log}.raw.jsonl"
LAST_MSG="${LOG%.log}.last.txt"
PROMPT_FILE="$REPO/monitoring/codex_prod_check_prompt.md"
LOCK="$LOG_DIR/.codex_prod_check.lock"
CODEX_BIN="${CODEX_BIN:-/home/administrator/.bun/bin/codex}"

if [ ! -f "$PROMPT_FILE" ]; then
  echo "missing prompt file: $PROMPT_FILE" >&2
  exit 2
fi
if [ ! -x "$CODEX_BIN" ]; then
  echo "missing codex binary: $CODEX_BIN" >&2
  exit 2
fi

exec 9> "$LOCK"
if ! flock -n 9; then
  echo "[codex-prod] previous run still holding lock ($LOCK), skipping" | tee -a "$LOG"
  exit 0
fi

echo "=== Codex prod audit $(date -u -Iseconds) ===" | tee -a "$LOG"
echo "prompt: $PROMPT_FILE ($(wc -c < "$PROMPT_FILE") bytes)" | tee -a "$LOG"
echo "codex: $CODEX_BIN" | tee -a "$LOG"

if [ "${CODEX_PROD_CHECK_DRY_RUN:-0}" = "1" ]; then
  echo "dry run: would execute codex prod check" | tee -a "$LOG"
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
if [ -s "$LAST_MSG" ]; then
  {
    echo "--- last message ---"
    cat "$LAST_MSG"
  } | tee -a "$LOG"
fi
echo "=== Codex prod audit complete $(date -u -Iseconds) ===" | tee -a "$LOG"
exit "$rc"
