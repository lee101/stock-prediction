#!/usr/bin/env bash
# Watch for new downloaded symbols and auto-run preaug + hyperparam for each.
# Run this after (or alongside) download_alpaca_stocks.py to process symbols
# as they arrive. Handles new symbols that didn't exist when the initial sweep
# was launched.
#
# Usage:
#   bash scripts/watch_and_sweep.sh &
#   # or foreground with nohup:
#   nohup bash scripts/watch_and_sweep.sh > logs/watch_and_sweep.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

HP_DIR="hyperparams/chronos2"
DATA_DIR="trainingdata"
PREAUG_OUT="preaugstrategies/chronos2"
PREAUG_MIRROR="preaugstrategies/best"
REPORT_DIR="preaug_sweeps/reports/bulk"
MIN_ROWS=200
POLL_INTERVAL=30   # seconds between scans
STRATEGIES="differencing log_diff diff_norm"

mkdir -p "$REPORT_DIR" "$PREAUG_OUT" "$PREAUG_MIRROR" "$HP_DIR" logs

echo "[watch_and_sweep] Started. Polling every ${POLL_INTERVAL}s..."
echo "[watch_and_sweep] HP_DIR=$HP_DIR  DATA_DIR=$DATA_DIR  PREAUG_OUT=$PREAUG_OUT"

process_symbol() {
  local sym="$1"
  local csv="$DATA_DIR/${sym}.csv"
  local hp="$HP_DIR/${sym}.json"
  local preaug_out="$PREAUG_OUT/${sym}.json"

  # Generate default hyperparam if missing
  if [ ! -f "$hp" ]; then
    rows=$(wc -l < "$csv")
    if [ "$rows" -lt $MIN_ROWS ]; then
      echo "[watch_and_sweep] $sym: thin ($rows rows), skipping"
      return
    fi
    python3 scripts/generate_default_hyperparams.py \
      --data-dir "$DATA_DIR" \
      --hyperparam-dir "$HP_DIR" \
      --min-rows $MIN_ROWS 2>/dev/null
    [ -f "$hp" ] || { echo "[watch_and_sweep] $sym: failed to generate hyperparam, skipping"; return; }
  fi

  # Skip if preaug already done
  if [ -f "$preaug_out" ]; then
    return
  fi

  echo "[watch_and_sweep] Processing $sym..."
  python3 preaug_sweeps/evaluate_preaug_chronos.py \
    --symbols "$sym" \
    --strategies $STRATEGIES \
    --strategy-repeats 1 \
    --output-dir "$PREAUG_OUT" \
    --mirror-best-dir "$PREAUG_MIRROR" \
    --report-dir "$REPORT_DIR" \
    --frequency daily \
    2>&1 | grep -E "best=|ERROR|WARN" || true
}

while true; do
  # Find CSVs that don't yet have preaug results
  for csv_file in "$DATA_DIR"/*.csv; do
    sym="$(basename "$csv_file" .csv)"
    if [ ! -f "$PREAUG_OUT/${sym}.json" ]; then
      process_symbol "$sym"
    fi
  done

  sleep "$POLL_INTERVAL"
done
