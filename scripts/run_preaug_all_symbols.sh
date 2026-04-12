#!/usr/bin/env bash
# Run preaug strategy sweep (differencing, log_diff, diff_norm) for all symbols
# that have both a CSV in trainingdata/ and a hyperparam config.
#
# Runs sequentially (single GPU). Skips already-done symbols.
# Estimated: ~30s/symbol × 3 strategies ≈ 1.5 min/symbol
#
# Usage:
#   bash scripts/run_preaug_all_symbols.sh            # all available symbols
#   bash scripts/run_preaug_all_symbols.sh 0 500      # symbols 0-499
#   bash scripts/run_preaug_all_symbols.sh 500 1000   # symbols 500-999
#   STRATEGIES="differencing log_diff" bash scripts/run_preaug_all_symbols.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

SKIP="${1:-0}"
LIMIT="${2:-0}"
STRATEGIES="${STRATEGIES:-differencing log_diff diff_norm}"
HP_DIR="hyperparams/chronos2"
DATA_DIR="trainingdata"
OUT_DIR="preaugstrategies/chronos2"
MIRROR_DIR="preaugstrategies/best"
REPORT_DIR="preaug_sweeps/reports/bulk"

mkdir -p "$REPORT_DIR" "$OUT_DIR" "$MIRROR_DIR"

# Build list of symbols that have both CSV and hyperparam config
mapfile -t ALL_SYMS < <(
  for hp_file in "$HP_DIR"/*.json; do
    sym="$(basename "$hp_file" .json)"
    if [ -f "$DATA_DIR/${sym}.csv" ]; then
      # Skip if already done (has output in preaugstrategies/chronos2/)
      if [ ! -f "$OUT_DIR/${sym}.json" ]; then
        echo "$sym"
      fi
    fi
  done | sort
)

TOTAL=${#ALL_SYMS[@]}
echo "Total pending symbols: $TOTAL"

# Apply skip/limit
if [ "$SKIP" -gt 0 ]; then
  ALL_SYMS=("${ALL_SYMS[@]:$SKIP}")
fi
if [ "$LIMIT" -gt 0 ]; then
  ALL_SYMS=("${ALL_SYMS[@]:0:$LIMIT}")
fi

echo "Processing ${#ALL_SYMS[@]} symbols (skip=$SKIP limit=$LIMIT)"
echo "Strategies: $STRATEGIES"
echo ""

DONE=0
TOTAL_THIS_RUN=${#ALL_SYMS[@]}

for sym in "${ALL_SYMS[@]}"; do
  DONE=$((DONE + 1))
  echo "[${DONE}/${TOTAL_THIS_RUN}] $sym"
  python3 preaug_sweeps/evaluate_preaug_chronos.py \
    --symbols "$sym" \
    --strategies $STRATEGIES \
    --strategy-repeats 1 \
    --output-dir "$OUT_DIR" \
    --mirror-best-dir "$MIRROR_DIR" \
    --report-dir "$REPORT_DIR" \
    --frequency daily \
    2>&1 | grep -E "^\[|best=|ERROR|WARN" || true
done

echo ""
echo "=== Done: $DONE symbols processed ==="
echo "Configs in $OUT_DIR: $(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l)"
