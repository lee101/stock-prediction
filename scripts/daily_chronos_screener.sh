#!/bin/bash
# Daily Chronos2 wide-stock screener.
#
# Runs compiled Chronos2 (cute_compiled_fp32) on 846 stocks,
# writes screener_latest.json with top 30 buy signals.
# Run this once per trading day (e.g., 30 min before market open).
#
# Usage:
#   bash scripts/daily_chronos_screener.sh
#   nohup bash scripts/daily_chronos_screener.sh > /tmp/daily_screener.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

SYMBOLS_FILE="symbol_lists/stocks_wide_1000_v1.txt"
OUTPUT_DIR="analysis/chronos_screener"
BACKEND="${CHRONOS_BACKEND:-cute_compiled_fp32}"   # override: CHRONOS_BACKEND=cute_fp32
CONTEXT="${CHRONOS_CONTEXT:-256}"
TOP_N="${CHRONOS_TOP_N:-30}"

echo "[daily_screener] $(date -u +%FT%TZ) backend=$BACKEND top_n=$TOP_N"

python -u scripts/chronos_wide_screener.py \
    --symbols-file "$SYMBOLS_FILE" \
    --backend "$BACKEND" \
    --context-length "$CONTEXT" \
    --top-n "$TOP_N" \
    --output-dir "$OUTPUT_DIR" \
    --data-root trainingdata

echo "[daily_screener] done $(date -u +%FT%TZ)"
echo "[daily_screener] Results: $OUTPUT_DIR/screener_latest.json"
echo
echo "TOP SIGNALS:"
python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/screener_latest.json'))
print(f'Screened {d[\"symbols_screened\"]} stocks, generated {d[\"generated_at\"]}')
print()
print(f'{\"#\":<3} {\"Symbol\":<8} {\"LastClose\":>10} {\"PredClose\":>10} {\"PredRet%\":>9} {\"High\":>8} {\"Low\":>8}')
print('-'*60)
for i, c in enumerate(d['top_candidates'], 1):
    print(f'{i:<3} {c[\"symbol\"]:<8} {c[\"last_close\"]:>10.2f} {c[\"predicted_close\"]:>10.2f} {c[\"predicted_return_pct\"]:>+9.2f}% {c[\"predicted_high\"]:>8.2f} {c[\"predicted_low\"]:>8.2f}')
" 2>/dev/null
