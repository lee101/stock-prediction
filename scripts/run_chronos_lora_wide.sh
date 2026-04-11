#!/bin/bash
# Run per-stock Chronos2 LoRA training for wide stock universe.
#
# Reads top-N symbols from screener_latest.json (or --symbols-file),
# then runs refit_chronos2_pair_configs.py to:
#   1. Select best preaug strategy per symbol
#   2. Benchmark hyperparams (context length, batch size, scaler)
#   3. Train per-stock LoRA (r=8/16, lr=1e-5/5e-5, 1000 steps)
#   4. Promote LoRAs with >5% MAE improvement to hyperparams/chronos2/
#
# Usage:
#   nohup bash scripts/run_chronos_lora_wide.sh > /tmp/chronos_lora_wide.log 2>&1 &
#   nohup bash scripts/run_chronos_lora_wide.sh --symbols-file symbol_lists/stocks_wide_1000_v1.txt > /tmp/chronos_lora_wide.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

SCREENER_JSON="${1:-analysis/chronos_screener/screener_latest.json}"
TOP_N="${2:-100}"           # number of top screener picks to train LoRAs for
FREQ="daily"
ASSET_KIND="stocks"
RUN_ID="chronos2_lora_wide_$(date +%Y%m%d_%H%M%S)"
REPORT_ROOT="reports/chronos2_pair_refits"

# Resolve symbols: from screener JSON if available, else from symbols file
if [ -f "$SCREENER_JSON" ]; then
    echo "[lora_wide] Reading top $TOP_N symbols from $SCREENER_JSON"
    SYMBOLS=$(python3 -c "
import json, sys
d = json.load(open('$SCREENER_JSON'))
candidates = d.get('top_candidates', d.get('all_results', []))
# take top N by predicted_return_pct, filter out crypto/tiny stocks
syms = [c['symbol'] for c in candidates if c['last_close'] >= 3.0][:$TOP_N]
print(','.join(syms))
" 2>/dev/null)
    if [ -z "$SYMBOLS" ]; then
        echo "[lora_wide] Failed to parse screener JSON, falling back to symbols file"
        SYMBOLS_ARG="--symbols"
        SYMBOLS_VAL="$(head -50 symbol_lists/stocks_wide_1000_v1.txt | tr '\n' ',' | sed 's/,$//')"
    else
        SYMBOLS_ARG="--symbols"
        SYMBOLS_VAL="$SYMBOLS"
    fi
else
    echo "[lora_wide] No screener JSON found at $SCREENER_JSON"
    SYMBOLS_ARG="--symbols"
    SYMBOLS_VAL="$(head -50 symbol_lists/stocks_wide_1000_v1.txt | tr '\n' ',' | sed 's/,$//')"
fi

echo "[lora_wide] run_id=$RUN_ID freq=$FREQ symbols=${SYMBOLS_VAL:0:80}..."
echo

python -u scripts/refit_chronos2_pair_configs.py \
    $SYMBOLS_ARG "$SYMBOLS_VAL" \
    --frequency "$FREQ" \
    --asset-kind "$ASSET_KIND" \
    --run-id "$RUN_ID" \
    --report-root "$REPORT_ROOT" \
    --pipeline-backend "cutechronos" \
    --torch-compile \
    --torch-dtype "float32" \
    --search-method "direct" \
    --context-lengths "128,256,512" \
    --batch-sizes "32,64" \
    --aggregations "median" \
    --sample-counts "0" \
    --scalers "none,meanstd" \
    --multivariate-modes "true" \
    --preaug-strategies "baseline,percent_change,log_returns,rolling_norm,detrending" \
    --lora-preaugs "baseline,rolling_norm,detrending" \
    --lora-context-lengths "128,256" \
    --lora-learning-rates "1e-5,5e-5" \
    --lora-rs "8,16" \
    --lora-batch-size 32 \
    --lora-num-steps 1000 \
    --lora-improvement-threshold 5.0 \
    --promote-selection-strategy "stable_family" \
    --promote-metric "val_mae_percent" \
    --execute

echo "[lora_wide] Done. Run ID: $RUN_ID"
