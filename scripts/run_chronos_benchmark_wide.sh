#!/bin/bash
# Run Chronos2 hyperparams benchmark for wide stock universe.
#
# Benchmarks context_length, scaler, and preaug strategies for each symbol,
# then promotes the best config to hyperparams/chronos2/{symbol}.json.
# This is the first step to MAE-calibrated stock screening.
#
# Runs benchmark-only (no LoRA), which is faster (~2-5 min/symbol).
# After benchmark, run run_chronos_lora_wide.sh to train LoRAs.
#
# Usage:
#   nohup bash scripts/run_chronos_benchmark_wide.sh > /tmp/chronos_bench_wide.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

SYMBOLS_FILE="${1:-symbol_lists/stocks_top200_v1.txt}"
FREQ="daily"
ASSET_KIND="stocks"
RUN_ID="chronos2_bench_wide_$(date +%Y%m%d_%H%M%S)"
REPORT_ROOT="reports/chronos2_pair_refits"

echo "[bench_wide] $(date -u +%FT%TZ) symbols_file=$SYMBOLS_FILE run_id=$RUN_ID"
echo

# Read symbols from file (refit script takes comma-separated)
SYMBOLS=$(grep -v '^#' "$SYMBOLS_FILE" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')

python -u scripts/refit_chronos2_pair_configs.py \
    --symbols "$SYMBOLS" \
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
    --preaug-strategies "baseline,percent_change,rolling_norm" \
    --skip-lora \
    --skip-promote \
    --execute

echo "[bench_wide] Done $(date -u +%FT%TZ)"
