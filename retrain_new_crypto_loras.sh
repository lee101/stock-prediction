#!/bin/bash
# Retrain Chronos2 LoRA adapters for new crypto symbols needed by the crypto10+ datasets.
# Uses scripts/run_crypto_lora_batch.py which sweeps preaugs, context lengths, and LRs.
#
# Default sweep grid (from run_crypto_lora_batch.py defaults):
#   preaugs:         baseline, percent_change, log_returns
#   context_lengths: 128, 256
#   learning_rates:  5e-5, 1e-4
#   num_steps:       1000
#   lora_r:          16
#
# Per symbol this is 3 preaugs x 2 ctx x 2 lr = 12 training runs.
# Total: 5 symbols x 12 = 60 training runs.
#
# Data: trainingdatahourly/crypto/{SYMBOL}.csv
# Results: hyperparams/crypto_lora_sweep/new_crypto_batch/
# Adapters: chronos2_finetuned/
set -euo pipefail

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

SYMBOLS="DOGEUSD,LINKUSD,ADAUSD,UNIUSD,AAVEUSD"
RUN_ID="new_crypto_$(date +%Y%m%d)"
RESULTS_DIR="hyperparams/crypto_lora_sweep/new_crypto_batch"

echo "=== Chronos2 LoRA Training for New Crypto Symbols ==="
echo "Symbols: ${SYMBOLS}"
echo "Run ID:  ${RUN_ID}"
echo "Results: ${RESULTS_DIR}"
echo ""

python scripts/run_crypto_lora_batch.py \
    --run-id "${RUN_ID}" \
    --symbols "${SYMBOLS}" \
    --data-root trainingdatahourly/crypto \
    --output-root chronos2_finetuned \
    --results-dir "${RESULTS_DIR}" \
    --preaugs "baseline,percent_change,log_returns" \
    --context-lengths "128,256" \
    --learning-rates "5e-5,1e-4" \
    --num-steps 1000 \
    --prediction-length 24 \
    --lora-r 16

echo ""
echo "=== Batch Summary ==="
if [ -f "${RESULTS_DIR}/${RUN_ID}_batch_summary.csv" ]; then
    echo "CSV: ${RESULTS_DIR}/${RUN_ID}_batch_summary.csv"
    column -t -s, "${RESULTS_DIR}/${RUN_ID}_batch_summary.csv" | head -20
fi
echo ""
echo "JSON results in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"*.json 2>/dev/null | tail -10
echo ""
echo "LoRA adapters in: chronos2_finetuned/"
ls -d chronos2_finetuned/*DOGE* chronos2_finetuned/*LINK* chronos2_finetuned/*ADA* chronos2_finetuned/*UNI* chronos2_finetuned/*AAVE* 2>/dev/null | tail -20
