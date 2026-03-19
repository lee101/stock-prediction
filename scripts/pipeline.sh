#!/usr/bin/env bash
set -euo pipefail

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
export PYTHONPATH="$PWD:$PWD/PufferLib:${PYTHONPATH:-}"

RUN_ID="bnb_chronos2_lora_proof_20260319_015620"
RUN_DIR="analysis/remote_runs/${RUN_ID}"

mkdir -p "${RUN_DIR}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ${RUN_ID}"

python -u scripts/run_crypto_lora_batch.py \
  --run-id "${RUN_ID}" \
  --symbols BNBUSDT \
  --data-root trainingdatahourlybinance \
  --output-root chronos2_finetuned \
  --results-dir "${RUN_DIR}/lora_results" \
  --preaugs baseline,percent_change,log_returns,differencing,robust_scaling \
  --context-lengths 128,256 \
  --learning-rates 5e-5,1e-4 \
  --num-steps 400 \
  --prediction-length 24 \
  --lora-r 16

python -u scripts/promote_chronos2_lora_reports.py \
  --report-dir "${RUN_DIR}/lora_results" \
  --output-dir hyperparams/chronos2/hourly \
  --symbols BNBUSDT \
  --run-id "${RUN_ID}" \
  --metric val_mae_percent

python -u scripts/build_hourly_forecast_caches.py \
  --symbols BNBUSDT \
  --data-root trainingdatahourlybinance \
  --forecast-cache-root "${RUN_DIR}/forecast_cache" \
  --horizons 1,24 \
  --lookback-hours 5000 \
  --output-json "${RUN_DIR}/forecast_mae.json" \
  --force-rebuild

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] completed ${RUN_ID}"
