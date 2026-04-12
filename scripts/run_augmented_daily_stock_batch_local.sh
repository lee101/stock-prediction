#!/usr/bin/env bash
set -euo pipefail

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

SYMBOLS="AAPL,MSFT,NVDA,GOOG,META,TSLA,AMZN,JPM,V,PLTR"
WORK_ROOT="analysis/augmented_daily_stock_runs"
DATA_ROOT="pufferlib_market/data/augmented_daily_stock"
CKPT_ROOT="pufferlib_market/checkpoints/augmented_daily_stock"
WANDB_PROJECT="${WANDB_PROJECT:-stock}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-15000000}"
VAL_DAYS="${VAL_DAYS:-120}"
NUM_ENVS="${NUM_ENVS:-128}"

run_one() {
  local run_name="$1"
  shift
  echo "[$(date -u +%FT%TZ)] starting ${run_name}"
  python -u scripts/run_augmented_daily_stock_pipeline.py \
    --run-name "${run_name}" \
    --symbols "${SYMBOLS}" \
    --work-root "${WORK_ROOT}" \
    --data-output-root "${DATA_ROOT}" \
    --checkpoint-root "${CKPT_ROOT}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-mode online \
    --val-days "${VAL_DAYS}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-envs "${NUM_ENVS}" \
    "$@"
  echo "[$(date -u +%FT%TZ)] finished ${run_name}"
}

run_one "stocks10h_baseline_s42" \
  --offsets 0 \
  --seed 42

run_one "stocks10h_shift4_s42" \
  --offsets 0,1,2,3 \
  --seed 42

run_one "stocks10h_shift4_s123" \
  --offsets 0,1,2,3 \
  --seed 123
