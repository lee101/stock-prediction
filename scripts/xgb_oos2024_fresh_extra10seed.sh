#!/usr/bin/env bash
# Train 10 extra seeds to convert oos2024_ensemble_gpu_fresh (5-seed) into
# a clean 15-seed ensemble. Enables 15-seed bonferroni re-test of the
# cs-dispersion gate (the only surviving cross-fold TRUE-OOS lever).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/oos2024_ensemble_gpu_fresh"
mkdir -p "$OUT_DIR"

python -m xgbnew.train_alltrain_ensemble \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2020-01-01 --train-end 2024-12-31 \
  --min-dollar-vol 0 \
  --seeds 1,2,3,4,5,6,8,9,10,11 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --out-dir "$OUT_DIR" \
  --verbose 2>&1 | tee -a "$OUT_DIR/train.log"
