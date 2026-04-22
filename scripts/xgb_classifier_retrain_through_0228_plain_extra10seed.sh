#!/usr/bin/env bash
# Add 10 extra seeds (1,2,3,4,5,6,8,9,10,11) to the PLAIN 5-seed
# retrain_through_2026_02_28_ensemble (classifier, n=400 d=5 lr=0.03,
# NO dispersion features). Gives clean 15-seed bonferroni that we can
# H2H against the 5-seed +23.61%/mo stress36x memory claim.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/retrain_through_2026_02_28_ensemble"
mkdir -p "$OUT_DIR"

python -m xgbnew.train_alltrain_ensemble \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2020-01-01 --train-end 2026-02-28 \
  --min-dollar-vol 0 \
  --seeds 1,2,3,4,5,6,8,9,10,11 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --out-dir "$OUT_DIR" \
  --verbose 2>&1 | tee -a "$OUT_DIR/train.log"
