#!/usr/bin/env bash
# Train 15-seed goodness-weighted XGB ranker at retrain-through-0228 cutoff.
# Weight each training-day group by max |target_oc| (clip 5%): emphasize
# high-dispersion days during training so the ranker cares more about
# days where its pick ordering actually generates PnL.
#
# Paired with the +disp big ranker + big classifier runs — goal is to see
# whether GW sample weighting stacks on top of retrain-through-0228 lever
# to push stress36x median past +27%/mo on 50d heldout.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/retrain_through_0228_ensemble_xgbrank_gw"
mkdir -p "$OUT_DIR"

python -m xgbnew.train_ensemble_family \
  --family xgb_rank \
  --device cuda \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --train-start 2020-01-01 --train-end 2026-02-28 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,42,73,197 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --ranker-deciles 10 \
  --ranker-sample-weight abs_target \
  --ranker-sample-weight-clip 0.05 \
  --include-dispersion \
  --out-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/train.log"
