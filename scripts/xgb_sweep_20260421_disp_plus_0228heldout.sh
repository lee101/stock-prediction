#!/usr/bin/env bash
# Cross-sectional dispersion gate ON TOP of retrain_through_2026_02_28
# ensemble's held-out window (2026-03-01 → 2026-04-20).
#
# Hypothesis: dispersion gate (gate-only, no retraining) filtered out
# 2025-H1 and oos2024 crash windows cleanly; stacked with a crash-aware
# RETRAINED ensemble it may also lift median past the deploy gate.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260421_disp_plus_0228heldout"
mkdir -p "$OUT_DIR"

MODELS="analysis/xgbnew_daily/retrain_through_2026_02_28_ensemble/alltrain_seed0.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_02_28_ensemble/alltrain_seed7.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_02_28_ensemble/alltrain_seed42.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_02_28_ensemble/alltrain_seed73.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_02_28_ensemble/alltrain_seed197.pkl"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2026-02-28 \
  --oos-start 2026-03-01 --oos-end 2026-04-20 \
  --window-days 5 --stride-days 2 \
  --leverage-grid "1.0,1.5,2.0" \
  --min-score-grid "0.55,0.60,0.65" \
  --top-n-grid "1" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --regime-cs-iqr-max-grid=0,0.042,0.06 \
  --regime-cs-skew-min-grid=-1000000000,0.0,1.0 \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
