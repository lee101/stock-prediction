#!/usr/bin/env bash
# True-OOS frontier sweep on the FRESH held-out ensemble (train 2020 →
# 2025-06-30, OOS 2025-07 → 2026-04-20 covers post-H1 + tariff crash).
# Pure TRUE-OOS — the fresh alltrain ensemble is in-sample on this window,
# the oos2024/2025 cutoffs were trained pre stale-CSV-fix (see
# `project_oos_ensembles_trained_pre_stale_fix.md`).
#
# Post-fix blended score caps at ~0.68, so the ms grid is centered in
# the 0.50-0.65 band — NOT 0.85 like the pre-fix deploy gate.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260421_fresh_heldout"
mkdir -p "$OUT_DIR"

MODELS="analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed0.pkl"
MODELS+=",analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed7.pkl"
MODELS+=",analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed42.pkl"
MODELS+=",analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed73.pkl"
MODELS+=",analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed197.pkl"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2025-06-30 \
  --oos-start 2025-07-01 --oos-end 2026-04-20 \
  --window-days 14 --stride-days 5 \
  --leverage-grid "1.0,1.5,2.0,2.5" \
  --min-score-grid "0.52,0.55,0.58,0.60,0.62,0.65" \
  --top-n-grid "1" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10,0.12,0.15,0.20" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
