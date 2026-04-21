#!/usr/bin/env bash
# SPY-based vol-target sweep on the fresh held-out ensemble.
#
# Context: the band-pass MAX-vol sweep (scripts/xgb_sweep_20260421_maxvol_band.sh)
# surfaced 21/180 positive-median cells on true-OOS 2025-07 → 2026-04, but
# EVERY one of those cells had ≥13/38 neg windows and p10 ≤ −22%. The
# positive median is real; the downside is catastrophic. Hypothesis: if
# we scale daily allocation DOWN when SPY's realised vol is high (the
# regime-shrink lever that the 2025-09+ tariff crash triggers), maybe
# we can drop the neg-window count while keeping the upside.
#
# Grid: lev {1.5, 2.0} × ms {0.55, 0.60} × max_vol {0.30, 0.40}
#     × vol_target_ann {0.0, 0.10, 0.15, 0.20, 0.25}
#     × fee {deploy, stress36x}
#   = 2 × 2 × 2 × 5 × 2 = 80 cells
#
# We hold inf_min_vol=0.10 fixed (matches deployed vol floor) and
# hold_through=True (matches deployed). Focus is exclusively on the
# vol-target knob.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260421_vol_target"
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
  --leverage-grid "1.5,2.0" \
  --min-score-grid "0.55,0.60" \
  --top-n-grid "1" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --inference-max-vol-grid "0.30,0.40" \
  --vol-target-ann-grid "0.0,0.10,0.15,0.20,0.25" \
  --spy-csv trainingdata/SPY.csv \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
