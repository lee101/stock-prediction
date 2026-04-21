#!/usr/bin/env bash
# Band-pass vol_20d filter sweep on the fresh held-out ensemble.
#
# True-OOS 2025-07 → 2026-04 returned 0/192 positive-median cells on the
# standard grid (project_xgb_true_oos_no_edge_2026_04_21.md). Hypothesis:
# the tariff-crash window (2025-09 onwards) hammered high-vol names; a
# vol ceiling might recover edge by keeping the model trained on the full
# universe but picking only moderate-vol names at inference.
#
# Grid: lev {1.0, 1.5, 2.0} × ms {0.55, 0.60, 0.65}
#     × inf_min_vol {0.10, 0.15} × inf_max_vol {0.30, 0.40, 0.50, 0.60, 0.80}
#     × fee {deploy, stress36x}
#   = 3 × 3 × 2 × 5 × 2 = 180 cells
#
# Deployed baseline: lev=2.0 ms=0.85 vol_20d∈[0.12, inf]. We center the ms
# grid where the fresh ensemble actually scores (max ~0.68).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260421_maxvol_band"
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
  --leverage-grid "1.0,1.5,2.0" \
  --min-score-grid "0.55,0.60,0.65" \
  --top-n-grid "1" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10,0.15" \
  --inference-max-vol-grid "0.30,0.40,0.50,0.60,0.80" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
