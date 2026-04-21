#!/bin/bash
# Cross-sectional regime-dispersion gate sweep on fresh 2025H1-cutoff
# ensemble. This is the FIRST lever that looks like it might flip
# true-OOS sign on the tariff-crash regime — all 4 prior sizing/masking
# levers failed.
#
# Diagnostic (scripts/xgb_diagnose_cs_dispersion_regime.py) showed:
#   top-1/day cum return baseline:              −37.39% (201 days)
#   gate cs_iqr_ret5 <= 0.042:                  +43.26% (59 days)
#   gate cs_iqr_ret5 <= 0.045:                  +27.17% (83 days)
#   gate rolling 90d iqr<=Q50 AND skew>=Q50:    +27.91% (42 days)
#
# This sweep runs the full 30d/5d-stride windowed backtest on the fresh
# 5-seed 2025H1 held-out ensemble at base cell (lev=2.0 ms=0.60 ht=1
# vol>=0.10 dolvol>=50M) and sweeps:
#   regime_cs_iqr_max  ∈ {0.0 (off), 0.040, 0.042, 0.045, 0.048, 0.050}
#   regime_cs_skew_min ∈ {-1e9 (off), 0.0, 0.5, 1.0}
# → 6 × 4 × 2 fee_regimes = 48 cells.
#
# Pass gate: median>0 AND neg<=5/35 AND p10>−30% at deploy fees.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUTDIR=analysis/xgbnew_daily/sweep_20260421_regime_cs_dispersion
mkdir -p "$OUTDIR"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --model-paths analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed0.pkl,\
analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed7.pkl,\
analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed42.pkl,\
analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed73.pkl,\
analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed197.pkl \
  --train-start 2020-01-01 --train-end 2025-06-30 \
  --oos-start   2025-07-01 --oos-end   2026-04-20 \
  --window-days 30 --stride-days 5 \
  --leverage-grid 2.0 \
  --min-score-grid 0.60 \
  --top-n-grid 1 \
  --hold-through \
  --fee-regimes deploy,stress36x \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid 0.10 \
  --regime-cs-iqr-max-grid 0.0,0.040,0.042,0.045,0.048,0.050 \
  --regime-cs-skew-min-grid=-1e9,0.0,0.5,1.0 \
  --output-dir "$OUTDIR" \
  --verbose 2>&1 | tee "$OUTDIR/sweep.log"
