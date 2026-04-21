#!/bin/bash
# Compare disp-feature ensemble vs baseline on same oos2024 cutoff.
# Baseline: analysis/xgbnew_daily/oos2024_ensemble_gpu_fresh/ (14 features)
# Disp:     analysis/xgbnew_daily/oos2024_ensemble_disp_gpu_fresh/ (16 features)
# Same seeds (0,7,42,73,197), same train window 2020 → 2024-12-31.
# OOS: 2025-01-01 → 2026-04-20 (15 months, 59 windows @ 30d/stride-5d).
#
# Run both through sweep_ensemble_grid with the deployed-style config
# (lev=2.0 ms=0.60 ht=1 vol>=0.10 dolvol>=50M) to directly compare
# frontier movement.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

for variant in fresh disp_gpu_fresh; do
    case "$variant" in
        fresh) ENS=oos2024_ensemble_gpu_fresh ;;
        disp_gpu_fresh) ENS=oos2024_ensemble_disp_gpu_fresh ;;
    esac
    OUTDIR=analysis/xgbnew_daily/sweep_20260421_${variant}_vs
    mkdir -p "$OUTDIR"
    echo "=== $variant ($ENS) ==="
    python -m xgbnew.sweep_ensemble_grid \
      --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
      --data-root trainingdata \
      --model-paths analysis/xgbnew_daily/${ENS}/alltrain_seed0.pkl,\
analysis/xgbnew_daily/${ENS}/alltrain_seed7.pkl,\
analysis/xgbnew_daily/${ENS}/alltrain_seed42.pkl,\
analysis/xgbnew_daily/${ENS}/alltrain_seed73.pkl,\
analysis/xgbnew_daily/${ENS}/alltrain_seed197.pkl \
      --train-start 2020-01-01 --train-end 2024-12-31 \
      --oos-start   2025-01-01 --oos-end   2026-04-20 \
      --window-days 30 --stride-days 5 \
      --leverage-grid 1.0,1.5,2.0 \
      --min-score-grid 0.55,0.60,0.65,0.70,0.75,0.80 \
      --top-n-grid 1 \
      --hold-through \
      --fee-regimes deploy,stress36x \
      --min-dollar-vol 50000000 \
      --inference-min-vol-grid 0.10 \
      --output-dir "$OUTDIR" \
      --verbose 2>&1 | tee "$OUTDIR/sweep.log"
done
