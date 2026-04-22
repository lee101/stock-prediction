#!/usr/bin/env bash
# Launches the leverage×ms sweep on oos2024_ensemble_gpu_fresh (15-seed).
# Output: analysis/xgbnew_ensemble_sweep/leverage_curve_freshens_20260422_1710/
exec 63>&- 2>/dev/null || true
exec 62>&- 2>/dev/null || true
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
MODELS=$(ls analysis/xgbnew_daily/oos2024_ensemble_gpu_fresh/alltrain_seed*.pkl | tr '\n' ',' | sed 's/,$//')
exec python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --blend-mode mean \
  --train-end 2024-12-31 \
  --oos-start 2025-01-02 --oos-end 2026-04-19 \
  --leverage-grid 1.0,1.25,1.5,1.75,2.0,2.5,3.0 \
  --min-score-grid 0.55,0.60,0.65,0.70,0.75,0.80,0.85 \
  --top-n-grid 1 \
  --hold-through \
  --fee-regimes deploy,stress36x \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid 0.12 \
  --skip-prob-grid 0.0 \
  --output-dir analysis/xgbnew_ensemble_sweep/leverage_curve_freshens_20260422_1710
