#!/usr/bin/env bash
# Blend 15-seed MuonMLP cutoff=0228 + 15-seed cutoff=0320 = 30 models.
# Evaluate on late subfold 2026-03-21→04-20 (where 0320 alone got 0/432
# pos-med and 0228 alone on this subfold got 0/432 pos-med).
# Hypothesis: blend across cutoffs might surface calibration signal neither
# cutoff has on its own.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_muonmlp_blend_cutoffs_late"
mkdir -p "$OUT_DIR"

M1=$(ls analysis/xgbnew_daily/retrain_through_2026_02_28_mlp_muon/alltrain_seed*.pkl)
M2=$(ls analysis/xgbnew_daily/retrain_through_2026_03_20_mlp_muon/alltrain_seed*.pkl)
MODELS=$(echo -e "$M1\n$M2" | paste -sd,)
N=$(echo -e "$M1\n$M2" | wc -l)
echo "[muonmlp_blend_cutoffs_late] N=$N (15+15) -> $OUT_DIR"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2026-03-20 \
  --oos-start 2026-03-21 --oos-end 2026-04-20 \
  --window-days 5 --stride-days 2 \
  --leverage-grid "1.0,2.0,3.0" \
  --min-score-grid "0.50,0.55,0.60" \
  --top-n-grid "1,2" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --regime-cs-iqr-max-grid=0,0.042 \
  --regime-cs-skew-min-grid=-1000000000,0.0 \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
