#!/usr/bin/env bash
# Per-pick 1/vol_20d sizing sweep on the fresh held-out ensemble.
#
# Context: the 80-cell SPY-vol-target sweep
# (scripts/xgb_sweep_20260421_vol_target.sh) proved the SPY lever is
# INACTIVE on this OOS — SPY 20d vol tops out at 20.4% so
# min(1, target/realised) is ≥0.98 on 90% of days. The tariff crash is
# cross-sectional (growth names hit, index calm), not index-wide.
#
# Per-pick inv-vol reads each pick's own `vol_20d` and scales its
# leverage by clip(target_ann / max(vol_20d, floor), 1/cap, cap). A
# high-vol pick gets down-levered; a low-vol pick gets up-levered —
# so this lever CAN respond to a cross-sectional regime that SPY cannot.
#
# Grid: lev {1.5, 2.0} × ms {0.55, 0.60} × max_vol {0.30, 0.40}
#     × inv_vol_target {0.0, 0.15, 0.20, 0.25, 0.30}
#     × fee {deploy, stress36x}
#   = 2 × 2 × 2 × 5 × 2 = 80 cells
#
# Hold fixed: inf_min_vol=0.10 (deployed floor), hold_through=True,
# inv_vol_floor=0.05 (ann-vol denominator floor), inv_vol_cap=3.0.
#
# Success criterion: cell with (a) positive median AND (b) ≤ 5/38 neg
# windows AND (c) p10 > −15% at both fee regimes. That threshold wasn't
# reachable by band-pass or SPY-vol-target alone.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260421_inv_vol"
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
  --inv-vol-target-grid "0.0,0.15,0.20,0.25,0.30" \
  --inv-vol-floor 0.05 \
  --inv-vol-cap 3.0 \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
