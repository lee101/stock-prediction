#!/usr/bin/env bash
# One-off: re-eval short-history 2024-start 15-seed on LATE SUBFOLD
# (2026-03-21 → 2026-04-20 only), to test whether its small deploy-fee
# positive median on the full heldout is a post-crash recovery-rally
# artifact like MuonMLP retrain-through-0228 was (memory:
# project_xgb_muonmlp_retrain_through_0228_candidate.md).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

MODELS=$(ls analysis/xgbnew_daily/short_history_2024start_through_0228_15seed/alltrain_seed*.pkl | paste -sd,)
SWEEP_DIR=analysis/xgbnew_daily/sweep_20260422_short_history_2024start_15seed_lateonly
mkdir -p "$SWEEP_DIR"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2024-01-01 --train-end 2026-02-28 \
  --oos-start 2026-03-21 --oos-end 2026-04-20 \
  --window-days 5 --stride-days 2 \
  --leverage-grid "1.0,1.5,2.0" \
  --min-score-grid "0.55,0.60,0.65,0.70,0.75,0.80,0.85" \
  --top-n-grid "1,2" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10,0.12" \
  --output-dir "$SWEEP_DIR" \
  --verbose
