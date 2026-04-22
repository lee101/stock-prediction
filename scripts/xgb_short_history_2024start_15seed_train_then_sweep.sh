#!/usr/bin/env bash
# NOVEL DIRECTION (2026-04-22 hourly monitor): short-history 15-seed XGB.
# Hypothesis: pre-2024 training data may be irrelevant in current
# tariff-crash regime. Train on 2024-01-01 → 2026-02-28 only (26 months,
# vs the standard 6+ years from 2020-01-01) and re-evaluate on the same
# 2026-03-01 → 2026-04-20 heldout used by retrain_through_0228 sweeps.
#
# All other refuted experiments use --train-start 2020-01-01. This is
# the first true regime-aware short-history test at 15-seed bonferroni.
#
# Hyperparameters held identical to the LIVE classifier
# (n_estimators=400 max_depth=5 learning_rate=0.03) so the only delta
# is training-window length.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/short_history_2024start_through_0228_15seed"
mkdir -p "$OUT_DIR"

echo "[$(date -u +%H:%M:%SZ)] training short-history 15-seed (2024-01-01 → 2026-02-28)"
python -m xgbnew.train_alltrain_ensemble \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2024-01-01 --train-end 2026-02-28 \
  --min-dollar-vol 0 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,42,73,197 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --out-dir "$OUT_DIR" \
  --verbose 2>&1 | tee -a "$OUT_DIR/train.log"

echo "[$(date -u +%H:%M:%SZ)] training done; launching sweep on heldout"

SWEEP_DIR="analysis/xgbnew_daily/sweep_20260422_short_history_2024start_15seed"
mkdir -p "$SWEEP_DIR"

MODELS=$(ls "$OUT_DIR"/alltrain_seed*.pkl | paste -sd,)

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2024-01-01 --train-end 2026-02-28 \
  --oos-start 2026-03-01 --oos-end 2026-04-20 \
  --window-days 5 --stride-days 2 \
  --leverage-grid "1.0,1.5,2.0" \
  --min-score-grid "0.55,0.60,0.65,0.70,0.75,0.80,0.85" \
  --top-n-grid "1,2" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10,0.12" \
  --output-dir "$SWEEP_DIR" \
  --verbose 2>&1 | tee "$SWEEP_DIR/stdout.log"

echo "[$(date -u +%H:%M:%SZ)] short-history 15-seed train+sweep complete -> $SWEEP_DIR"
