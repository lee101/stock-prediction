#!/usr/bin/env bash
# Retrain the held-out OOS ensembles on FRESH-CSV features.
#
# Context: the oos2024_ensemble_gpu, oos2025h1_ensemble_gpu, and
# oos2025_ensemble_gpu bundles were trained 2026-04-20 at 06:48-06:50 UTC,
# roughly 14 hours before the stale-CSV loader fix landed
# (commit debe551d 2026-04-20 20:50 UTC). Applying those ensembles to
# features now built from fresh CSVs creates a distribution mismatch
# that caps blended scores at ~0.84 on the true-OOS window — live
# gate ms=0.85 never fires.
#
# Fix: retrain all three cutoff bundles with the fresh loader. Same
# champion hyperparameters, same 5-seed set, new directories with
# `_fresh` suffix so the pre-fix pickles stay available for audit.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

SYMS="symbol_lists/stocks_wide_1000_v1.txt"
SEEDS="0,7,42,73,197"
HP=(--n-estimators 400 --max-depth 5 --learning-rate 0.03 --device cuda)

run_one() {
  local tag="$1" tr_end="$2"
  local out="analysis/xgbnew_daily/${tag}_ensemble_gpu_fresh"
  echo "[$(date -u +%H:%M:%SZ)] training ${tag} (train_end=${tr_end}) -> ${out}"
  python -m xgbnew.train_alltrain_ensemble \
    --symbols-file "$SYMS" \
    --data-root trainingdata \
    --train-start 2020-01-01 \
    --train-end "$tr_end" \
    --min-dollar-vol 0 \
    --seeds "$SEEDS" \
    "${HP[@]}" \
    --out-dir "$out" \
    --verbose
}

run_one oos2024   2024-12-31
run_one oos2025h1 2025-06-30
run_one oos2025   2025-12-31

echo "[$(date -u +%H:%M:%SZ)] all three fresh-CSV held-out ensembles trained"
