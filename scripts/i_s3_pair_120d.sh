#!/usr/bin/env bash
# 120d eval for I_s3 solo + I_s3+AD_s4 pair on screened32_full_val.bin
# Context: 50d crash-val memo numbers (project_rl_i_s3_guard_robust_candidate,
#   project_rl_i_s3_ad_s4_pair_curve):
#   I_s3 solo lev=2 guard-on:      med +12.77 p10 -3.56 neg 18/81
#   I_s3+AD_s4 pair lev=4 guard-on: med +21.59 (~+12.95%/mo) [best realizable RL]
# Test whether those numbers generalize to 120d windows on full_val (194 windows).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_120d_baseline"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_full_val.bin

run_eval() {
  local name="$1"
  shift
  local out_json="$OUT_DIR/${name}.json"
  [ -f "$out_json" ] && { echo "SKIP $name"; return; }
  echo "=== $name ==="
  python -m pufferlib_market.evaluate_holdout \
    --data-path "$DATA" \
    --eval-hours 120 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --decision-lag 2 --disable-shorts --deterministic \
    --death-spiral-tolerance-bps 50.0 \
    --death-spiral-overnight-tolerance-bps 500.0 \
    --death-spiral-stale-after-bars 8 \
    --out "$out_json" \
    "$@" 2>&1 | tee "$OUT_DIR/${name}.log" | grep -E "median_total|p10_total|negative_w|median_max_drawdown" | head -5
}

# I_s3 solo — lev 1, 2, 3
for lev in 1.0 2.0 3.0; do
  run_eval "i_s3_solo_lev${lev/./p}" \
    --checkpoint "$BASE/I_s3.pt" --max-leverage "$lev"
done

# I_s3 + AD_s4 pair — lev 1, 2, 4 (strict-dom curve from memo)
for lev in 1.0 2.0 4.0; do
  run_eval "i_s3_ad_s4_pair_lev${lev/./p}" \
    --checkpoint "$BASE/I_s3.pt" \
    --extra-checkpoints "$BASE/AD_s4.pt" \
    --max-leverage "$lev"
done

echo "DONE."
