#!/usr/bin/env bash
# Per-member memo showed D_s16, D_s42, D_s57 are "guard-HELPED" (Δp10 +3-9pp).
# Test their 120d SOLO performance at the champion's lev=2.5 + tp=0.05 cell.
# Then try pairs of each with I_s3 to see if diversification works (unlike AD_s4/I_s32).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_120d_baseline/guard_helped"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_full_val.bin

GUARD_ARGS=(
  --death-spiral-tolerance-bps 50.0
  --death-spiral-overnight-tolerance-bps 500.0
  --death-spiral-stale-after-bars 8
)

run_solo() {
  local name="$1"
  local ckpt="$2"
  local out_json="$OUT_DIR/${name}.json"
  [ -f "$out_json" ] && { echo "SKIP $name"; return; }
  echo "=== $name SOLO ==="
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$ckpt" \
    --data-path "$DATA" \
    --eval-hours 120 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --decision-lag 2 --disable-shorts --deterministic \
    --max-leverage 2.5 --min-top-prob 0.05 \
    "${GUARD_ARGS[@]}" \
    --out "$out_json" \
    2>&1 | tee "$OUT_DIR/${name}.log" \
    | grep -E "median_total|p10_total|negative_w|median_max_drawdown|candidate_window" | head -6
}

run_pair() {
  local name="$1"
  local extra="$2"
  local out_json="$OUT_DIR/${name}.json"
  [ -f "$out_json" ] && { echo "SKIP $name"; return; }
  echo "=== $name PAIR (I_s3 + $(basename "$extra" .pt)) ==="
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$BASE/I_s3.pt" \
    --extra-checkpoints "$extra" \
    --data-path "$DATA" \
    --eval-hours 120 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --decision-lag 2 --disable-shorts --deterministic \
    --max-leverage 2.5 --min-top-prob 0.05 \
    "${GUARD_ARGS[@]}" \
    --out "$out_json" \
    2>&1 | tee "$OUT_DIR/${name}.log" \
    | grep -E "median_total|p10_total|negative_w|median_max_drawdown|candidate_window" | head -6
}

# Solos at champion cell
run_solo "d_s16_solo_lev2p5_tp0p05" "$BASE/D_s16.pt"
run_solo "d_s42_solo_lev2p5_tp0p05" "$BASE/D_s42.pt"
run_solo "d_s57_solo_lev2p5_tp0p05" "$BASE/D_s57.pt"

# I_s3 paired with each at champion cell
run_pair "i_s3_pair_d_s16_lev2p5_tp0p05" "$BASE/D_s16.pt"
run_pair "i_s3_pair_d_s42_lev2p5_tp0p05" "$BASE/D_s42.pt"
run_pair "i_s3_pair_d_s57_lev2p5_tp0p05" "$BASE/D_s57.pt"

echo "DONE."
