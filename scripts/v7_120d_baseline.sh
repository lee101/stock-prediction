#!/usr/bin/env bash
# 120-day baseline eval for v7 11-model ensemble on screened32_full_val.bin
# (314d data → 194 exhaustive windows).
# Guard-on at prod defaults: intraday 50bps / overnight 500bps / stale-after 8 bars.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_120d_baseline"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_full_val.bin

EXTRAS=(
  "$BASE/D_s16.pt" "$BASE/D_s42.pt" "$BASE/AD_s4.pt"
  "$BASE/I_s3.pt" "$BASE/D_s2.pt" "$BASE/D_s14.pt"
  "$BASE/D_s28.pt" "$BASE/D_s57.pt" "$BASE/D_s64.pt"
  "$BASE/I_s32.pt"
)

run_eval() {
  local name="$1"
  shift
  local out_json="$OUT_DIR/${name}.json"
  [ -f "$out_json" ] && { echo "SKIP $name"; return; }
  echo "=== $name ==="
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$BASE/C_s7.pt" \
    --extra-checkpoints "${EXTRAS[@]}" \
    --data-path "$DATA" \
    --eval-hours 120 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --decision-lag 2 --disable-shorts --deterministic \
    --out "$out_json" \
    "$@" 2>&1 | tee "$OUT_DIR/${name}.log" | grep -E "median_total|p10_total|negative_w|median_max_drawdown|windows:" | head -10
}

# v7 11-model guard-on lev=1 and lev=2
run_eval "v7_guard_on_lev1" --max-leverage 1.0 \
  --death-spiral-tolerance-bps 50.0 \
  --death-spiral-overnight-tolerance-bps 500.0 \
  --death-spiral-stale-after-bars 8

run_eval "v7_guard_on_lev2" --max-leverage 2.0 \
  --death-spiral-tolerance-bps 50.0 \
  --death-spiral-overnight-tolerance-bps 500.0 \
  --death-spiral-stale-after-bars 8

# Guard-off reference for delta
run_eval "v7_guard_off_lev1" --max-leverage 1.0
run_eval "v7_guard_off_lev2" --max-leverage 2.0

echo "DONE."
