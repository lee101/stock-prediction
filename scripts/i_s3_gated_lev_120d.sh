#!/usr/bin/env bash
# Does the tp=0.05 gate unlock higher leverage without tail blowup?
# lev=2.25 @ tp=0.05 was +41.79 med, p10 +1.30, 16 neg, ddW 19.10%.
# Try lev=2.5 / 2.75 / 3.0 with same gate — if ddW stays bounded, PnL scales.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_120d_baseline"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_full_val.bin

run_eval() {
  local name="$1"; shift
  local out_json="$OUT_DIR/${name}.json"
  [ -f "$out_json" ] && { echo "SKIP $name"; return; }
  echo "=== $name ==="
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$BASE/I_s3.pt" \
    --data-path "$DATA" \
    --eval-hours 120 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --decision-lag 2 --disable-shorts --deterministic \
    --death-spiral-tolerance-bps 50.0 \
    --death-spiral-overnight-tolerance-bps 500.0 \
    --death-spiral-stale-after-bars 8 \
    --min-top-prob 0.05 \
    --out "$out_json" \
    "$@" 2>&1 | tee "$OUT_DIR/${name}.log" | grep -E "median_total|p10_total|negative_w|median_max_drawdown" | head -5
}

for lev in 2.5 2.75 3.0 3.5 4.0; do
  run_eval "i_s3_tp0p05_lev${lev/./p}" --max-leverage "$lev"
done

echo "DONE."
