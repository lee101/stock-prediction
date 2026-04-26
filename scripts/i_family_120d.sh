#!/usr/bin/env bash
# Extend I_s3 solo findings: test I_s32 solo + I_s3+I_s32 pair on 120d windows.
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

# I_s32 solo at lev=1, 2 (where I_s3 was best)
for lev in 1.0 2.0; do
  run_eval "i_s32_solo_lev${lev/./p}" \
    --checkpoint "$BASE/I_s32.pt" --max-leverage "$lev"
done

# I_s3 + I_s32 pair at lev 1, 2
for lev in 1.0 2.0; do
  run_eval "i_s3_i_s32_pair_lev${lev/./p}" \
    --checkpoint "$BASE/I_s3.pt" \
    --extra-checkpoints "$BASE/I_s32.pt" \
    --max-leverage "$lev"
done

echo "DONE."
