#!/usr/bin/env bash
# Cross-check the 120d champion (I_s3 solo lev=2.5 + tp=0.05 gate) at 60d
# and 90d windows on the SAME screened32_full_val.bin dataset. If the
# champion is window-length-robust, same lev/gate wins at 60d and 90d too.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_120d_baseline/window_robustness"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_full_val.bin
CKPT="$BASE/I_s3.pt"

GUARD_ARGS=(
  --death-spiral-tolerance-bps 50.0
  --death-spiral-overnight-tolerance-bps 500.0
  --death-spiral-stale-after-bars 8
)

run_eval() {
  local name="$1"
  local hrs="$2"
  local lev="$3"
  local tp="$4"
  local out_json="$OUT_DIR/${name}.json"
  [ -f "$out_json" ] && { echo "SKIP $name"; return; }
  echo "=== $name (eval-hours=$hrs lev=$lev tp=$tp) ==="
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$CKPT" \
    --data-path "$DATA" \
    --eval-hours "$hrs" --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --decision-lag 2 --disable-shorts --deterministic \
    --max-leverage "$lev" \
    --min-top-prob "$tp" \
    "${GUARD_ARGS[@]}" \
    --out "$out_json" \
    2>&1 | tee "$OUT_DIR/${name}.log" \
    | grep -E "median_total|p10_total|negative_w|median_max_drawdown|candidate_window|windows:" | head -10
}

# Champion config (lev=2.5 tp=0.05) and neighbors at shorter windows.
for hrs in 60 90; do
  run_eval "i_s3_${hrs}d_lev2p0_tp0p00" "$hrs" 2.0 0.0
  run_eval "i_s3_${hrs}d_lev2p5_tp0p00" "$hrs" 2.5 0.0
  run_eval "i_s3_${hrs}d_lev3p0_tp0p00" "$hrs" 3.0 0.0
  run_eval "i_s3_${hrs}d_lev2p0_tp0p05" "$hrs" 2.0 0.05
  run_eval "i_s3_${hrs}d_lev2p5_tp0p05" "$hrs" 2.5 0.05
  run_eval "i_s3_${hrs}d_lev3p0_tp0p05" "$hrs" 3.0 0.05
done

echo "DONE."
