#!/usr/bin/env bash
# D_s16 lev curve WITHOUT drawdown-vs-profit early stopping.
# If lev=3.0 is still Pareto optimum, raw policy breakdown is real.
# If lev=4.0+ improves, the earlier ceiling was an early-stop artifact.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_120d_baseline/d_s16_no_earlystop"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_full_val.bin
CKPT="$BASE/D_s16.pt"

GUARD_ARGS=(
  --death-spiral-tolerance-bps 50.0
  --death-spiral-overnight-tolerance-bps 500.0
  --death-spiral-stale-after-bars 8
)

for lev in 2.0 2.5 3.0 3.5 4.0 5.0 6.0; do
  lev_tag=$(echo "$lev" | tr . p)
  name="d_s16_120d_lev${lev_tag}_no_earlystop"
  out_json="$OUT_DIR/${name}.json"
  [ -f "$out_json" ] && { echo "SKIP $name"; continue; }
  echo "=== $name ==="
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$CKPT" --data-path "$DATA" \
    --eval-hours 120 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --decision-lag 2 --disable-shorts --deterministic \
    --max-leverage "$lev" --min-top-prob 0.0 \
    --no-early-stop \
    "${GUARD_ARGS[@]}" \
    --out "$out_json" \
    2>&1 | tee "$OUT_DIR/${name}.log" \
    | grep -E "median_total|p10_total|negative_w|median_max_drawdown|candidate_window" | head -6
done

echo "DONE."
