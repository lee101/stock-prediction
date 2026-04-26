#!/usr/bin/env bash
# Full characterization of D_s16 solo:
# 1. lev × gate sweep at 120d (like I did for I_s3)
# 2. Window-length robustness at 60d/90d
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_120d_baseline/d_s16_characterization"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_full_val.bin
CKPT="$BASE/D_s16.pt"

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
    | grep -E "median_total|p10_total|negative_w|median_max_drawdown|candidate_window" | head -6
}

# 120d lev × gate sweep
for lev in 1.0 1.5 2.0 2.5 3.0; do
  for tp in 0.0 0.05; do
    lev_tag=$(echo "$lev" | tr . p)
    tp_tag=$(echo "$tp" | tr . p)
    run_eval "d_s16_120d_lev${lev_tag}_tp${tp_tag}" 120 "$lev" "$tp"
  done
done

# 60d / 90d at champion cell and neighbors
for hrs in 60 90; do
  for lev in 2.0 2.5 3.0; do
    for tp in 0.0 0.05; do
      lev_tag=$(echo "$lev" | tr . p)
      tp_tag=$(echo "$tp" | tr . p)
      run_eval "d_s16_${hrs}d_lev${lev_tag}_tp${tp_tag}" "$hrs" "$lev" "$tp"
    done
  done
done

echo "DONE."
