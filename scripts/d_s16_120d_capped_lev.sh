#!/usr/bin/env bash
# D_s16 solo lev curve under Reg-T overnight 2x gross-leverage cap.
#
# Mirrors prod xgbnew/live_trader._eod_deleverage_tick which forces
# account gross exposure <= equity * 2.0 before close. In daily-bar
# mode each bar IS one overnight, so --overnight-max-gross-leverage 2.0
# clips realized leverage to 2.0 (no-op at lev<=2.0, identical to
# lev=2.0 at lev>=2.0).
#
# Compare against analysis/rl_v7_120d_baseline/d_s16_characterization/
# (uncapped) to size the deploy haircut.
#
# Refs: project_eod_deleverage_audit_2026_04_29
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_120d_baseline/d_s16_capped"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_full_val.bin
CKPT="$BASE/D_s16.pt"
CAP=2.0

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
  echo "=== $name (eval-hours=$hrs lev=$lev cap=$CAP tp=$tp) ==="
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$CKPT" \
    --data-path "$DATA" \
    --eval-hours "$hrs" --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --decision-lag 2 --disable-shorts --deterministic \
    --max-leverage "$lev" \
    --overnight-max-gross-leverage "$CAP" \
    --min-top-prob "$tp" \
    "${GUARD_ARGS[@]}" \
    --out "$out_json" \
    2>&1 | tee "$OUT_DIR/${name}.log" \
    | grep -E "median_total|p10_total|negative_w|median_max_drawdown|candidate_window|sortino" | head -6
}

# 120d lev curve under cap=2.0
for lev in 1.0 1.5 2.0 2.5 3.0; do
  for tp in 0.0 0.05; do
    lev_tag=$(echo "$lev" | tr . p)
    tp_tag=$(echo "$tp" | tr . p)
    run_eval "d_s16_120d_lev${lev_tag}_tp${tp_tag}_cap${CAP/./p}" 120 "$lev" "$tp"
  done
done

echo "DONE."
