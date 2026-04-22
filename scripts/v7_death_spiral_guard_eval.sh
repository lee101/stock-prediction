#!/usr/bin/env bash
# Re-evaluate LIVE v7 ensemble on crash-val WITH the new prod-parity
# death-spiral guard enabled. Compare vs baseline (guard off):
#   baseline lev=1: med +5.94 p10 +2.03 neg 3/81
#   baseline lev=2: med +9.43 p10 +2.38 neg 5/81
# Intraday tol 50bps, overnight 500bps, stale-after 8 bars (prod defaults).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_death_spiral"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_recent_val.bin

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
    --eval-hours 50 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --decision-lag 2 --disable-shorts --deterministic \
    --out "$out_json" \
    "$@" 2>&1 | grep -E "median_total|p10_total|negative_w|median_max_drawdown" | head -5
}

# Guard-on at prod defaults (intraday 50 / overnight 500 / stale-after 8h)
run_eval "guard_on_lev1" --max-leverage 1.0 \
  --death-spiral-tolerance-bps 50.0 \
  --death-spiral-overnight-tolerance-bps 500.0 \
  --death-spiral-stale-after-bars 8

run_eval "guard_on_lev2" --max-leverage 2.0 \
  --death-spiral-tolerance-bps 50.0 \
  --death-spiral-overnight-tolerance-bps 500.0 \
  --death-spiral-stale-after-bars 8

# Overnight-only (always wide) — how much of the delta comes from intraday refusals
run_eval "guard_stale0_lev1" --max-leverage 1.0 \
  --death-spiral-tolerance-bps 50.0 \
  --death-spiral-overnight-tolerance-bps 500.0 \
  --death-spiral-stale-after-bars 0

# Baseline repro (guard off) — sanity check the reported numbers match memory
run_eval "guard_off_lev1" --max-leverage 1.0
run_eval "guard_off_lev2" --max-leverage 2.0

echo
echo "=== SUMMARY (vs baseline guard-off) ==="
python3 - <<'PY'
import json
from pathlib import Path

V7 = {
    "lev1": {"med": 5.94, "p10": 2.03, "neg": 3, "dd": 5.43},
    "lev2": {"med": 9.43, "p10": 2.38, "neg": 5, "dd": 10.47},
}
rows = []
for p in sorted(Path("analysis/rl_v7_death_spiral").glob("*.json")):
    d = json.loads(p.read_text())
    s = d.get("summary", d)
    med = s["median_total_return"] * 100
    p10 = s["p10_total_return"] * 100
    neg = s["negative_windows"]
    dd  = s["median_max_drawdown"] * 100
    name = p.stem
    if "lev1" in name: bar = V7["lev1"]
    elif "lev2" in name: bar = V7["lev2"]
    else: bar = V7["lev1"]
    dm = med - bar["med"]; dp = p10 - bar["p10"]
    dn = neg - bar["neg"]; ddd = dd - bar["dd"]
    rows.append({
        "name": name, "med": med, "p10": p10, "neg": neg, "dd": dd,
        "dm": dm, "dp": dp, "dn": dn, "ddd": ddd,
    })
rows.sort(key=lambda r: r["name"])
print(f"{'name':<22} {'med%':>7} {'p10%':>7} {'neg':>4} {'dd%':>6} "
      f"{'Δmed':>7} {'Δp10':>7} {'Δneg':>5} {'Δdd':>7}")
for r in rows:
    print(f"{r['name']:<22} {r['med']:>+7.2f} {r['p10']:>+7.2f} {r['neg']:>4} "
          f"{r['dd']:>6.2f} {r['dm']:>+7.2f} {r['dp']:>+7.2f} "
          f"{r['dn']:>+5d} {r['ddd']:>+7.2f}")
PY
