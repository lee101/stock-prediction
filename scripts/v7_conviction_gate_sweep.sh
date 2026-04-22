#!/usr/bin/env bash
# Sweep v7 inference-time conviction gates on crash val.
# --min-member-agreement K: require K of 11 seeds to argmax same action
# --min-top-prob P: require softmax-avg top prob >= P
# Goal: find a gate that strict-dominates v7 (med +5.94 p10 +2.03 neg 3/81)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_gates"
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

# Member-agreement sweep at lev=1 (K=2..7 of 11 members)
for K in 2 3 4 5 6 7; do
  run_eval "ma${K}_lev1" --max-leverage 1.0 --min-member-agreement "$K"
done

# Member-agreement sweep at lev=2
for K in 2 3 4 5 6 7; do
  run_eval "ma${K}_lev2" --max-leverage 2.0 --min-member-agreement "$K"
done

# Top-prob sweep at lev=1 (softmax-avg top prob >= threshold)
for P in 0.10 0.15 0.20 0.30 0.40 0.50; do
  run_eval "tp${P}_lev1" --max-leverage 1.0 --min-top-prob "$P"
done

# Top-prob sweep at lev=2
for P in 0.10 0.15 0.20 0.30 0.40 0.50; do
  run_eval "tp${P}_lev2" --max-leverage 2.0 --min-top-prob "$P"
done

echo
echo "=== SUMMARY (vs v7 baselines) ==="
python3 - <<'PY'
import json
from pathlib import Path
V7 = {
    "lev1": {"med": 5.94, "p10": 2.03, "neg": 3, "dd": 5.43},
    "lev2": {"med": 9.43, "p10": 2.38, "neg": 5, "dd": 10.47},
}
rows = []
for p in sorted(Path("analysis/rl_v7_gates").glob("*.json")):
    d = json.loads(p.read_text())
    s = d.get("summary", d)
    med = s["median_total_return"]*100
    p10 = s["p10_total_return"]*100
    neg = s["negative_windows"]
    dd  = s["median_max_drawdown"]*100
    name = p.stem
    if "_lev1" in name: bar = V7["lev1"]
    elif "_lev2" in name: bar = V7["lev2"]
    else: bar = V7["lev1"]
    dm = med - bar["med"]; dp = p10 - bar["p10"]
    dn = neg - bar["neg"]; dd_ = dd - bar["dd"]
    strict = (dm >= 0 and dp >= 0 and dn <= 0 and dd_ <= 0)
    rows.append({
        "name": name, "med": med, "p10": p10, "neg": neg, "dd": dd,
        "dm": dm, "dp": dp, "dn": dn, "dd_": dd_, "strict": strict,
    })
rows.sort(key=lambda r: (not r["strict"], -r["p10"]))
print(f"{'name':<16} {'med%':>7} {'p10%':>7} {'neg':>4} {'dd%':>6} "
      f"{'Δmed':>6} {'Δp10':>6} {'Δneg':>5} {'Δdd':>6} {'strict':>6}")
for r in rows:
    print(f"{r['name']:<16} {r['med']:>+7.2f} {r['p10']:>+7.2f} {r['neg']:>4} {r['dd']:>6.2f} "
          f"{r['dm']:>+6.2f} {r['dp']:>+6.2f} {r['dn']:>+5d} {r['dd_']:>+6.2f} "
          f"{'✅' if r['strict'] else '❌'}")
PY
