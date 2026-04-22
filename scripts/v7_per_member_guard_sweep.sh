#!/usr/bin/env bash
# Per-member v7 guard-robustness sweep.
# For each of the 11 v7 members, evaluate solo at lev=1 with guard-on
# (intraday 50bps / overnight 500bps / stale-after 8 bars) and compare
# against the same member's guard-off baseline.
#
# Hypothesis: most members rely on refused sells for their sim PnL; a
# few may be guard-robust (policy doesn't set up mid-episode death-spirals).
# Those robust members are candidates for a guard-safe reduced ensemble.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_per_member_guard"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_recent_val.bin

MEMBERS=(
  "C_s7"  "D_s16" "D_s42" "AD_s4" "I_s3"  "D_s2"
  "D_s14" "D_s28" "D_s57" "D_s64" "I_s32"
)

run_single() {
  local name="$1" guard="$2"
  local out_json="$OUT_DIR/${name}_${guard}.json"
  [ -f "$out_json" ] && { echo "SKIP $name $guard"; return; }
  echo "=== $name $guard ==="
  local args=(
    --checkpoint "$BASE/${name}.pt"
    --data-path "$DATA"
    --eval-hours 50 --exhaustive
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0
    --decision-lag 2 --disable-shorts --deterministic
    --max-leverage 1.0
    --out "$out_json"
  )
  if [ "$guard" = "on" ]; then
    args+=(
      --death-spiral-tolerance-bps 50.0
      --death-spiral-overnight-tolerance-bps 500.0
      --death-spiral-stale-after-bars 8
    )
  fi
  python -m pufferlib_market.evaluate_holdout "${args[@]}" 2>&1 \
    | grep -E "median_total|p10_total|negative_w" | head -3
}

for m in "${MEMBERS[@]}"; do
  run_single "$m" "off"
  run_single "$m" "on"
done

echo
echo "=== PER-MEMBER SUMMARY lev=1 (guard-off vs guard-on) ==="
python3 - <<'PY'
import json
from pathlib import Path

root = Path("analysis/rl_v7_per_member_guard")
members = ["C_s7", "D_s16", "D_s42", "AD_s4", "I_s3", "D_s2",
           "D_s14", "D_s28", "D_s57", "D_s64", "I_s32"]

def load(path):
    d = json.loads(path.read_text())
    s = d.get("summary", d)
    return (s["median_total_return"]*100, s["p10_total_return"]*100,
            s["negative_windows"], s["median_max_drawdown"]*100)

rows = []
for m in members:
    off = root / f"{m}_off.json"
    on  = root / f"{m}_on.json"
    if not off.exists() or not on.exists():
        continue
    mo, po, no, do = load(off)
    mn, pn, nn, dn = load(on)
    rows.append({
        "m": m, "mo": mo, "po": po, "no": no, "do": do,
        "mn": mn, "pn": pn, "nn": nn, "dn": dn,
        "dmed": mn - mo, "dp10": pn - po, "dneg": nn - no,
    })

# Rank by how much the guard preserved the guard-off signal (smaller |Δp10| is better)
rows.sort(key=lambda r: r["dp10"], reverse=True)

print(f"{'member':<7} | {'off_med':>7} {'off_p10':>7} {'off_neg':>7} | "
      f"{'on_med':>7} {'on_p10':>7} {'on_neg':>6} | "
      f"{'Δmed':>6} {'Δp10':>6} {'Δneg':>5}")
print("-" * 100)
for r in rows:
    print(f"{r['m']:<7} | {r['mo']:>+7.2f} {r['po']:>+7.2f} {r['no']:>7d} | "
          f"{r['mn']:>+7.2f} {r['pn']:>+7.2f} {r['nn']:>6d} | "
          f"{r['dmed']:>+6.2f} {r['dp10']:>+6.2f} {r['dneg']:>+5d}")
PY
