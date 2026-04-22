#!/usr/bin/env bash
# Follow-up on per-member guard sweep:
#  (a) I_s3 alone at lev 1/2/3 under guard-on (scales?)
#  (b) Guard-robust ensemble candidate: {I_s3, AD_s4, D_s57, D_s16, D_s42}
#      — members with best on_p10 or positive Δp10 under guard.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_guard_robust"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_recent_val.bin

run_eval() {
  local name="$1"
  shift
  local out_json="$OUT_DIR/${name}.json"
  [ -f "$out_json" ] && { echo "SKIP $name"; return; }
  echo "=== $name ==="
  python -m pufferlib_market.evaluate_holdout \
    --data-path "$DATA" \
    --eval-hours 50 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --decision-lag 2 --disable-shorts --deterministic \
    --out "$out_json" \
    "$@" 2>&1 | grep -E "median_total|p10_total|negative_w" | head -3
}

# (a) I_s3 scaling under guard-on
for LEV in 1.0 1.5 2.0 3.0; do
  run_eval "i_s3_solo_guardon_lev${LEV//./}" \
    --checkpoint "$BASE/I_s3.pt" \
    --max-leverage "$LEV" \
    --death-spiral-tolerance-bps 50.0 \
    --death-spiral-overnight-tolerance-bps 500.0 \
    --death-spiral-stale-after-bars 8
done

# (a.control) I_s3 scaling under guard-off (baseline curve)
for LEV in 1.0 1.5 2.0 3.0; do
  run_eval "i_s3_solo_guardoff_lev${LEV//./}" \
    --checkpoint "$BASE/I_s3.pt" \
    --max-leverage "$LEV"
done

# (b) 5-member guard-robust blend: I_s3 anchor + AD_s4 + D_s57 + D_s16 + D_s42
#     at lev {1,2} under guard-on AND guard-off
EXTRAS=(
  "$BASE/AD_s4.pt" "$BASE/D_s57.pt" "$BASE/D_s16.pt" "$BASE/D_s42.pt"
)

for LEV in 1.0 2.0; do
  run_eval "gr5_guardon_lev${LEV//./}" \
    --checkpoint "$BASE/I_s3.pt" \
    --extra-checkpoints "${EXTRAS[@]}" \
    --max-leverage "$LEV" \
    --death-spiral-tolerance-bps 50.0 \
    --death-spiral-overnight-tolerance-bps 500.0 \
    --death-spiral-stale-after-bars 8

  run_eval "gr5_guardoff_lev${LEV//./}" \
    --checkpoint "$BASE/I_s3.pt" \
    --extra-checkpoints "${EXTRAS[@]}" \
    --max-leverage "$LEV"
done

echo
echo "=== SUMMARY ==="
python3 - <<'PY'
import json
from pathlib import Path

rows = []
for p in sorted(Path("analysis/rl_v7_guard_robust").glob("*.json")):
    d = json.loads(p.read_text())
    s = d.get("summary", d)
    rows.append({
        "name": p.stem,
        "med": s["median_total_return"]*100,
        "p10": s["p10_total_return"]*100,
        "neg": s["negative_windows"],
        "dd":  s["median_max_drawdown"]*100,
    })
rows.sort(key=lambda r: r["name"])
print(f"{'name':<28} {'med%':>7} {'p10%':>7} {'neg':>4} {'dd%':>6}")
for r in rows:
    print(f"{r['name']:<28} {r['med']:>+7.2f} {r['p10']:>+7.2f} {r['neg']:>4d} {r['dd']:>6.2f}")
PY
