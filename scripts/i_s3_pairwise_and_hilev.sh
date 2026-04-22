#!/usr/bin/env bash
# Follow-up to i_s3_solo_and_guard_robust_ensemble.sh.
#  (a) Pairwise blend: I_s3 + ONE other guard-friendly member at lev=2 guard-on.
#      Partners: D_s16, D_s42, D_s57, AD_s4, I_s32 (I-shape comparator).
#  (b) I_s3 solo at high leverage: 4.0, 5.0 guard-on. Where does curve break?
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_i_s3_pairwise"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_recent_val.bin

GUARD_ARGS=(
  --death-spiral-tolerance-bps 50.0
  --death-spiral-overnight-tolerance-bps 500.0
  --death-spiral-stale-after-bars 8
)

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

# (a) Pairwise I_s3 + partner at lev=2 guard-on
for PARTNER in D_s16 D_s42 D_s57 AD_s4 I_s32; do
  run_eval "pair_i_s3_${PARTNER}_lev2_on" \
    --checkpoint "$BASE/I_s3.pt" \
    --extra-checkpoints "$BASE/${PARTNER}.pt" \
    --max-leverage 2.0 "${GUARD_ARGS[@]}"
done

# (b) I_s3 solo at higher leverage (guard-on only — guard-off known worse)
for LEV in 4.0 5.0; do
  run_eval "i_s3_solo_guardon_lev${LEV//./}" \
    --checkpoint "$BASE/I_s3.pt" \
    --max-leverage "$LEV" "${GUARD_ARGS[@]}"
done

echo
echo "=== SUMMARY (baselines + new cells) ==="
python3 - <<'PY'
import json
from pathlib import Path

# Pull baseline: I_s3 solo lev=2 guard-on from the previous run
prev = Path("analysis/rl_v7_guard_robust/i_s3_solo_guardon_lev20.json")
baseline = json.loads(prev.read_text()).get("summary", {})
bm = baseline["median_total_return"]*100
bp = baseline["p10_total_return"]*100
bn = baseline["negative_windows"]
bd = baseline["median_max_drawdown"]*100
print(f"baseline I_s3 solo lev=2 guard-on: med {bm:+.2f} p10 {bp:+.2f} neg {bn} dd {bd:.2f}")
print()

rows = []
for p in sorted(Path("analysis/rl_i_s3_pairwise").glob("*.json")):
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
print(f"{'name':<34} {'med%':>7} {'p10%':>7} {'neg':>4} {'dd%':>6} {'Δmed':>7} {'Δp10':>7}")
for r in rows:
    dm = r["med"] - bm
    dp = r["p10"] - bp
    # Baseline differences only meaningful for lev=2 cells
    is_lev2 = "lev2" in r["name"]
    dm_str = f"{dm:+.2f}" if is_lev2 else "  n/a"
    dp_str = f"{dp:+.2f}" if is_lev2 else "  n/a"
    print(f"{r['name']:<34} {r['med']:>+7.2f} {r['p10']:>+7.2f} {r['neg']:>4d} {r['dd']:>6.2f} {dm_str:>7} {dp_str:>7}")
PY
