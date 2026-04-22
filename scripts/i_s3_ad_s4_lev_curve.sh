#!/usr/bin/env bash
# I_s3+AD_s4 strict-dom'd I_s3 solo at lev=2 guard-on (same med/p10/dd,
# 18→16 neg). Sweep across lev to see if the dominance holds.
# Also test 3-way {I_s3, AD_s4, D_s57}: D_s57 brought smooth p10 in pairwise.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_i_s3_blend_levcurve"
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

# (a) I_s3+AD_s4 lev-sweep guard-on
for LEV in 1.0 1.5 3.0 4.0; do
  run_eval "pair_iad_lev${LEV//./}_on" \
    --checkpoint "$BASE/I_s3.pt" \
    --extra-checkpoints "$BASE/AD_s4.pt" \
    --max-leverage "$LEV" "${GUARD_ARGS[@]}"
done

# (b) 3-way {I_s3, AD_s4, D_s57} at lev {1, 2, 3}
for LEV in 1.0 2.0 3.0; do
  run_eval "triad_lev${LEV//./}_on" \
    --checkpoint "$BASE/I_s3.pt" \
    --extra-checkpoints "$BASE/AD_s4.pt" "$BASE/D_s57.pt" \
    --max-leverage "$LEV" "${GUARD_ARGS[@]}"
done

echo
echo "=== I_s3 blend leverage curves (guard-on, crash val) ==="
python3 - <<'PY'
import json
from pathlib import Path

# Baselines for context
solo = {}
for lev in ["10", "15", "20", "30", "40", "50"]:
    p = Path(f"analysis/rl_v7_guard_robust/i_s3_solo_guardon_lev{lev}.json")
    if p.exists():
        s = json.loads(p.read_text())["summary"]
        solo[lev] = (s["median_total_return"]*100, s["p10_total_return"]*100,
                     s["negative_windows"], s["median_max_drawdown"]*100)

# Also pull the lev4/5 from analysis/rl_i_s3_pairwise (stored there)
for lev in ["40", "50"]:
    p = Path(f"analysis/rl_i_s3_pairwise/i_s3_solo_guardon_lev{lev}.json")
    if p.exists():
        s = json.loads(p.read_text())["summary"]
        solo[lev] = (s["median_total_return"]*100, s["p10_total_return"]*100,
                     s["negative_windows"], s["median_max_drawdown"]*100)

# Pair lev=2 baseline (from earlier run)
pair_lev2 = None
p = Path("analysis/rl_i_s3_pairwise/pair_i_s3_AD_s4_lev2_on.json")
if p.exists():
    s = json.loads(p.read_text())["summary"]
    pair_lev2 = (s["median_total_return"]*100, s["p10_total_return"]*100,
                 s["negative_windows"], s["median_max_drawdown"]*100)

print(f"{'config':<22} {'lev':>5} {'med%':>7} {'p10%':>7} {'neg':>4} {'dd%':>6}")
print("-" * 60)
for lev, key in [("1.0","10"), ("1.5","15"), ("2.0","20"), ("3.0","30"),
                 ("4.0","40"), ("5.0","50")]:
    if key in solo:
        m, p10, n, dd = solo[key]
        print(f"{'I_s3 solo':<22} {lev:>5} {m:>+7.2f} {p10:>+7.2f} {n:>4d} {dd:>6.2f}")
print()
new_rows = []
for p in sorted(Path("analysis/rl_i_s3_blend_levcurve").glob("*.json")):
    d = json.loads(p.read_text())
    s = d.get("summary", d)
    new_rows.append({
        "name": p.stem,
        "med": s["median_total_return"]*100,
        "p10": s["p10_total_return"]*100,
        "neg": s["negative_windows"],
        "dd":  s["median_max_drawdown"]*100,
    })
for r in new_rows:
    print(f"{r['name']:<22} {'':>5} {r['med']:>+7.2f} {r['p10']:>+7.2f} {r['neg']:>4d} {r['dd']:>6.2f}")
if pair_lev2 is not None:
    m, p10, n, dd = pair_lev2
    print(f"{'pair_iad_lev2_on':<22} {'':>5} {m:>+7.2f} {p10:>+7.2f} {n:>4d} {dd:>6.2f}  (prior)")
PY
