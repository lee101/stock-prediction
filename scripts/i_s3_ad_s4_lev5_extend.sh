#!/usr/bin/env bash
# Extend the I_s3+AD_s4 pair curve to lev=5 and run triad at lev=4.
# Pair strict-dominates solo to lev=4 (same med/p10, -2 neg, ~30% lower dd).
# Does pair roll over at lev=5 like solo does (solo peak lev=4, drops at 5)?
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

# Pair lev=5 (does rollover match solo?)
run_eval "pair_iad_lev50_on" \
  --checkpoint "$BASE/I_s3.pt" \
  --extra-checkpoints "$BASE/AD_s4.pt" \
  --max-leverage 5.0 "${GUARD_ARGS[@]}"

# Triad lev=4/5 (does triad extend further than pair?)
for LEV in 4.0 5.0; do
  run_eval "triad_lev${LEV//./}_on" \
    --checkpoint "$BASE/I_s3.pt" \
    --extra-checkpoints "$BASE/AD_s4.pt" "$BASE/D_s57.pt" \
    --max-leverage "$LEV" "${GUARD_ARGS[@]}"
done

echo
echo "=== FINAL LEVERAGE CURVE (guard-on, crash val) ==="
python3 - <<'PY'
import json
from pathlib import Path

def load(p):
    if not p.exists(): return None
    s = json.loads(p.read_text())["summary"]
    return (s["median_total_return"]*100, s["p10_total_return"]*100,
            s["negative_windows"], s["median_max_drawdown"]*100)

print(f"{'config':<28} {'lev':>5} {'med%':>7} {'p10%':>7} {'neg':>4} {'dd%':>6}")
print("-" * 64)

# Solo reference
solo_lookup = {
    "1.0": "analysis/rl_v7_guard_robust/i_s3_solo_guardon_lev10.json",
    "1.5": "analysis/rl_v7_guard_robust/i_s3_solo_guardon_lev15.json",
    "2.0": "analysis/rl_v7_guard_robust/i_s3_solo_guardon_lev20.json",
    "3.0": "analysis/rl_v7_guard_robust/i_s3_solo_guardon_lev30.json",
    "4.0": "analysis/rl_i_s3_pairwise/i_s3_solo_guardon_lev40.json",
    "5.0": "analysis/rl_i_s3_pairwise/i_s3_solo_guardon_lev50.json",
}
for lev, path in solo_lookup.items():
    r = load(Path(path))
    if r:
        m, p10, n, dd = r
        print(f"{'I_s3 solo':<28} {lev:>5} {m:>+7.2f} {p10:>+7.2f} {n:>4d} {dd:>6.2f}")

# Pair curve (lev=1/1.5/2/3/4/5)
pair_lookup = {
    "1.0": "analysis/rl_i_s3_blend_levcurve/pair_iad_lev10_on.json",
    "1.5": "analysis/rl_i_s3_blend_levcurve/pair_iad_lev15_on.json",
    "2.0": "analysis/rl_i_s3_pairwise/pair_i_s3_AD_s4_lev2_on.json",
    "3.0": "analysis/rl_i_s3_blend_levcurve/pair_iad_lev30_on.json",
    "4.0": "analysis/rl_i_s3_blend_levcurve/pair_iad_lev40_on.json",
    "5.0": "analysis/rl_i_s3_blend_levcurve/pair_iad_lev50_on.json",
}
print()
for lev, path in pair_lookup.items():
    r = load(Path(path))
    if r:
        m, p10, n, dd = r
        print(f"{'I_s3 + AD_s4':<28} {lev:>5} {m:>+7.2f} {p10:>+7.2f} {n:>4d} {dd:>6.2f}")

# Triad curve
triad_lookup = {
    "1.0": "analysis/rl_i_s3_blend_levcurve/triad_lev10_on.json",
    "2.0": "analysis/rl_i_s3_blend_levcurve/triad_lev20_on.json",
    "3.0": "analysis/rl_i_s3_blend_levcurve/triad_lev30_on.json",
    "4.0": "analysis/rl_i_s3_blend_levcurve/triad_lev40_on.json",
    "5.0": "analysis/rl_i_s3_blend_levcurve/triad_lev50_on.json",
}
print()
for lev, path in triad_lookup.items():
    r = load(Path(path))
    if r:
        m, p10, n, dd = r
        print(f"{'I_s3 + AD_s4 + D_s57':<28} {lev:>5} {m:>+7.2f} {p10:>+7.2f} {n:>4d} {dd:>6.2f}")
PY
