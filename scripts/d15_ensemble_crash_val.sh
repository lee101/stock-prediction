#!/usr/bin/env bash
# Test D-15 and D-30 pre-crash-trained ensembles on crash TRUE-OOS.
# D seeds trained on screened32_augmented_train.bin (pre-crash, same as v7).
# This is the RL analog of GBDT's 15-seed bonferroni — does v7's edge
# generalize to a fresh 15-seed D-only ensemble?
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_d_crash"
mkdir -p "$OUT_DIR"

D_BASE=pufferlib_market/checkpoints/screened32_sweep/D
DATA=pufferlib_market/data/screened32_recent_val.bin

ANCHOR="$D_BASE/s1/val_best.pt"

# D-15 (s1..s15)
EXTRAS15=()
for s in 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  EXTRAS15+=("$D_BASE/s$s/val_best.pt")
done

# D-30 (s1..s30)
EXTRAS30=()
for s in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  EXTRAS30+=("$D_BASE/s$s/val_best.pt")
done

run_eval() {
  local name="$1" lev="$2"
  shift 2
  local extras=("$@")
  local out_json="$OUT_DIR/${name}_lev${lev}.json"
  echo "=== $name @ lev=$lev (${#extras[@]} extras = $((${#extras[@]}+1)) models) ==="
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$ANCHOR" \
    --extra-checkpoints "${extras[@]}" \
    --data-path "$DATA" \
    --eval-hours 50 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --max-leverage "$lev" --decision-lag 2 \
    --disable-shorts --deterministic \
    --out "$out_json" 2>&1 | grep -E "median_total|p10_total|negative_w|median_max_drawdown" | head -5
}

run_eval d15 1.0 "${EXTRAS15[@]}"
run_eval d15 2.0 "${EXTRAS15[@]}"
run_eval d30 1.0 "${EXTRAS30[@]}"
run_eval d30 2.0 "${EXTRAS30[@]}"

echo
echo "=== Strict-dominance vs v7 ==="
python3 - <<'PY'
import json
from pathlib import Path
V7 = {
    "lev1.0": {"med": 5.94, "p10": 2.03, "neg": 3, "dd": 5.43},
    "lev2.0": {"med": 9.43, "p10": 2.38, "neg": 5, "dd": 10.47},
}
for name in ("d15", "d30"):
    for lev in ("1.0", "2.0"):
        fn = f"analysis/rl_d_crash/{name}_lev{lev}.json"
        p = Path(fn)
        if not p.exists():
            print(f"{name}_lev{lev}: MISSING"); continue
        d = json.loads(p.read_text())
        s = d.get("summary", d)
        med = s["median_total_return"]*100
        p10 = s["p10_total_return"]*100
        neg = s["negative_windows"]
        dd  = s["median_max_drawdown"]*100
        bar = V7[f"lev{lev}"]
        dm = med - bar["med"]; dp = p10 - bar["p10"]
        dn = neg - bar["neg"]; dd_ = dd - bar["dd"]
        strict = (dm >= 0 and dp >= 0 and dn <= 0 and dd_ <= 0)
        print(f"{name}_lev{lev}: med={med:+.2f} (Δ{dm:+.2f})  p10={p10:+.2f} (Δ{dp:+.2f})  "
              f"neg={neg} (Δ{dn:+d})  dd={dd:.2f} (Δ{dd_:+.2f})  "
              f"{'✅ STRICT-DOM v7' if strict else '❌ fails strict-dom'}")
PY
