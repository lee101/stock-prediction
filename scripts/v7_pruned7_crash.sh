#!/usr/bin/env bash
# Test pruned-7 candidate: v7 minus {D_s42, I_s32, AD_s4, D_s16}
# on crash-heavy recent_val. Must strictly dominate v7
# (med>=5.94 p10>=2.03 neg<=3/81 DD<=5.43) to promote.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_pruned"
mkdir -p "$OUT_DIR"

BASE=pufferlib_market/prod_ensemble_screened32
DATA=pufferlib_market/data/screened32_recent_val.bin

# Pruned-7: drop {D_s42, I_s32, AD_s4, D_s16}
# Keep: C_s7 (anchor), I_s3, D_s2, D_s14, D_s28, D_s57, D_s64
OUT="$OUT_DIR/pruned7_deploy_lev1.json"
echo "=== Pruned-7 ensemble at deploy fees lev=1 ==="
python -m pufferlib_market.evaluate_holdout \
  --checkpoint "$BASE/C_s7.pt" \
  --extra-checkpoints \
    "$BASE/I_s3.pt" \
    "$BASE/D_s2.pt" \
    "$BASE/D_s14.pt" \
    "$BASE/D_s28.pt" \
    "$BASE/D_s57.pt" \
    "$BASE/D_s64.pt" \
  --data-path "$DATA" \
  --eval-hours 50 --exhaustive \
  --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
  --max-leverage 1.0 --decision-lag 2 \
  --disable-shorts --deterministic \
  --out "$OUT" 2>&1 | grep -E "median_total|p10_total|negative_w|median_max_drawdown" | head -5

echo
echo "=== Pruned-7 @ lev=2.0 ==="
OUT2="$OUT_DIR/pruned7_deploy_lev2.json"
python -m pufferlib_market.evaluate_holdout \
  --checkpoint "$BASE/C_s7.pt" \
  --extra-checkpoints \
    "$BASE/I_s3.pt" \
    "$BASE/D_s2.pt" \
    "$BASE/D_s14.pt" \
    "$BASE/D_s28.pt" \
    "$BASE/D_s57.pt" \
    "$BASE/D_s64.pt" \
  --data-path "$DATA" \
  --eval-hours 50 --exhaustive \
  --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
  --max-leverage 2.0 --decision-lag 2 \
  --disable-shorts --deterministic \
  --out "$OUT2" 2>&1 | grep -E "median_total|p10_total|negative_w|median_max_drawdown" | head -5

echo
echo "=== Summary ==="
python3 - <<'PY'
import json
from pathlib import Path
V7_LEV1 = {"med": 5.94, "p10": 2.03, "neg": 3, "dd": 5.43}
V7_LEV2 = {"med": 9.43, "p10": 2.38, "neg": 5, "dd": 10.47}
for label, fn, bar in [
    ("pruned7_lev1", "analysis/rl_v7_pruned/pruned7_deploy_lev1.json", V7_LEV1),
    ("pruned7_lev2", "analysis/rl_v7_pruned/pruned7_deploy_lev2.json", V7_LEV2),
]:
    p = Path(fn)
    if not p.exists():
        print(f"{label}: MISSING {fn}")
        continue
    d = json.loads(p.read_text())
    s = d.get("summary", d)
    med = s["median_total_return"]*100
    p10 = s["p10_total_return"]*100
    neg = s["negative_windows"]
    dd  = s["median_max_drawdown"]*100
    dm = med - bar["med"]; dp = p10 - bar["p10"]
    dn = neg - bar["neg"]; dd_ = dd - bar["dd"]
    strict = (dm >= 0 and dp >= 0 and dn <= 0 and dd_ <= 0)
    print(f"{label}: med={med:+.2f} (Δ{dm:+.2f})  p10={p10:+.2f} (Δ{dp:+.2f})  "
          f"neg={neg} (Δ{dn:+d})  dd={dd:.2f} (Δ{dd_:+.2f})  "
          f"{'✅ STRICT-DOM v7' if strict else '❌ fails strict-dom'}")
PY
