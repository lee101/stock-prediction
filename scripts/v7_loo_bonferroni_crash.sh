#!/usr/bin/env bash
# Leave-one-out bonferroni: run 11 LOO-10 variants of v7 ensemble on
# crash-heavy val to test that no single seed carries the edge
# (analog of GBDT 15-seed bonferroni stability test).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_v7_loo_crash"
mkdir -p "$OUT_DIR"

MEMBERS=(C_s7 D_s16 D_s42 AD_s4 I_s3 D_s2 D_s14 D_s28 D_s57 D_s64 I_s32)
BASE=pufferlib_market/prod_ensemble_screened32

for drop in "${MEMBERS[@]}"; do
  out_json="$OUT_DIR/drop_${drop}.json"
  if [ -f "$out_json" ]; then continue; fi
  # Build anchor + extras, excluding the dropped member
  anchor=""
  extras=()
  for m in "${MEMBERS[@]}"; do
    [ "$m" = "$drop" ] && continue
    if [ -z "$anchor" ]; then
      anchor="$BASE/${m}.pt"
    else
      extras+=("$BASE/${m}.pt")
    fi
  done
  echo "=== drop=$drop (anchor=$anchor, $(( ${#MEMBERS[@]} - 1 )) members) ==="
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$anchor" \
    --extra-checkpoints "${extras[@]}" \
    --data-path pufferlib_market/data/screened32_recent_val.bin \
    --eval-hours 50 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --max-leverage 1.0 --decision-lag 2 \
    --disable-shorts --deterministic \
    --out "$out_json" 2>&1 | grep -E "\"median_total|\"p10_total|\"negative_w" | head -3
done

echo "=== LOO-10 BONFERRONI LEADERBOARD ==="
python3 - <<'PY'
import json
from pathlib import Path
rows = []
for p in sorted(Path("analysis/rl_v7_loo_crash").glob("drop_*.json")):
    d = json.loads(p.read_text())
    s = d.get("summary", d)
    rows.append({
        "dropped": p.stem.replace("drop_",""),
        "med": s["median_total_return"],
        "p10": s["p10_total_return"],
        "neg": s["negative_windows"],
        "dd":  s["median_max_drawdown"],
    })
rows.sort(key=lambda r: r["p10"], reverse=True)
print(f"{'dropped':<12} {'med%':>7} {'p10%':>7} {'neg':>5} {'dd%':>6}")
for r in rows:
    print(f"{r['dropped']:<12} {r['med']*100:>7.2f} {r['p10']*100:>7.2f} {r['neg']:>5} {r['dd']*100:>6.2f}")
print()
# Pass/fail check: all LOO-10 variants must have positive med AND positive p10
pass_count = sum(1 for r in rows if r["med"] > 0 and r["p10"] > 0)
print(f"LOO bonferroni: {pass_count}/{len(rows)} variants keep positive med + p10")
if pass_count == len(rows):
    print("PASS — no single member carries the edge")
else:
    print("FAIL — edge is concentrated in some members")
    for r in rows:
        if r["med"] <= 0 or r["p10"] <= 0:
            print(f"  dropping {r['dropped']} makes ensemble: med={r['med']*100:.2f}% p10={r['p10']*100:.2f}%")
PY
