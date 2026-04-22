#!/usr/bin/env bash
# Screen all 32 screened32 checkpoints solo on crash-heavy recent_val to
# identify top seeds by p10 for ensemble expansion (task #22 follow-up).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/rl_screen32_solo_crash"
mkdir -p "$OUT_DIR"

for ckpt_path in pufferlib_market/prod_ensemble_screened32/*.pt; do
  name=$(basename "$ckpt_path" .pt)
  out_json="$OUT_DIR/${name}.json"
  if [ -f "$out_json" ]; then
    echo "SKIP $name (exists)"
    continue
  fi
  echo "=== $name ==="
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$ckpt_path" \
    --data-path pufferlib_market/data/screened32_recent_val.bin \
    --eval-hours 50 --exhaustive \
    --fee-rate 0.001 --slippage-bps 5.0 --fill-buffer-bps 5.0 \
    --max-leverage 1.0 --decision-lag 2 \
    --disable-shorts --deterministic \
    --out "$out_json" 2>&1 | grep -E "\"median_total|\"p10_total|\"negative_w" | head -3
done

echo "=== LEADERBOARD (by p10 desc) ==="
python3 - <<'PY'
import json, os
from pathlib import Path
rows = []
for p in sorted(Path("analysis/rl_screen32_solo_crash").glob("*.json")):
    d = json.loads(p.read_text())
    rows.append({
        "name": p.stem,
        "med": d["median_total_return"],
        "p10": d["p10_total_return"],
        "neg": d["negative_windows"],
        "dd": d["median_max_drawdown"],
    })
rows.sort(key=lambda r: r["p10"], reverse=True)
print(f"{'name':<20} {'med%':>7} {'p10%':>7} {'neg':>5} {'dd%':>6}")
for r in rows:
    print(f"{r['name']:<20} {r['med']*100:>7.2f} {r['p10']*100:>7.2f} {r['neg']:>5} {r['dd']*100:>6.2f}")
PY
