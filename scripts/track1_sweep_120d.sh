#!/bin/bash
# Track 1: 120-day OOS marketsim sweep across {xgb, lgb, cat, mlp} ensembles.
# All trained through 2025-12-17 on the same 846-symbol feature frame.
# OOS = 2025-12-18 → 2026-04-17 (~120 days, 30d windows, 7d stride).
set -euo pipefail
source .venv/bin/activate

SYMBOLS=symbol_lists/stocks_wide_1000_v1.txt
OOS_START=2025-12-18
OOS_END=2026-04-17
LEV=2.0
MS_GRID="0.50,0.60,0.70,0.80,0.85"
TOPN=1
DV=50000000
VOL=0.12

for family in xgb lgb cat mlp; do
    DIR=analysis/xgbnew_daily/track1_oos120d_${family}
    if [ ! -f "$DIR/alltrain_seed0.pkl" ]; then
        echo "[SKIP] $family — no pickles found in $DIR"
        continue
    fi
    MODEL_PATHS=$(ls ${DIR}/alltrain_seed*.pkl | paste -sd,)
    OUT=analysis/xgbnew_daily/track1_oos120d_${family}/sweep_out
    mkdir -p "$OUT"
    echo "[SWEEP] family=$family"
    python -m xgbnew.sweep_ensemble_grid \
        --symbols-file "$SYMBOLS" \
        --model-paths "$MODEL_PATHS" \
        --oos-start "$OOS_START" --oos-end "$OOS_END" \
        --window-days 30 --stride-days 7 \
        --leverage-grid "$LEV" \
        --min-score-grid "$MS_GRID" \
        --top-n-grid "$TOPN" \
        --hold-through \
        --min-dollar-vol "$DV" \
        --inference-min-vol-grid "$VOL" \
        --fee-regimes deploy,stress36x \
        --output-dir "$OUT" \
        2>&1 | tee "logs/track1/sweep_${family}.log"
done

echo ""
echo "=== TRACK 1 COMPARISON ==="
for family in xgb lgb cat mlp; do
    J=$(ls -t analysis/xgbnew_daily/track1_oos120d_${family}/sweep_out/sweep_*.json 2>/dev/null | head -1)
    [ -n "$J" ] && [ -f "$J" ] || continue
    python -c "
import json, sys
data = json.load(open('$J'))
rows = data.get('cells', data) if isinstance(data, dict) else data
best = None
for r in rows:
    if r.get('fee_regime') != 'deploy': continue
    if best is None or r.get('median_monthly_pct',-999) > best.get('median_monthly_pct',-999):
        best = r
if best:
    print(f'$family  lev={best[\"leverage\"]} ms={best[\"min_score\"]}  med={best[\"median_monthly_pct\"]:+7.2f}%/mo  p10={best[\"p10_monthly_pct\"]:+7.2f}  neg={best[\"n_neg\"]}/{best[\"n_windows\"]}  ddW={best[\"worst_dd_pct\"]:5.2f}')
else:
    print(f'$family  (no deploy rows)')
"
done
