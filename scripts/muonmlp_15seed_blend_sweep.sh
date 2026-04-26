#!/bin/bash
# 15-seed MuonMLP blend sweep: 5 original seeds (0,7,42,73,197) + 10 extension
# (1,2,3,4,5,6,8,9,10,11) to test whether more averaging stabilizes the
# +23.18%/mo ensemble edge or reveals 5-seed was seed-selection-lucky.
set -euo pipefail
source .venv/bin/activate

MUON5=$(ls analysis/xgbnew_daily/heldout2025h2_mlp_muon/alltrain_seed*.pkl)
MUON10=$(ls analysis/xgbnew_daily/heldout2025h2_mlp_muon_xt/alltrain_seed*.pkl)
ALL=$(echo -e "${MUON5}\n${MUON10}")
PATHS=$(echo -e "$ALL" | paste -sd,)
N=$(echo -e "$ALL" | wc -l)

OUT=analysis/xgbnew_daily/heldout2025h2_muonmlp_15seed/sweep_out
mkdir -p "$OUT" logs/track1
echo "[muonmlp_15seed] $N models -> $OUT"

python -m xgbnew.sweep_ensemble_grid \
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
    --model-paths "$PATHS" \
    --oos-start 2025-07-01 --oos-end 2025-12-17 \
    --window-days 15 --stride-days 5 \
    --leverage-grid 1.0,2.0,3.0 \
    --min-score-grid 0.55,0.58,0.60,0.61,0.62,0.63,0.65 \
    --top-n-grid 1,2 \
    --hold-through \
    --min-dollar-vol 50000000 \
    --inference-min-vol-grid 0.12 \
    --fee-regimes deploy,stress36x \
    --output-dir "$OUT" \
    2>&1 | tee logs/track1/sweep_heldout2025h2_muonmlp_15seed.log
