#!/bin/bash
# No-picks fallback OOS sweep. Runs sweep_ensemble_grid with SPY / QQQ
# fallback at multiple allocation scales, plus conviction-scaled alloc,
# against both fresh post-crash ensembles (train_end=2025-06-30 and
# train_end=2024-12-31).
#
# Usage: scripts/xgb_no_picks_fallback_sweep.sh <fold_tag> <fallback_symbol>
# fold_tag in {h1,2024}
# fallback_symbol in {SPY,QQQ}

set -euo pipefail

FOLD_TAG="${1:-h1}"
FB_SYM="${2:-SPY}"

cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

if [ "$FOLD_TAG" = "h1" ]; then
    ENS_DIR="analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh"
    TRAIN_END="2025-06-30"
    OOS_START="2025-07-01"
    OOS_END="2026-04-20"
elif [ "$FOLD_TAG" = "2024" ]; then
    ENS_DIR="analysis/xgbnew_daily/oos2024_ensemble_gpu_fresh"
    TRAIN_END="2024-12-31"
    OOS_START="2025-01-02"
    OOS_END="2026-04-20"
else
    echo "unknown fold_tag: $FOLD_TAG (expected h1|2024)" >&2
    exit 1
fi

MODELS=$(ls ${ENS_DIR}/alltrain_seed*.pkl | head -15 | paste -sd,)
N_SEEDS=$(echo "$MODELS" | tr ',' '\n' | wc -l)
echo "[sweep] fold=$FOLD_TAG fb_sym=$FB_SYM n_seeds=$N_SEEDS"
echo "[sweep] train_end=$TRAIN_END oos=$OOS_START→$OOS_END"

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_no_picks_fallback_${FOLD_TAG}_${FB_SYM}"
mkdir -p "$OUT_DIR"

python -u -m xgbnew.sweep_ensemble_grid \
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
    --data-root trainingdata \
    --model-paths "${MODELS}" \
    --train-start 2020-01-01 \
    --train-end "$TRAIN_END" \
    --oos-start "$OOS_START" \
    --oos-end "$OOS_END" \
    --window-days 30 \
    --stride-days 7 \
    --leverage-grid 2.0 \
    --min-score-grid 0.85,0.0 \
    --top-n-grid 1 \
    --hold-through --no-hold-through \
    --fee-regimes deploy \
    --min-dollar-vol 50000000 \
    --inference-min-vol-grid 0.12 \
    --no-picks-fallback "$FB_SYM" \
    --no-picks-fallback-alloc-grid 0.0,0.25,0.5,1.0 \
    --conviction-scaled-alloc-grid 0,1 \
    --conviction-alloc-low 0.55 \
    --conviction-alloc-high 0.85 \
    --output-dir "$OUT_DIR" \
    --verbose 2>&1 | tee "$OUT_DIR/stdout.log"

echo "[sweep] done -> $OUT_DIR"
