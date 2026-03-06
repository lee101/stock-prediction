#!/usr/bin/env bash
# End-to-end 8-stock candidate trainer aligned to the current live meta stack.
# Long-only: NVDA, PLTR, GOOG, TSLA
# Short-only: DBX, TRIP, MTCH, NYT
set -euo pipefail

PYTHON="${PYTHON:-.venv313/bin/python -u}"
TRAIN="unified_hourly_experiment/train_bf16_efficient.py"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"

SYMBOLS="${SYMBOLS:-NVDA,PLTR,GOOG,TSLA,DBX,TRIP,MTCH,NYT}"
EVAL_SYMBOLS="${EVAL_SYMBOLS:-$SYMBOLS}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-5}"
WD="${WD:-0.04}"
RW="${RW:-0.15}"
SEQ="${SEQ:-48}"
SEED="${SEED:-42}"
HIDDEN="${HIDDEN:-512}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"

EVAL_PERIODS="${EVAL_PERIODS:-1,7,30,60,120,150}"
EVAL_FEE="${EVAL_FEE:-0.001}"
EVAL_MARGIN="${EVAL_MARGIN:-0.0625}"
MAX_POS="${MAX_POS:-5}"
MAX_HOLD="${MAX_HOLD:-5}"
MIN_EDGE="${MIN_EDGE:-0.001}"

if [[ "${1:-}" == "--quick" ]]; then
    EPOCHS=8
    EVAL_PERIODS="7,30"
    echo "=== QUICK MODE: ${EPOCHS} epochs, eval ${EVAL_PERIODS} ==="
fi

RUN_NAME="${RUN_NAME:-live8_tsla_rw${RW//.}_wd${WD//.}_seq${SEQ}_s${SEED}}"
CKPT_DIR="unified_hourly_experiment/checkpoints/${RUN_NAME}"

echo "=========================================="
echo " Live8 Candidate Training"
echo "=========================================="
echo " Run: ${RUN_NAME}"
echo " Symbols: ${SYMBOLS}"
echo " Training: epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LR} wd=${WD} rw=${RW} seq=${SEQ}"
echo " Eval periods: ${EVAL_PERIODS}d"
echo " Output: ${CKPT_DIR}"
echo "=========================================="
echo ""

echo "=== Phase 1: BF16 Training ==="
$PYTHON $TRAIN \
    --symbols "$SYMBOLS" \
    --crypto-symbols "" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight-decay "$WD" \
    --return-weight "$RW" \
    --sequence-length "$SEQ" \
    --seed "$SEED" \
    --hidden-dim "$HIDDEN" \
    --num-layers "$LAYERS" \
    --num-heads "$HEADS" \
    --dropout 0.1 \
    --grad-clip 1.0 \
    --warmup-steps 100 \
    --maker-fee "$EVAL_FEE" \
    --max-leverage 2.0 \
    --margin-annual-rate "$EVAL_MARGIN" \
    --fill-buffer-pct 0.0005 \
    --decision-lag-bars 1 \
    --fill-temperature 5e-4 \
    --logits-softcap 12.0 \
    --forecast-horizons 1 \
    --checkpoint-name "$RUN_NAME" \
    --eval-periods "$EVAL_PERIODS" \
    --eval-after-epoch 5 \
    --min-edge "$MIN_EDGE" \
    --max-positions "$MAX_POS" \
    --max-hold-hours "$MAX_HOLD" \
    --eval-fee-rate "$EVAL_FEE" \
    --eval-margin-rate "$EVAL_MARGIN"

echo ""
echo "=== Phase 2: Detailed Multi-Period Evaluation ==="
$PYTHON $SWEEP \
    --checkpoint-dir "$CKPT_DIR" \
    --symbols "$EVAL_SYMBOLS" \
    --fee-rate "$EVAL_FEE" \
    --margin-rate "$EVAL_MARGIN" \
    --no-close-at-eod \
    --max-positions "$MAX_POS" \
    --max-hold-hours "$MAX_HOLD" \
    --min-edge "$MIN_EDGE" \
    --holdout-days "$EVAL_PERIODS"

echo ""
echo "=== Live8 Candidate Complete ==="
echo "Checkpoint: ${CKPT_DIR}"
echo "Results: ${CKPT_DIR}/epoch_sweep_portfolio.json"
