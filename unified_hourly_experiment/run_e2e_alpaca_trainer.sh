#!/usr/bin/env bash
# End-to-end Alpaca stock trader: train + multi-timeframe market simulator eval
# Tests profitability over 1d, 7d, 30d, 60d, 120d, 150d time windows
#
# Usage:
#   ./unified_hourly_experiment/run_e2e_alpaca_trainer.sh
#   ./unified_hourly_experiment/run_e2e_alpaca_trainer.sh --quick  # fewer epochs for testing
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python -u}"
TRAIN="unified_hourly_experiment/train_bf16_efficient.py"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"

# Symbols
SYMBOLS="NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT"
EVAL_SYMBOLS="$SYMBOLS"

# Training defaults
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-5}"
WD="${WD:-0.06}"
RW="${RW:-0.15}"
SEQ="${SEQ:-32}"
SEED="${SEED:-42}"
HIDDEN="${HIDDEN:-512}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"

# Eval periods
EVAL_PERIODS="1,7,30,60,120,150"
EVAL_FEE="0.001"
EVAL_MARGIN="0.0625"
MAX_POS="${MAX_POS:-7}"
MAX_HOLD="${MAX_HOLD:-6}"
MIN_EDGE="${MIN_EDGE:-0.0}"

# Quick mode for testing
if [[ "${1:-}" == "--quick" ]]; then
    EPOCHS=5
    EVAL_PERIODS="7,30"
    echo "=== QUICK MODE: ${EPOCHS} epochs, eval ${EVAL_PERIODS} ==="
fi

RUN_NAME="e2e_rw${RW//.}_wd${WD//.}_seq${SEQ}_s${SEED}"
CKPT_DIR="unified_hourly_experiment/checkpoints/${RUN_NAME}"

echo "=========================================="
echo " E2E Alpaca Stock Trader Training"
echo "=========================================="
echo " Run: ${RUN_NAME}"
echo " Symbols: ${SYMBOLS}"
echo " Architecture: h${HIDDEN} ${LAYERS}L ${HEADS}H"
echo " Training: epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LR} wd=${WD} rw=${RW} seq=${SEQ}"
echo " Eval periods: ${EVAL_PERIODS}d"
echo " Output: ${CKPT_DIR}"
echo "=========================================="
echo ""

# Phase 1: Training
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
    --maker-fee 0.001 \
    --max-leverage 2.0 \
    --margin-annual-rate 0.0625 \
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
# Re-run eval with full period sweep for all epochs
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
echo "=== Results Summary ==="
if [ -f "${CKPT_DIR}/epoch_sweep_portfolio.json" ]; then
    $PYTHON -c "
import json, sys
with open('${CKPT_DIR}/epoch_sweep_portfolio.json') as f:
    data = json.load(f)
results = data.get('results', [])
summaries = data.get('summaries', [])

print('\\nPer-Period Results:')
print(f'{\"Epoch\":>6s} {\"Period\":>6s} {\"Return\":>8s} {\"Sortino\":>8s} {\"MaxDD\":>8s} {\"Buys\":>6s} {\"WR%\":>6s}')
print('-' * 52)
for r in sorted(results, key=lambda x: (x['epoch'], x.get('holdout_days', 0))):
    print(f'{r[\"epoch\"]:6d} {r.get(\"period\",\"?\"):>6s} {r[\"return\"]:+7.2f}% {r[\"sortino\"]:8.2f} {r.get(\"max_drawdown\",0):7.1f}% {r.get(\"buys\",0):6d} {r.get(\"win_rate\",0):5.1f}%')

if summaries:
    print('\\nEpoch Summaries:')
    print(f'{\"Epoch\":>6s} {\"Smooth\":>8s} {\"AvgSort\":>8s} {\"AvgRet\":>8s} {\"WorstRet\":>8s} {\"AllPos\":>6s}')
    print('-' * 52)
    for s in sorted(summaries, key=lambda x: -x['smoothness']):
        status = 'YES' if s['all_positive'] else 'NO'
        print(f'{s[\"epoch\"]:6d} {s[\"smoothness\"]:8.2f} {s[\"avg_sortino\"]:8.2f} {s[\"avg_return\"]:+7.2f}% {s[\"worst_return\"]:+7.2f}% {status:>6s}')

    qualified = [s for s in summaries if s['all_positive']]
    if qualified:
        best = max(qualified, key=lambda x: x['smoothness'])
        print(f'\\nBEST QUALIFIED: Epoch {best[\"epoch\"]}')
        print(f'  Smoothness: {best[\"smoothness\"]:.2f}')
        print(f'  Avg Sortino: {best[\"avg_sortino\"]:.2f}')
        print(f'  Avg Return: {best[\"avg_return\"]:+.2f}%')
        for p, v in best['periods'].items():
            print(f'  {p}: ret={v[\"ret\"]:+.2f}% sort={v[\"sort\"]:.2f}')
    else:
        print('\\nWARNING: No epoch was profitable across ALL periods')
"
fi

echo ""
echo "=== E2E Training Complete ==="
echo "Checkpoint: ${CKPT_DIR}"
echo "Results: ${CKPT_DIR}/epoch_sweep_portfolio.json"
