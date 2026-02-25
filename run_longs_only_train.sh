#!/bin/bash
set -e
PYTHON=".venv312/bin/python"
TRAIN="unified_hourly_experiment/train_unified_policy.py"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"
SYMS="NVDA,PLTR,GOOG,NET,DBX"

echo "=== TRAINING: longs-only 5 stocks, fnoise01, wd03, rw10, seq48 ==="
$PYTHON $TRAIN \
    --symbols "$SYMS" \
    --crypto-symbols "" \
    --forecast-horizons 1 \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-5 \
    --weight-decay 0.03 \
    --return-weight 0.10 \
    --sequence-length 48 \
    --hidden-dim 512 \
    --num-heads 8 \
    --num-layers 6 \
    --decision-lag-bars 1 \
    --feature-noise-std 0.01 \
    --seed 1337 \
    --checkpoint-name longs5_lag1_fnoise01 \
    --no-compile

echo "=== EVAL: longs-only trained model ==="
for DAYS in 30 60 90; do
    echo "--- longs5_trained ${DAYS}d ---"
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/longs5_lag1_fnoise01 \
        --symbols "$SYMS" --max-positions 5 --max-hold-hours 6 --min-edge 0.0 \
        --decision-lag-bars 1 --holdout-days $DAYS --leverage 2.0
done

echo "=== DONE ==="
