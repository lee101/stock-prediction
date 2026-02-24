#!/bin/bash
set -e
PYTHON=".venv312/bin/python"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"

echo "=== EXPANDED LONGS (8 stocks) pos=5 - fnoise01 ep5 ==="
for DAYS in 30 90; do
    echo "--- expanded_longs ${DAYS}d ---"
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01 \
        --symbols NVDA,PLTR,GOOG,NET,DBX,META,MSFT,AAPL --max-positions 5 --max-hold-hours 6 --min-edge 0.0 \
        --decision-lag-bars 1 --holdout-days $DAYS --leverage 2.0 --epoch 5
done

echo "=== EXPANDED LONGS (8 stocks) pos=8 - fnoise01 ep5 ==="
for DAYS in 30 90; do
    echo "--- expanded_pos8 ${DAYS}d ---"
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01 \
        --symbols NVDA,PLTR,GOOG,NET,DBX,META,MSFT,AAPL --max-positions 8 --max-hold-hours 6 --min-edge 0.0 \
        --decision-lag-bars 1 --holdout-days $DAYS --leverage 2.0 --epoch 5
done

echo "=== ORIGINAL 5 LONGS with pos=3 ==="
for DAYS in 30 90; do
    echo "--- orig5_pos3 ${DAYS}d ---"
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01 \
        --symbols NVDA,PLTR,GOOG,NET,DBX --max-positions 3 --max-hold-hours 6 --min-edge 0.0 \
        --decision-lag-bars 1 --holdout-days $DAYS --leverage 2.0 --epoch 5
done

echo "=== DONE ==="
