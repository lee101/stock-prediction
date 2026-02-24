#!/bin/bash
set -e
PYTHON=".venv312/bin/python"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"

echo "=== LONGS ONLY (5 stocks) - fnoise01 ep5 ==="
$PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01 \
    --symbols NVDA,PLTR,GOOG,NET,DBX --max-positions 5 --max-hold-hours 6 --min-edge 0.0 \
    --decision-lag-bars 1 --holdout-days 30 --leverage 2.0 --epoch 5

echo "=== SHORTS ONLY (4 stocks) - fnoise01 ep5 ==="
$PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01 \
    --symbols TRIP,EBAY,MTCH,NYT --max-positions 4 --max-hold-hours 6 --min-edge 0.0 \
    --decision-lag-bars 1 --holdout-days 30 --leverage 2.0 --epoch 5

echo "=== LONGS ONLY 90d ==="
$PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01 \
    --symbols NVDA,PLTR,GOOG,NET,DBX --max-positions 5 --max-hold-hours 6 --min-edge 0.0 \
    --decision-lag-bars 1 --holdout-days 90 --leverage 2.0 --epoch 5

echo "=== SHORTS ONLY 90d ==="
$PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01 \
    --symbols TRIP,EBAY,MTCH,NYT --max-positions 4 --max-hold-hours 6 --min-edge 0.0 \
    --decision-lag-bars 1 --holdout-days 90 --leverage 2.0 --epoch 5

echo "=== DONE ==="
