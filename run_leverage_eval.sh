#!/bin/bash
set -e
PYTHON=".venv312/bin/python"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"
SYMS="NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT"

echo "=== FNOISE01 ep5 with leverage=2.0 ==="
for DAYS in 30 60 90; do
    echo "--- fnoise01 lev2 ${DAYS}d ---"
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01 \
        --symbols "$SYMS" --max-positions 9 --max-hold-hours 6 --min-edge 0.0 \
        --decision-lag-bars 1 --holdout-days $DAYS --leverage 2.0 --epoch 5
done

echo "=== COMBINED ep8 with leverage=2.0 ==="
for DAYS in 30 60 90; do
    echo "--- combined lev2 ${DAYS}d ---"
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01_drop02 \
        --symbols "$SYMS" --max-positions 9 --max-hold-hours 6 --min-edge 0.0 \
        --decision-lag-bars 1 --holdout-days $DAYS --leverage 2.0 --epoch 8
done

echo "=== DONE ==="
