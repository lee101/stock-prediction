#!/bin/bash
set -e
PYTHON=".venv312/bin/python"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"
SYMS="NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT"

echo "=== COMBINED (fnoise01+drop02) - corrected short PnL ==="
for DAYS in 30 60 90 120; do
    echo "--- combined ${DAYS}d ---"
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01_drop02 \
        --symbols "$SYMS" --max-positions 9 --max-hold-hours 6 --min-edge 0.0 \
        --decision-lag-bars 1 --holdout-days $DAYS
done

echo "=== BASELINE (wd03 rw10 seq48) - corrected short PnL ==="
for DAYS in 30 60 90 120; do
    echo "--- baseline ${DAYS}d ---"
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_6L_lr1e5_wd03_rw10_seq48 \
        --symbols "$SYMS" --max-positions 9 --max-hold-hours 6 --min-edge 0.0 \
        --decision-lag-bars 1 --holdout-days $DAYS
done

echo "=== FNOISE01 only - corrected short PnL ==="
for DAYS in 30 60 90; do
    echo "--- fnoise01 ${DAYS}d ---"
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/top9_lag1_fnoise01 \
        --symbols "$SYMS" --max-positions 9 --max-hold-hours 6 --min-edge 0.0 \
        --decision-lag-bars 1 --holdout-days $DAYS
done

echo "=== ALL REEVAL COMPLETE ==="
