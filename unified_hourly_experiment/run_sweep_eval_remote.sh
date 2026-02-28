#!/bin/bash
set -e
cd /home/lee/code/stock
export PYTHONUNBUFFERED=1

PYTHON=".venv312/bin/python -u"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"
COMMON="--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --holdout-days 30 --no-close-at-eod --leverage 2.0 --margin-rate 0.0625 --max-positions 9 --decision-lag-bars 1 --fee-rate 0.001"

for dir in sweep_h256_4L sweep_h256_6L sweep_h384_6L sweep_h512_4L sweep_h512_8L sweep_seed42 sweep_seed123 sweep_seed7; do
    echo "=== $dir ==="
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/$dir $COMMON
    echo ""
done

echo "=== All remote evals complete ==="
