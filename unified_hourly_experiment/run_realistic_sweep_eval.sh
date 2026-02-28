#!/bin/bash
set -e
cd /home/lee/code/stock
export PYTHONUNBUFFERED=1

PYTHON=".venv312/bin/python -u"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"
COMMON="--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --holdout-days 30 --no-close-at-eod --leverage 2.0 --margin-rate 0.0625 --max-positions 9 --decision-lag-bars 1 --fee-rate 0.001"

for dir in realistic_rw010 realistic_rw015 realistic_rw020 realistic_cosine realistic_sortino_dd realistic_fillbuf001; do
    echo "=== Sweeping $dir ==="
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/$dir $COMMON
    echo ""
done

echo "=== All sweeps complete ==="
