#!/bin/bash
set -e
cd /home/lee/code/stock
export PYTHONUNBUFFERED=1

PYTHON=".venv312/bin/python -u"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"
COMMON="--symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT --holdout-days 30 --no-close-at-eod --leverage 2.0 --margin-rate 0.0625 --max-positions 7 --decision-lag-bars 1 --fee-rate 0.001"

for dir in 7sym_rw010 7sym_rw015 7sym_rw020 7sym_rw015_wd05; do
    echo "=== $dir ==="
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/$dir $COMMON
    echo ""
done

echo "=== Also eval 9sym realistic_rw015 with 7sym blacklist ==="
$PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/realistic_rw015 $COMMON
echo ""

echo "=== All 7sym evals complete ==="
