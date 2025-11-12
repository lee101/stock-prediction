#!/bin/bash
# Production Pre-Augmentation Sweep
# Runs all strategies on all target symbols

set -e

echo "=============================================================================="
echo "FULL PRE-AUGMENTATION SWEEP"
echo "=============================================================================="
echo ""
echo "This will test ALL augmentation strategies on ETHUSD, UNIUSD, and BTCUSD"
echo "This will take several hours to complete!"
echo ""

# Configuration
SYMBOLS="ETHUSD UNIUSD BTCUSD"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LOOKBACK="${LOOKBACK:-64}"
HORIZON="${HORIZON:-30}"
VALIDATION_DAYS="${VALIDATION_DAYS:-30}"
BEST_DIR="${BEST_DIR:-preaugstrategies/best}"
SELECTION_METRIC="${SELECTION_METRIC:-mae_percent}"

echo "Configuration:"
echo "  Symbols: $SYMBOLS"
echo "  Strategies: ALL (baseline, percent_change, log_returns, differencing,"
echo "                   detrending, robust_scaling, minmax_standard, rolling_norm)"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Lookback: $LOOKBACK"
echo "  Horizon: $HORIZON"
echo "  Validation Days: $VALIDATION_DAYS"
echo "  Best Dir: $BEST_DIR"
echo "  Selection Metric: $SELECTION_METRIC"
echo ""
echo "Results will be saved to:"
echo "  - preaug_sweeps/results/"
echo "  - preaug_sweeps/reports/"
echo "  - $BEST_DIR/"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting sweep at $(date)"
echo "=============================================================================="
echo ""

# Run the sweep
.venv312/bin/python3 preaug_sweeps/sweep_runner.py \
    --symbols $SYMBOLS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lookback $LOOKBACK \
    --horizon $HORIZON \
    --validation-days $VALIDATION_DAYS \
    --best-dir "$BEST_DIR" \
    --selection-metric "$SELECTION_METRIC" \
    2>&1 | tee preaug_sweeps/logs/full_sweep_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "=============================================================================="
echo "Sweep finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Sweep completed successfully!"
    echo ""
    echo "Analyzing results..."
    .venv312/bin/python3 preaug_sweeps/analyze_results.py

    echo ""
    echo "Results summary:"
    echo "  - Full results: preaug_sweeps/reports/"
    echo "  - Best configs: $BEST_DIR/"
    echo "  - Training logs: preaug_sweeps/logs/"
    echo ""
else
    echo "✗ Sweep failed with exit code $EXIT_CODE"
    echo "Check logs in preaug_sweeps/logs/"
    exit $EXIT_CODE
fi
