#!/bin/bash
# Quick start script for running pre-augmentation sweeps

set -e

echo "=================================="
echo "Pre-Augmentation Sweep Runner"
echo "=================================="
echo ""

# Default values
SYMBOLS="${SYMBOLS:-ETHUSD UNIUSD BTCUSD}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LOOKBACK="${LOOKBACK:-64}"
HORIZON="${HORIZON:-30}"
VALIDATION_DAYS="${VALIDATION_DAYS:-30}"
BEST_DIR="${BEST_DIR:-preaugstrategies/best}"
SELECTION_METRIC="${SELECTION_METRIC:-mae_percent}"

echo "Configuration:"
echo "  Symbols: $SYMBOLS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Lookback: $LOOKBACK"
echo "  Horizon: $HORIZON"
echo "  Validation Days: $VALIDATION_DAYS"
echo "  Best Dir: $BEST_DIR"
echo "  Selection Metric: $SELECTION_METRIC"
echo ""

# Run the sweep
python3 preaug_sweeps/sweep_runner.py \
    --symbols $SYMBOLS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lookback $LOOKBACK \
    --horizon $HORIZON \
    --validation-days $VALIDATION_DAYS \
    --best-dir "$BEST_DIR" \
    --selection-metric "$SELECTION_METRIC" \
    "$@"

echo ""
echo "=================================="
echo "Sweep Complete!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  - preaug_sweeps/reports/"
echo "  - $BEST_DIR/"
echo ""
