#!/bin/bash
# Run full strategy PnL dataset collection
# This will collect data for ALL strategies on ALL symbols

set -e

echo "=============================================================================="
echo "STRATEGY PNL DATASET COLLECTION"
echo "=============================================================================="
echo ""
echo "This will collect PnL data for:"
echo "  - 7 strategies (simple, all_signals, entry_takeprofit, highlow, maxdiff, ci_guard, buy_hold)"
echo "  - All symbols in trainingdata/train/"
echo "  - 52 rolling 7-day windows per symbol (1 year of data)"
echo ""
echo "Expected runtime: 2-4 hours"
echo "Expected output: ~500MB of data"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled"
    exit 1
fi

# Run collection
echo ""
echo "Starting collection..."
echo ""

python strategytraining/collect_strategy_pnl_dataset.py \
    --data-dir trainingdata/train \
    --output-dir strategytraining/datasets \
    --window-days 7 \
    --stride-days 7 \
    --dataset-name full_strategy_dataset \
    2>&1 | tee strategytraining/collection.log

echo ""
echo "=============================================================================="
echo "COLLECTION COMPLETE"
echo "=============================================================================="
echo ""
echo "Log saved to: strategytraining/collection.log"
echo ""
echo "Next steps:"
echo "  1. Find your dataset:"
echo "     ls strategytraining/datasets/full_strategy_dataset_*_metadata.json"
echo ""
echo "  2. Analyze results:"
echo "     python strategytraining/analyze_strategy_dataset.py \\"
echo "         strategytraining/datasets/full_strategy_dataset_TIMESTAMP"
echo ""
