#!/bin/bash
# Batch collection script for position sizing dataset
# Processes symbols in groups to avoid memory issues

set -e

# Configuration
DATA_DIR="trainingdata/train"
OUTPUT_DIR="strategytraining/datasets"
WINDOW_DAYS=7
STRIDE_DAYS=7
BATCH_SIZE=10

# Get all available symbols
SYMBOLS=($(ls $DATA_DIR/*.csv | xargs -n1 basename | sed 's/\.csv//'))
TOTAL_SYMBOLS=${#SYMBOLS[@]}

echo "Found $TOTAL_SYMBOLS symbols to process"
echo "Processing in batches of $BATCH_SIZE"

# Process in batches
BATCH_NUM=0
for ((i=0; i<$TOTAL_SYMBOLS; i+=BATCH_SIZE)); do
    BATCH_NUM=$((BATCH_NUM + 1))
    END=$((i + BATCH_SIZE))
    if [ $END -gt $TOTAL_SYMBOLS ]; then
        END=$TOTAL_SYMBOLS
    fi

    # Get batch symbols
    BATCH_SYMBOLS="${SYMBOLS[@]:i:BATCH_SIZE}"

    echo ""
    echo "========================================"
    echo "Processing Batch $BATCH_NUM"
    echo "Symbols $(($i+1)) to $END of $TOTAL_SYMBOLS"
    echo "========================================"

    # Run collection for this batch
    python strategytraining/collect_position_sizing_dataset.py \
        --data-dir $DATA_DIR \
        --output-dir $OUTPUT_DIR \
        --window-days $WINDOW_DAYS \
        --stride-days $STRIDE_DAYS \
        --symbols $BATCH_SYMBOLS \
        --dataset-name "batch_${BATCH_NUM}_dataset"

    echo "Batch $BATCH_NUM complete"
done

echo ""
echo "========================================"
echo "All batches complete!"
echo "========================================"
echo "Collected $BATCH_NUM datasets in $OUTPUT_DIR"
echo ""
echo "To merge all batches, run:"
echo "  python strategytraining/merge_datasets.py"
