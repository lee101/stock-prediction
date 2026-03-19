#!/bin/bash
# Script to run after forecast regeneration completes
# Wait for forecasts, then run training and market simulation

set -e
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

echo "=========================================="
echo "WAITING FOR FORECAST REGENERATION"
echo "=========================================="

# Wait for forecasts to complete (321 symbols expected)
EXPECTED_SYMBOLS=321
while true; do
    COMPLETED=$(ls strategytraining/forecast_cache/*.parquet 2>/dev/null | wc -l)
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed: $COMPLETED / $EXPECTED_SYMBOLS symbols"

    if [ "$COMPLETED" -ge "$EXPECTED_SYMBOLS" ]; then
        echo "All forecasts completed!"
        break
    fi

    # Check if forecast process is still running
    if ! pgrep -f "update_chronos_forecasts.py" > /dev/null; then
        echo "Warning: Forecast process not running, proceeding with $COMPLETED symbols"
        break
    fi

    sleep 120  # Check every 2 minutes
done

# Show final forecast stats
echo ""
echo "=========================================="
echo "FORECAST REGENERATION COMPLETE"
echo "=========================================="
FORECAST_COUNT=$(ls strategytraining/forecast_cache/*.parquet 2>/dev/null | wc -l)
echo "Generated forecasts for $FORECAST_COUNT symbols"

# Check how many use multivariate
echo "Checking multivariate usage..."
grep -c "multivariate=True" reports/forecast_regeneration.log || echo "0 multivariate"
grep -c "multivariate=False" reports/forecast_regeneration.log || echo "0 univariate"

# Step 1: Train neural model for 1 epoch
echo ""
echo "=========================================="
echo "STEP 1: TRAINING NEURAL MODEL (1 EPOCH)"
echo "=========================================="
RUN_NAME="multivariate_retrain_$(date +%Y%m%d_%H%M%S)"
python neural_trade_stock_e2e.py \
    --mode train \
    --epochs 1 \
    --batch-size 96 \
    --sequence-length 128 \
    --fit-all-data \
    --run-name "$RUN_NAME" \
    --validation-days 0 \
    2>&1 | tee reports/multivariate_training.log

# Find the new checkpoint
echo ""
echo "Finding new checkpoint..."
NEW_CHECKPOINT_DIR=$(ls -td neuraldailytraining/checkpoints/*/ 2>/dev/null | head -1)
echo "New checkpoint directory: $NEW_CHECKPOINT_DIR"

# Find the best.pt or any .pt file
CHECKPOINT_PATH="${NEW_CHECKPOINT_DIR}best.pt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH=$(find "$NEW_CHECKPOINT_DIR" -name "*.pt" -type f | head -1)
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Could not find checkpoint file"
    exit 1
fi
echo "Using checkpoint: $CHECKPOINT_PATH"

# Step 2: Run market simulator for 10 days validation
echo ""
echo "=========================================="
echo "STEP 2: MARKET SIMULATOR (10 DAYS)"
echo "=========================================="
python -m neuraldailymarketsimulator.simulator \
    --checkpoint "$CHECKPOINT_PATH" \
    --days 10 \
    --initial-cash 1.0 \
    2>&1 | tee reports/multivariate_simulation.log

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "Results:"
echo "  - Multivariate config: reports/multivariate_config.json"
echo "  - Training log: reports/multivariate_training.log"
echo "  - Simulation log: reports/multivariate_simulation.log"
echo "  - Checkpoint: $CHECKPOINT_PATH"
