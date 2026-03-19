#!/bin/bash
# Full multivariate forecasting pipeline
# Run after tune_multivariate_per_symbol.py completes

set -e

echo "=========================================="
echo "MULTIVARIATE FORECASTING PIPELINE"
echo "=========================================="

# Step 1: Verify multivariate config exists
if [ ! -f "reports/multivariate_config.json" ]; then
    echo "ERROR: reports/multivariate_config.json not found!"
    echo "Run: python analysis/tune_multivariate_per_symbol.py --validation-days 15 --output reports/multivariate_config.json"
    exit 1
fi

echo "Step 1: Multivariate config found"
cat reports/multivariate_config.json | python -c "import json,sys; d=json.load(sys.stdin); print(f'  Symbols: {d[\"summary\"][\"n_symbols\"]}, Stocks with multivariate: {d[\"summary\"][\"stocks_using_multivariate\"]}')"

# Step 2: Backup old forecasts
echo ""
echo "Step 2: Backing up old forecast cache..."
BACKUP_DIR="strategytraining/forecast_cache_backup_$(date +%Y%m%d_%H%M%S)"
if [ -d "strategytraining/forecast_cache" ]; then
    mv strategytraining/forecast_cache "$BACKUP_DIR"
    echo "  Backed up to: $BACKUP_DIR"
else
    echo "  No existing cache to backup"
fi
mkdir -p strategytraining/forecast_cache

# Step 3: Regenerate forecasts with new multivariate settings
echo ""
echo "Step 3: Regenerating Chronos2 forecasts with new multivariate settings..."
echo "  This may take 30-60 minutes for all symbols..."
python update_chronos_forecasts.py 2>&1 | tail -20

# Verify forecasts
FORECAST_COUNT=$(ls strategytraining/forecast_cache/*.parquet 2>/dev/null | wc -l)
echo "  Generated forecasts for $FORECAST_COUNT symbols"

# Step 4: Train neural model for 1 epoch
echo ""
echo "Step 4: Training neural model with updated forecasts (1 epoch)..."
python neural_trade_stock_e2e.py \
    --mode train \
    --epochs 1 \
    --batch-size 96 \
    --sequence-length 128 \
    --fit-all-data \
    --run-name "multivariate_retrain_$(date +%Y%m%d)" \
    --validation-days 0 \
    2>&1 | tee reports/multivariate_training.log

# Find the new checkpoint
NEW_CHECKPOINT=$(ls -td neuraldailytraining/checkpoints/*/ | head -1)
echo "  New checkpoint: $NEW_CHECKPOINT"

# Step 5: Run market simulator
echo ""
echo "Step 5: Running market simulator validation..."
CHECKPOINT_PATH="${NEW_CHECKPOINT}best.pt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH=$(find "$NEW_CHECKPOINT" -name "*.pt" | head -1)
fi

if [ -f "$CHECKPOINT_PATH" ]; then
    python -m neuraldailymarketsimulator.simulator \
        --checkpoint "$CHECKPOINT_PATH" \
        --days 30 \
        --initial-cash 1.0 \
        2>&1 | tee reports/multivariate_simulation.log
else
    echo "  WARNING: Could not find checkpoint at $CHECKPOINT_PATH"
fi

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "Results:"
echo "  - Multivariate config: reports/multivariate_config.json"
echo "  - Training log: reports/multivariate_training.log"
echo "  - Simulation log: reports/multivariate_simulation.log"
echo "  - Forecast backup: $BACKUP_DIR"
