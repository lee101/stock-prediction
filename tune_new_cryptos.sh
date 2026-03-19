#!/bin/bash
# Script to run Chronos2 hyperparameter tuning and preaugmentation sweeps
# for newly added crypto symbols: AAVEUSD, LINKUSD, SOLUSD

set -e  # Exit on error

echo "=================================="
echo "Chronos2 Tuning for New Cryptos"
echo "=================================="
echo ""

# Activate virtual environment
source .venv313/bin/activate

SYMBOLS="AAVEUSD LINKUSD SOLUSD"
echo "Symbols to process: $SYMBOLS"
echo ""

# Step 1: Run hyperparameter tuning for AAVEUSD (LINKUSD and SOLUSD already have configs)
echo "Step 1: Running Chronos2 hyperparameter tuning for AAVEUSD..."
echo "-------------------------------------------------------------"
python benchmark_chronos2.py \
    --symbols AAVEUSD \
    --search-method direct \
    --data-dir trainingdata \
    --model-id amazon/chronos-2 \
    --device-map cuda \
    --auto-context-lengths \
    --auto-context-min 128 \
    --auto-context-max 2048 \
    --auto-context-step 64 \
    --direct-sample-counts 256 512 1024 2048 \
    --direct-aggregations median mean trimmed_mean_10 \
    --direct-scalers none meanstd \
    --direct-batch-sizes 64 128 \
    --quantile-levels 0.1 0.5 0.9 \
    --torch-dtype bfloat16 \
    --update-hyperparams \
    --hyperparam-root hyperparams/chronos2 \
    --output-dir chronos2_benchmarks \
    --direct-maxfun 50 \
    --direct-objective test_pct_mae \
    --verbose

echo ""
echo "Step 1 complete: AAVEUSD hyperparameters generated"
echo ""

# Step 2: Run preaugmentation sweeps for all three symbols
echo "Step 2: Running preaugmentation sweeps for AAVEUSD, LINKUSD, SOLUSD..."
echo "-----------------------------------------------------------------------"
python preaug_sweeps/evaluate_preaug_chronos.py \
    --symbols AAVEUSD LINKUSD SOLUSD \
    --hyperparam-root hyperparams/chronos2 \
    --best-selection-root hyperparams/best \
    --selection-metric mae_percent \
    --output-dir preaugstrategies/chronos2 \
    --mirror-best-dir preaugstrategies/best \
    --data-dir trainingdata \
    --model-id amazon/chronos-2 \
    --device-map cuda \
    --torch-dtype bfloat16 \
    --benchmark-cache-dir chronos2_benchmarks/preaug_cache \
    --report-dir preaug_sweeps/reports \
    --verbose

echo ""
echo "Step 2 complete: Preaugmentation strategies selected"
echo ""

# Step 3: Update best configs (copy chronos2 configs to best/ with preaug)
echo "Step 3: Updating best configs with preaugmentation results..."
echo "-------------------------------------------------------------"

for SYMBOL in $SYMBOLS; do
    PREAUG_FILE="preaugstrategies/best/${SYMBOL}.json"
    HYPERPARAM_FILE="hyperparams/chronos2/${SYMBOL}.json"
    BEST_FILE="hyperparams/best/${SYMBOL}.json"

    if [ -f "$PREAUG_FILE" ]; then
        echo "  Copying $PREAUG_FILE to $BEST_FILE"
        cp "$PREAUG_FILE" "$BEST_FILE"
    elif [ -f "$HYPERPARAM_FILE" ]; then
        echo "  Warning: No preaug file for $SYMBOL, using base hyperparam config"
        cp "$HYPERPARAM_FILE" "$BEST_FILE"
    else
        echo "  Error: No config found for $SYMBOL"
    fi
done

echo ""
echo "Step 3 complete: Best configs updated"
echo ""

echo "=================================="
echo "All steps completed successfully!"
echo "=================================="
echo ""
echo "Summary:"
echo "  - Hyperparameters generated: hyperparams/chronos2/AAVEUSD.json"
echo "  - Preaugmentation configs: preaugstrategies/chronos2/{AAVEUSD,LINKUSD,SOLUSD}.json"
echo "  - Best configs updated: hyperparams/best/{AAVEUSD,LINKUSD,SOLUSD}.json"
echo ""
echo "You can now use these symbols with Chronos2 forecasting!"
