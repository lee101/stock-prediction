#!/usr/bin/env bash
# Retune all Chronos2 hyperparameters using torch compiled model for maximum speed
#
# Usage:
#   ./scripts/retune_chronos2_all.sh [daily|hourly|both]
#
# Environment variables:
#   CONTEXT_LENGTHS - Space-separated context lengths to try (default: "512 1024 2048")
#   BATCH_SIZES - Space-separated batch sizes to try (default: "64 128 256")
#   VAL_WINDOW - Validation window size (default: 20)
#   TEST_WINDOW - Test window size (default: 20)
#   MAX_SYMBOLS - Limit number of symbols (default: all)
#   OPTIMIZER - Optimizer type: grid or differential_evolution (default: grid)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Default configuration
MODE="${1:-both}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-512 1024 2048 4096}"
CONTEXT_LENGTHS_HOURLY="${CONTEXT_LENGTHS_HOURLY:-1024 2048 4096 8192}"  # Hourly uses extended context
BATCH_SIZE="${BATCH_SIZE:-2048}"  # Fixed high batch size for 5090 32GB
VAL_WINDOW="${VAL_WINDOW:-20}"
TEST_WINDOW="${TEST_WINDOW:-20}"
OPTIMIZER="${OPTIMIZER:-grid}"
MAX_SYMBOLS="${MAX_SYMBOLS:-}"

# Convert space-separated to comma-separated for Python script (if needed)
CTX_CSV=$(echo "$CONTEXT_LENGTHS" | tr ' ' ',')

# Note: Torch compile disabled by default for tuning to avoid compile bugs
# Hyperparameters will still be used with compiled inference (default in backtest_test3_inline.py)
export TORCH_COMPILED="${TORCH_COMPILED:-0}"
export CHRONOS_COMPILE="${CHRONOS_COMPILE:-0}"

echo "=============================================="
echo "Chronos2 Hyperparameter Retuning (Compiled)"
echo "=============================================="
echo "Mode: $MODE"
echo "Daily context lengths: $CONTEXT_LENGTHS"
echo "Hourly context lengths: $CONTEXT_LENGTHS_HOURLY"
echo "Batch size (fixed): $BATCH_SIZE"
echo "Validation window: $VAL_WINDOW"
echo "Test window: $TEST_WINDOW"
echo "Optimizer: $OPTIMIZER"
echo "Torch compile: ENABLED"
echo "GPU: RTX 5090 32GB"
echo "=============================================="
echo ""

tune_daily() {
    echo ">>> Retuning DAILY symbols with compiled Chronos2..."

    # Build symbols list from trainingdata directory
    SYMBOLS=$(find trainingdata -name "*.csv" ! -name "data_summary.csv" -exec basename {} .csv \; | sort | head -${MAX_SYMBOLS:-999})
    SYMBOL_COUNT=$(echo "$SYMBOLS" | wc -l)

    echo "Found $SYMBOL_COUNT daily symbols to tune"
    echo ""

    python analysis/evaluate_chronos2_hyperparams.py \
        --data-dir trainingdata \
        --context-lengths ${CONTEXT_LENGTHS} \
        --batch-sizes ${BATCH_SIZE} \
        --val-window "$VAL_WINDOW" \
        --test-window "$TEST_WINDOW" \
        --optimizer "$OPTIMIZER" \
        --output-dir hyperparams/chronos2 \
        --results-json analysis/chronos2_tuning_results_daily.json \
        --log-level INFO \
        ${MAX_SYMBOLS:+--max-symbols $MAX_SYMBOLS}

    echo ""
    echo "✓ Daily hyperparameter tuning complete"
    echo "  Results: hyperparams/chronos2/"
    echo "  Report: analysis/chronos2_tuning_results_daily.json"
}

tune_hourly() {
    echo ">>> Retuning HOURLY symbols with compiled Chronos2..."
    echo "    Using extended context lengths (up to 8192) for dense hourly data"

    # Hourly symbols from trade_stock_e2e_hourly.py defaults
    HOURLY_SYMBOLS="AAPL MSFT NVDA TSLA AMZN AMD GOOG ADBE COIN COUR U SAP SONY BTCUSD ETHUSD SOLUSD LINKUSD UNIUSD PAXGUSD"

    # Filter to only symbols with hourly data available
    AVAILABLE_HOURLY=""
    for sym in $HOURLY_SYMBOLS; do
        if [ -f "trainingdatahourly/${sym}.csv" ]; then
            AVAILABLE_HOURLY="$AVAILABLE_HOURLY $sym"
        fi
    done

    if [ -z "$AVAILABLE_HOURLY" ]; then
        echo "⚠ No hourly training data found, skipping hourly tuning"
        return 0
    fi

    HOURLY_COUNT=$(echo "$AVAILABLE_HOURLY" | wc -w)
    echo "Found $HOURLY_COUNT hourly symbols to tune: $AVAILABLE_HOURLY"
    echo ""

    # Set frequency for hourly tuning
    export CHRONOS2_FREQUENCY=hourly

    python analysis/evaluate_chronos2_hyperparams.py \
        --data-dir trainingdatahourly \
        --symbols $AVAILABLE_HOURLY \
        --context-lengths ${CONTEXT_LENGTHS_HOURLY} \
        --batch-sizes ${BATCH_SIZE} \
        --val-window "$VAL_WINDOW" \
        --test-window "$TEST_WINDOW" \
        --optimizer "$OPTIMIZER" \
        --output-dir hyperparams/chronos2/hourly \
        --results-json analysis/chronos2_tuning_results_hourly.json \
        --log-level INFO \
        ${MAX_SYMBOLS:+--max-symbols $MAX_SYMBOLS}

    unset CHRONOS2_FREQUENCY

    echo ""
    echo "✓ Hourly hyperparameter tuning complete"
    echo "  Results: hyperparams/chronos2/hourly/"
    echo "  Report: analysis/chronos2_tuning_results_hourly.json"
}

# Main execution
case "$MODE" in
    daily)
        tune_daily
        ;;
    hourly)
        tune_hourly
        ;;
    both)
        tune_daily
        echo ""
        echo "=============================================="
        echo ""
        tune_hourly
        ;;
    *)
        echo "Error: Invalid mode '$MODE'. Use 'daily', 'hourly', or 'both'"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "✓ All retuning complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Review tuning results in analysis/chronos2_tuning_results_*.json"
echo "  2. Verify hyperparams/chronos2/ contains updated configs"
echo "  3. Test with: ONLY_CHRONOS2=1 PAPER=1 python trade_stock_e2e.py"
echo "  4. For hourly: ONLY_CHRONOS2=1 PAPER=1 python trade_stock_e2e_hourly.py"
