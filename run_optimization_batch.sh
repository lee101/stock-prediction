#!/bin/bash
# Batch runner for per-stock optimization
# Usage: ./run_optimization_batch.sh [--symbols "AAPL NVDA SPY"] [--trials 50]

SYMBOLS="${SYMBOLS:-AAPL NVDA SPY AMD META TSLA}"
TRIALS="${TRIALS:-50}"
PARALLEL="${PARALLEL:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running optimization for symbols: $SYMBOLS"
echo "Trials per model: $TRIALS"
echo "Parallel mode: $PARALLEL"
echo ""

mkdir -p logs

if [ "$PARALLEL" = true ]; then
    # Run in parallel (requires GNU parallel or xargs)
    echo "$SYMBOLS" | tr ' ' '\n' | xargs -P 3 -I {} bash -c "
        echo 'Starting {}...'
        python optimize_per_stock.py --symbol {} --model both --trials $TRIALS > logs/{}_optimization.log 2>&1
        echo 'Completed {}'
    "
else
    # Run sequentially
    for symbol in $SYMBOLS; do
        echo "==============================================="
        echo "Optimizing $symbol"
        echo "==============================================="
        python optimize_per_stock.py --symbol "$symbol" --model both --trials "$TRIALS" 2>&1 | tee "logs/${symbol}_optimization.log"
        echo ""
    done
fi

echo ""
echo "Optimization complete! Results saved to hyperparams_optimized/"
echo "Logs saved to logs/"
