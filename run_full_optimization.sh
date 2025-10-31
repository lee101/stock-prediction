#!/bin/bash
# Quick batch runner for full optimization
# Usage: ./run_full_optimization.sh [--trials 30] [--workers 3]

TRIALS="${TRIALS:-30}"
WORKERS="${WORKERS:-3}"
SYMBOLS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--trials 30] [--workers 3] [--symbols 'AAPL NVDA SPY']"
            exit 1
            ;;
    esac
done

echo "Starting full optimization..."
echo "Trials per model: $TRIALS"
echo "Parallel workers: $WORKERS"
echo ""

if [ -n "$SYMBOLS" ]; then
    python run_full_optimization.py --trials "$TRIALS" --workers "$WORKERS" --symbols $SYMBOLS --save-summary optimization_summary.json
else
    python run_full_optimization.py --trials "$TRIALS" --workers "$WORKERS" --save-summary optimization_summary.json
fi

echo ""
echo "✅ Optimization complete!"
echo "Results in: hyperparams_optimized_all/"
echo "Summary: optimization_summary.json"
