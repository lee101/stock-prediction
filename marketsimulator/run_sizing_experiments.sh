#!/bin/bash
# Run comprehensive sizing strategy experiments
#
# Usage:
#   ./marketsimulator/run_sizing_experiments.sh [quick|full]

set -e

MODE=${1:-quick}

# Enable FAST_SIMULATE for faster backtests
export MARKETSIM_FAST_SIMULATE=1

echo "=================================================="
echo "Position Sizing Strategy Experiments"
echo "Mode: $MODE"
echo "=================================================="

# Define symbol sets
CRYPTO_SYMBOLS="BTCUSD ETHUSD SOLUSD"
STOCK_SYMBOLS="AAPL MSFT AMZN NVDA META GOOG"
ETF_SYMBOLS="SPY QQQ IWM"

# Define strategy sets
CORE_STRATEGIES="fixed_50 kelly_25 optimal_f"
ALL_STRATEGIES="fixed_25 fixed_50 fixed_75 fixed_100 kelly_10 kelly_25 kelly_50 voltarget_10 voltarget_15 riskparity_5 riskparity_10 optimal_f"

if [ "$MODE" == "quick" ]; then
    echo "Quick mode: testing core strategies on limited symbols"
    SYMBOLS="BTCUSD AAPL"
    STRATEGIES=$CORE_STRATEGIES
else
    echo "Full mode: testing all strategies on all symbols"
    SYMBOLS="$CRYPTO_SYMBOLS $STOCK_SYMBOLS $ETF_SYMBOLS"
    STRATEGIES=$ALL_STRATEGIES
fi

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="marketsimulator/results/experiment_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run experiments
python marketsimulator/test_sizing_strategies.py \
    --symbols $SYMBOLS \
    --strategies $STRATEGIES \
    --output-dir "$OUTPUT_DIR"

# Check if results were generated
if [ -f "$OUTPUT_DIR"/*.csv ]; then
    echo ""
    echo "=================================================="
    echo "Experiments completed successfully!"
    echo "Results in: $OUTPUT_DIR"
    echo "=================================================="

    # Show quick summary of best performers
    echo ""
    echo "Top 5 strategies by Sharpe ratio:"
    head -1 "$OUTPUT_DIR"/*.csv
    tail -n +2 "$OUTPUT_DIR"/*.csv | sort -t',' -k4 -rn | head -5

else
    echo "ERROR: No results generated"
    exit 1
fi
