#!/bin/bash
# Quick runner for hyperparameter testing

set -e

echo "=== Hyperparameter Testing Script ==="
echo ""

# Parse command line arguments
SYMBOLS="${1:-AAPL MSFT}"  # Default to AAPL and MSFT if no symbols provided
MAX_KRONOS="${2:-50}"      # Default to 50 Kronos configs
MAX_TOTO="${3:-50}"        # Default to 50 Toto configs

echo "Testing symbols: $SYMBOLS"
echo "Max Kronos configs per symbol: $MAX_KRONOS"
echo "Max Toto configs per symbol: $MAX_TOTO"
echo ""

# Run the extended hyperparameter test
python3 test_hyperparameters_extended.py \
    --symbols $SYMBOLS \
    --max-kronos-configs $MAX_KRONOS \
    --max-toto-configs $MAX_TOTO

echo ""
echo "=== Testing Complete ==="
echo "Results saved to hyperparams_extended/"
echo ""
echo "To view results:"
echo "  - Kronos: ls hyperparams_extended/kronos/"
echo "  - Toto: ls hyperparams_extended/toto/"
