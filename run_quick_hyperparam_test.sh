#!/bin/bash
# Quick hyperparameter testing on multiple stock pairs
# Usage: ./run_quick_hyperparam_test.sh [symbols...]
# Example: ./run_quick_hyperparam_test.sh AAPL MSFT BTCUSD

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Quick Hyperparameter Testing${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Default symbols if none provided
if [ $# -eq 0 ]; then
    SYMBOLS="AAPL MSFT NVDA BTCUSD TSLA"
    echo -e "${YELLOW}No symbols specified, using defaults:${NC} $SYMBOLS"
else
    SYMBOLS="$@"
    echo -e "${YELLOW}Testing symbols:${NC} $SYMBOLS"
fi

echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}Warning: .venv not found, using system Python${NC}"
fi

echo ""
echo -e "${BLUE}Step 1: Running quick hyperparameter tests${NC}"
echo -e "${BLUE}===========================================${NC}"
python test_hyperparameters_quick.py --symbols $SYMBOLS

echo ""
echo -e "${GREEN}✓ Testing complete!${NC}"
echo ""

echo -e "${BLUE}Step 2: Analyzing results${NC}"
echo -e "${BLUE}=========================${NC}"
python analyze_hyperparam_results.py --results-dir hyperparams_quick --export-csv quick_results_$(date +%Y%m%d_%H%M%S).csv

echo ""
echo -e "${GREEN}✓ Analysis complete!${NC}"
echo ""

echo -e "${BLUE}Results Summary${NC}"
echo -e "${BLUE}===============${NC}"
echo "Results saved to:"
echo "  - JSON configs: hyperparams_quick/{kronos,toto}/"
echo "  - CSV export: quick_results_*.csv"
echo ""
echo "Next steps:"
echo "  1. Review the analysis output above"
echo "  2. Check JSON files for best configurations"
echo "  3. Run extended test on promising symbols:"
echo "     ${GREEN}python test_hyperparameters_extended.py --symbols SYMBOL --max-kronos-configs 100${NC}"
echo ""
echo -e "${GREEN}Done!${NC} 🎉"
