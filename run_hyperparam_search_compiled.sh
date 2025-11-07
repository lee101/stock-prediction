#!/bin/bash
#
# Comprehensive hyperparameter search for compiled Toto and Kronos models
#
# This script runs extensive hyperparameter optimization across all symbols
# and only updates configs if we achieve better PnL.
#

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Compiled Model Hyperparameter Optimization                   ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo ""

# Default values
TRIALS=${TRIALS:-100}
WORKERS=${WORKERS:-2}
MODEL=${MODEL:-toto}
SYMBOLS=${SYMBOLS:-""}

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
        --model)
            MODEL="$2"
            shift 2
            ;;
        --symbols)
            shift
            SYMBOLS="$@"
            break
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --trials N      Number of optimization trials per model (default: 100)"
            echo "  --workers N     Number of parallel workers (default: 2)"
            echo "  --model MODEL   Which model: toto, kronos, or both (default: toto)"
            echo "  --symbols ...   Specific symbols to optimize (default: all)"
            echo ""
            echo "Examples:"
            echo "  # Optimize all symbols with 100 trials each"
            echo "  $0"
            echo ""
            echo "  # Optimize specific symbols with 200 trials"
            echo "  $0 --trials 200 --symbols BTCUSD ETHUSD AAPL"
            echo ""
            echo "  # Run with more workers for faster parallel execution"
            echo "  $0 --workers 4"
            echo ""
            echo "  # Optimize both toto and kronos"
            echo "  $0 --model both --trials 150"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Configuration:${NC}"
echo "  Trials per model: $TRIALS"
echo "  Parallel workers: $WORKERS"
echo "  Model: $MODEL"
if [ -n "$SYMBOLS" ]; then
    echo "  Symbols: $SYMBOLS"
else
    echo "  Symbols: ALL"
fi
echo ""

# Check if required dependencies are installed
echo -e "${BLUE}Checking dependencies...${NC}"

if ! python -c "import optuna" 2>/dev/null; then
    echo -e "${YELLOW}Installing optuna...${NC}"
    uv pip install optuna
fi

if ! python -c "import rich" 2>/dev/null; then
    echo -e "${YELLOW}Installing rich...${NC}"
    uv pip install rich
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"
echo ""

# Create output directory
mkdir -p hyperparams/optimized_compiled
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/hyperparam_optimization_compiled_${TIMESTAMP}.log"

echo -e "${BLUE}Starting optimization...${NC}"
echo "  Log file: $LOG_FILE"
echo ""

# Build command
CMD="python run_compiled_optimization_all.py"
CMD="$CMD --trials $TRIALS"
CMD="$CMD --workers $WORKERS"
CMD="$CMD --model $MODEL"
CMD="$CMD --save-summary results/compiled_optimization_${TIMESTAMP}.json"

if [ -n "$SYMBOLS" ]; then
    CMD="$CMD --symbols $SYMBOLS"
fi

# Run optimization
echo -e "${GREEN}Running: $CMD${NC}"
echo ""

$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ Optimization completed successfully!                       ║${NC}"
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo ""
    echo "  Log file: $LOG_FILE"
    echo "  Results: results/compiled_optimization_${TIMESTAMP}.json"
    echo ""
    echo -e "${YELLOW}Updated hyperparameters are in: hyperparams/best/${NC}"
    echo ""
else
    echo ""
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  ⚠ Optimization completed with errors                         ║${NC}"
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo ""
    echo "  Check log file: $LOG_FILE"
    exit 1
fi
