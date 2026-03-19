#!/usr/bin/env bash
# Quick retune of core symbols with torch compiled Chronos2
# Useful for testing and validating the compiled model before full retune
#
# Usage:
#   ./scripts/retune_chronos2_quick.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Quick configuration - fewer context lengths, high fixed batch size
CONTEXT_LENGTHS="512 1024 2048"
BATCH_SIZE="2048"  # Fixed high batch size for 5090 32GB
VAL_WINDOW=15
TEST_WINDOW=15

# Core symbols to test
CORE_SYMBOLS="BTCUSD ETHUSD AAPL MSFT NVDA TSLA"

# Always enable torch compile
export TORCH_COMPILED=1
export CHRONOS_COMPILE=1

echo "=============================================="
echo "Chronos2 Quick Retune (Compiled)"
echo "=============================================="
echo "Symbols: $CORE_SYMBOLS"
echo "Context lengths: $CONTEXT_LENGTHS"
echo "Batch size (fixed): $BATCH_SIZE"
echo "Torch compile: ENABLED"
echo "GPU: RTX 5090 32GB"
echo "=============================================="
echo ""

python analysis/evaluate_chronos2_hyperparams.py \
    --data-dir trainingdata \
    --symbols $CORE_SYMBOLS \
    --context-lengths ${CONTEXT_LENGTHS} \
    --batch-sizes ${BATCH_SIZE} \
    --val-window "$VAL_WINDOW" \
    --test-window "$TEST_WINDOW" \
    --optimizer grid \
    --output-dir hyperparams/chronos2 \
    --results-json analysis/chronos2_tuning_results_quick.json \
    --log-level INFO

echo ""
echo "✓ Quick retune complete!"
echo "  Results: hyperparams/chronos2/"
echo "  Report: analysis/chronos2_tuning_results_quick.json"
echo ""
echo "To run full retune:"
echo "  ./scripts/retune_chronos2_all.sh both"
