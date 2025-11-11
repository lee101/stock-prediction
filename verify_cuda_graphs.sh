#!/bin/bash
# Quick verification script to check if CUDA graphs are working

set -e

# Activate virtual environment
if [ -f ".venv313/bin/activate" ]; then
    source .venv313/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "Toto CUDA Graphs Verification"
echo "=========================================="
echo ""

# Check environment
echo "1. Checking environment..."
if [ -z "$TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS" ]; then
    echo "   ⚠️  TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS not set"
    echo "   Setting it now..."
    export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
fi
echo "   TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=$TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"

# Check CUDA availability
echo ""
echo "2. Checking CUDA..."
python3 -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run the quick test
echo ""
echo "3. Running KVCache CUDA graph compatibility test..."
echo ""
python tests/test_kvcache_fix.py 2>&1 | grep -A5 "RESULTS:"

echo ""
echo "=========================================="
echo "What to Look For in Your Backtest Logs:"
echo "=========================================="
echo ""
echo "❌ BAD (before fix):"
echo "   'skipping cudagraphs due to incompatible op aten._local_scalar_dense.default'"
echo ""
echo "✅ GOOD (after fix):"
echo "   No such warnings"
echo "   (Some 'mutated inputs' warnings are OK - those are from cache updates)"
echo ""
echo "=========================================="
echo "Run Your Backtest:"
echo "=========================================="
echo "python backtest_test3_inline.py 2>&1 | grep -i cudagraph"
echo ""
