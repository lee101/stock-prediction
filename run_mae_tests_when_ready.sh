#!/bin/bash
# Run MAE tests when GPU becomes available

set -e

echo "================================================================================"
echo "MAE Testing Script - Waiting for GPU"
echo "================================================================================"
echo ""

# Function to check if GPU is available
check_gpu() {
    nvidia-smi > /dev/null 2>&1
    return $?
}

# Function to check if processes are using GPU
check_gpu_usage() {
    fuser /dev/nvidia0 2>&1 | grep -q "administrator"
    return $?
}

# Wait for GPU to be available
echo "Checking GPU status..."
if ! check_gpu; then
    echo "❌ GPU not detected by nvidia-smi"
    echo "   There may be a driver issue or GPU is crashed"
    echo "   Try: sudo nvidia-smi -r (reset GPU)"
    exit 1
fi

echo "✅ GPU detected"

# Check if something is using it
if check_gpu_usage; then
    echo ""
    echo "⚠️  GPU is currently in use:"
    fuser -v /dev/nvidia* 2>&1 | head -10
    echo ""
    echo "Options:"
    echo "  1. Wait for current process to finish"
    echo "  2. Kill process: kill -9 <PID>"
    echo "  3. Run this script later"
    echo ""
    read -p "Wait for GPU? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Run this script when GPU is free."
        exit 0
    fi

    echo "Waiting for GPU to become available..."
    while check_gpu_usage; do
        echo -n "."
        sleep 10
    done
    echo ""
    echo "✅ GPU is now free!"
fi

# Give GPU a moment to recover
echo "Waiting 5 seconds for GPU to stabilize..."
sleep 5

# Activate venv
echo ""
echo "Activating .venv313..."
if [ -f ".venv313/bin/activate" ]; then
    source .venv313/bin/activate
else
    echo "❌ .venv313 not found"
    exit 1
fi

echo "✅ Virtual environment activated"
echo ""

# Run tests
echo "================================================================================"
echo "Running MAE Tests"
echo "================================================================================"
echo ""

echo "Test 1: Quick KVCache fix verification (~30 sec)"
echo "-----------------------------------------------------------"
python tests/test_kvcache_fix.py
echo ""

echo "Test 2: Toto MAE on training data (~2-3 min)"
echo "-----------------------------------------------------------"
python tests/test_mae_integration.py
echo ""

echo "Test 3: Both models MAE test (~5-10 min)"
echo "-----------------------------------------------------------"
python tests/test_mae_both_models.py
echo ""

echo "================================================================================"
echo "All Tests Complete!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - Check: tests/mae_baseline.txt"
echo "  - Check: tests/mae_baseline_both_models.txt"
echo ""
echo "Verify:"
echo "  - MAE < 10% of mean price = Good"
echo "  - MAPE < 15% = Acceptable for financial forecasting"
echo ""
