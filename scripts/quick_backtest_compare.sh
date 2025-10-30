#!/bin/bash
# Quick backtest comparison: compiled vs eager for both Toto and Kronos

set -e

PYTHON=".venv/bin/python"
SYMBOL="BTCUSD"
OUTPUT_DIR="evaltests/backtests_compile_comparison"

mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "BACKTEST COMPARISON - COMPILED VS EAGER"
echo "════════════════════════════════════════════════════════════════"
echo "Symbol: $SYMBOL"
echo "Output: $OUTPUT_DIR"
echo ""

# Function to run backtest
run_backtest() {
    local mode=$1
    local model=$2
    local output_suffix=$3

    echo "──────────────────────────────────────────────────────────────"
    echo "Running $model $mode mode..."
    echo "──────────────────────────────────────────────────────────────"

    # Set environment
    export COMPILED_MODELS_DIR=/vfast
    export TORCHINDUCTOR_CACHE_DIR=/vfast/torch_inductor

    if [ "$model" = "TOTO" ]; then
        if [ "$mode" = "EAGER" ]; then
            export TOTO_DISABLE_COMPILE=1
        else
            export TOTO_DISABLE_COMPILE=0
            export TOTO_COMPILE_MODE=max-autotune
        fi
        export FORCE_KRONOS=0  # Use Toto
    else
        # Kronos - always eager for now
        export TOTO_DISABLE_COMPILE=1
        export FORCE_KRONOS=1
    fi

    # Run backtest (simplified - just check if it runs)
    timeout 120 $PYTHON -c "
import sys
sys.path.insert(0, '/home/administrator/code/stock-prediction')

# Quick test - just load the models and make a prediction
print('Testing $model in $mode mode...')

if '$model' == 'TOTO':
    from src.models.toto_wrapper import TotoPipeline
    import numpy as np

    test_series = np.random.randn(256) * 10 + 100

    pipeline = TotoPipeline.from_pretrained(
        'Datadog/Toto-Open-Base-1.0',
        device_map='cuda',
        torch_compile=('$mode' == 'COMPILED'),
    )

    pred = pipeline.predict(test_series, prediction_length=1, num_samples=64)
    print(f'✓ Toto $mode prediction: {pred[0].mean():.2f}')

else:
    print('Kronos test skipped (requires full setup)')

print('✓ $model $mode test completed')
" 2>&1 | tail -20

    echo ""
}

# Test Toto Eager
run_backtest "EAGER" "TOTO" "toto_eager"

# Test Toto Compiled
run_backtest "COMPILED" "TOTO" "toto_compiled"

echo "════════════════════════════════════════════════════════════════"
echo "COMPARISON COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Review results above to determine:"
echo "  - Which mode completed successfully"
echo "  - Whether compiled mode has issues"
echo "  - Performance differences"
echo ""
