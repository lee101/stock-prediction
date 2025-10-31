#!/bin/bash
#
# Complete Toto Optimization & Comparison Workflow
# ===================================================
# 1. Establish baseline (naive model)
# 2. Train stock-specific Toto models
# 3. Compare against Kronos
# 4. Generate optimization recommendations
#

set -e  # Exit on error

echo "================================================================================================"
echo "TOTO STOCK PREDICTION OPTIMIZATION PIPELINE"
echo "================================================================================================"
echo ""

# Configuration
DATA_DIR="trainingdata"
PRIORITY_ONLY=${1:-"false"}  # Set to "true" to train only priority stocks
FORECAST_HORIZON=${2:-64}    # Forecast horizon for comparison

# Step 1: Baseline Evaluation
echo ""
echo "Step 1: Evaluating Naive Baseline"
echo "================================================================================================"
if [ ! -f "tototraining/baseline_results.json" ]; then
    echo "Running baseline evaluation..."
    python tototraining/baseline_eval_simple.py
else
    echo "✅ Baseline results already exist (tototraining/baseline_results.json)"
    echo "   To regenerate, delete the file and rerun"
fi

# Step 2: Train Stock-Specific Models
echo ""
echo "Step 2: Training Stock-Specific Toto Models"
echo "================================================================================================"

if [ "$PRIORITY_ONLY" = "true" ]; then
    echo "Training PRIORITY stocks only..."
    uv run python tototraining/toto_retrain_wrapper.py --priority-only
else
    echo "Training ALL stocks..."
    echo "This may take several hours depending on the number of stocks!"
    echo ""
    read -p "Continue with full training? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        uv run python tototraining/toto_retrain_wrapper.py
    else
        echo "Skipping full training. Run with argument 'true' for priority stocks only."
        exit 0
    fi
fi

# Step 3: Compare vs Kronos
echo ""
echo "Step 3: Comparing Toto vs Kronos"
echo "================================================================================================"

# Check if we have any trained models
if [ ! -d "hyperparams/toto" ] || [ -z "$(ls -A hyperparams/toto/*.json 2>/dev/null)" ]; then
    echo "⚠️  No Toto models found in hyperparams/toto/"
    echo "   Skipping comparison step"
else
    echo "Running comparisons (forecast horizon = $FORECAST_HORIZON)..."
    uv run python tototraining/compare_toto_vs_kronos.py \
        --all \
        --forecast-horizon $FORECAST_HORIZON
fi

# Step 4: Generate Summary Report
echo ""
echo "Step 4: Generating Summary Report"
echo "================================================================================================"

if [ -f "tototraining/stock_models/training_summary.json" ]; then
    echo "Training Summary:"
    python -c "
import json
with open('tototraining/stock_models/training_summary.json', 'r') as f:
    data = json.load(f)
    successful = sum(1 for v in data.values() if v.get('success'))
    total = len(data)
    print(f'  Successful: {successful}/{total}')

    # Show best improvements
    improvements = [(k, v.get('improvement_pct', 0)) for k, v in data.items()
                   if v.get('success') and 'improvement_pct' in v]
    if improvements:
        improvements.sort(key=lambda x: x[1], reverse=True)
        print(f'\n  Top 5 Improvements over Baseline:')
        for stock, imp in improvements[:5]:
            print(f'    {stock}: {imp:+.1f}%')
"
fi

if [ -f "comparison_results/comparison_summary_h${FORECAST_HORIZON}.json" ]; then
    echo ""
    echo "Comparison Summary (Toto vs Kronos):"
    python -c "
import json
import sys
horizon = $FORECAST_HORIZON
with open(f'comparison_results/comparison_summary_h{horizon}.json', 'r') as f:
    data = json.load(f)
    results = data.get('results', {})
    toto_wins = sum(1 for v in results.values() if v.get('winner') == 'toto')
    kronos_wins = sum(1 for v in results.values() if v.get('winner') == 'kronos')
    total = len(results)

    print(f'  Toto wins: {toto_wins}/{total}')
    print(f'  Kronos wins: {kronos_wins}/{total}')

    # Calculate average improvement
    valid = [(k, v.get('improvement_pct', 0)) for k, v in results.items()
            if v.get('toto_mae') is not None and v.get('kronos_mae') is not None]
    if valid:
        avg_imp = sum(v for _, v in valid) / len(valid)
        print(f'  Average improvement: {avg_imp:+.1f}%')
"
fi

echo ""
echo "================================================================================================"
echo "OPTIMIZATION COMPLETE!"
echo "================================================================================================"
echo ""
echo "Results locations:"
echo "  - Baseline: tototraining/baseline_results.json"
echo "  - Trained models: tototraining/stock_models/"
echo "  - Hyperparameter configs: hyperparams/toto/"
echo "  - Comparison results: comparison_results/"
echo ""
echo "Next steps:"
echo "  1. Review training_summary.json for model performance"
echo "  2. Review comparison results to see Toto vs Kronos"
echo "  3. For stocks where Kronos wins, try:"
echo "     - Different loss functions (heteroscedastic, quantile)"
echo "     - Larger LoRA rank"
echo "     - More epochs"
echo "     - Different learning rates"
echo ""
echo "To retrain specific stocks:"
echo "  uv run python tototraining/toto_retrain_wrapper.py --stocks SPY NVDA AMD"
echo ""
echo "To compare specific stocks:"
echo "  uv run python tototraining/compare_toto_vs_kronos.py --stocks SPY NVDA"
echo ""
echo "================================================================================================"
