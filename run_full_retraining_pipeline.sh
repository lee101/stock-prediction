#!/bin/bash
set -e

#######################################################################
# Full retraining and hyperparameter validation pipeline
#
# This script:
# 1. Retrains Kronos models per stock with extended epochs
# 2. Retrains Toto models per stock with extended epochs
# 3. Validates hyperparameters on 7-day holdout
# 4. Generates comparison reports
#######################################################################

PRIORITY_ONLY=${1:-true}
HOLDOUT_DAYS=${2:-7}
PREDICTION_LENGTH=${3:-64}

echo "========================================"
echo "Full Retraining Pipeline"
echo "========================================"
echo "Priority only: $PRIORITY_ONLY"
echo "Holdout days: $HOLDOUT_DAYS"
echo "Prediction length: $PREDICTION_LENGTH"
echo ""

# Step 1: Retrain Kronos models
echo "========================================"
echo "Step 1: Retraining Kronos models"
echo "========================================"

if [ "$PRIORITY_ONLY" = "true" ]; then
    uv run python retrain_all_stocks.py \
        --priority-only \
        --model-type kronos
else
    uv run python retrain_all_stocks.py \
        --all \
        --model-type kronos
fi

echo ""
echo "Kronos retraining complete!"
echo ""

# Step 2: Retrain Toto models
echo "========================================"
echo "Step 2: Retraining Toto models"
echo "========================================"

if [ "$PRIORITY_ONLY" = "true" ]; then
    uv run python retrain_all_stocks.py \
        --priority-only \
        --model-type toto
else
    uv run python retrain_all_stocks.py \
        --all \
        --model-type toto
fi

echo ""
echo "Toto retraining complete!"
echo ""

# Step 3: Validate hyperparameters on holdout
echo "========================================"
echo "Step 3: Validating hyperparameters"
echo "========================================"

if [ "$PRIORITY_ONLY" = "true" ]; then
    uv run python validate_hyperparams_holdout.py \
        --stocks SPY QQQ MSFT AAPL GOOG NVDA AMD META TSLA BTCUSD \
        --holdout-days "$HOLDOUT_DAYS" \
        --prediction-length "$PREDICTION_LENGTH" \
        --model-type both
else
    uv run python validate_hyperparams_holdout.py \
        --all \
        --holdout-days "$HOLDOUT_DAYS" \
        --prediction-length "$PREDICTION_LENGTH" \
        --model-type both
fi

echo ""
echo "Hyperparameter validation complete!"
echo ""

# Step 4: Compare retrained models
echo "========================================"
echo "Step 4: Comparing retrained models"
echo "========================================"

if [ -f "tototraining/compare_toto_vs_kronos.py" ]; then
    if [ "$PRIORITY_ONLY" = "true" ]; then
        uv run python tototraining/compare_toto_vs_kronos.py \
            --stocks SPY QQQ MSFT AAPL GOOG NVDA AMD META TSLA BTCUSD \
            --forecast-horizon "$PREDICTION_LENGTH"
    else
        uv run python tototraining/compare_toto_vs_kronos.py \
            --all \
            --forecast-horizon "$PREDICTION_LENGTH"
    fi
fi

echo ""
echo "Comparison complete!"
echo ""

# Step 5: Generate summary report
echo "========================================"
echo "Summary"
echo "========================================"

if [ -f "retraining_results.json" ]; then
    echo "Retraining results:"
    cat retraining_results.json | jq '.kronos_successes | length' | xargs -I {} echo "  Kronos successes: {}"
    cat retraining_results.json | jq '.toto_successes | length' | xargs -I {} echo "  Toto successes: {}"
fi

echo ""

if [ -f "hyperparameter_validation_results.json" ]; then
    echo "Validation results saved to: hyperparameter_validation_results.json"
fi

if [ -f "comparison_results/comparison_summary_h${PREDICTION_LENGTH}.json" ]; then
    echo "Comparison results saved to: comparison_results/comparison_summary_h${PREDICTION_LENGTH}.json"
fi

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review hyperparameter_validation_results.json for best inference params"
echo "2. Check comparison_results/ for model performance comparisons"
echo "3. Update hyperparams/kronos/ and hyperparams/toto/ with best configs"
echo "4. Run live trading tests with updated models"
echo ""
