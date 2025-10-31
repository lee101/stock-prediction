# Complete Retraining & Hyperparameter Optimization Guide

## Overview

This guide covers the complete process for:
1. Analyzing which models got worse in recent hyperparameter tuning
2. Retraining Kronos and Toto models per stock pair from checkpoints
3. Training longer/harder with optimized configurations
4. Validating hyperparameters on 7-day holdout data
5. Comparing final performance

## Quick Start

### 1. Analyze Current Performance Issues

```bash
# See which models regressed after last hyperparam tuning
uv run python analyze_hyperparam_regression.py
```

This will show:
- Stocks where performance got worse
- Stocks that failed during optimization
- Specific recommendations

### 2. Run Full Retraining Pipeline

**Priority stocks only (recommended for iteration):**
```bash
./run_full_retraining_pipeline.sh true
```

**All stocks (takes several hours):**
```bash
./run_full_retraining_pipeline.sh false
```

**Custom holdout validation:**
```bash
# 10-day holdout, 128-step predictions
./run_full_retraining_pipeline.sh true 10 128
```

### 3. Manual Retraining (Stock-Specific)

**Retrain specific stocks:**
```bash
# Both Kronos and Toto
uv run python retrain_all_stocks.py --stocks AAPL NVDA AMD

# Kronos only
uv run python retrain_all_stocks.py --stocks TSLA BTCUSD --model-type kronos

# Toto only
uv run python retrain_all_stocks.py --stocks SPY QQQ --model-type toto
```

**Validate specific stocks:**
```bash
# Validate on 7-day holdout
uv run python validate_hyperparams_holdout.py --stocks SPY NVDA AMD

# Custom holdout period
uv run python validate_hyperparams_holdout.py --stocks SPY --holdout-days 14
```

## Detailed Workflow

### Phase 1: Identify Problems

```bash
uv run python analyze_hyperparam_regression.py
```

**Output:**
- List of stocks with worse MAE after tuning
- Failed optimization attempts
- Magnitude of regression (percentage points)
- Best model type that regressed

**Example output:**
```
Found 5 stocks with worse performance:

  ADSK    - Baseline:  8.55% → Optimized: 12.94% (+4.39%) [kronos_standard]
  TSLA    - Baseline: 19.13% → Optimized: 22.50% (+3.37%) [toto]
  ...

Recommendations:
  uv run python retrain_all_stocks.py --stocks ADSK TSLA AMZN
```

### Phase 2: Extended Retraining

The retraining script (`retrain_all_stocks.py`) automatically:

1. **Loads baseline data** to determine stock difficulty
2. **Adjusts training configuration** per stock:
   - Sample count → Context/prediction length
   - Baseline MAE → Loss function & LoRA rank
   - Difficulty level → Epochs and learning rate
3. **Trains with extended epochs** (1.5x default)
4. **Saves checkpoints** and configurations

**Training Configuration Logic:**

| Sample Count | Context | Pred Length | Base Epochs |
|--------------|---------|-------------|-------------|
| < 500        | 256     | 16          | 15          |
| 500-1000     | 512     | 32          | 20          |
| 1000-1500    | 768     | 48          | 25          |
| 1500+        | 1024    | 64          | 30          |

| Difficulty (Baseline MAE%) | LoRA Rank | Loss Type       | Epochs Bonus |
|----------------------------|-----------|-----------------|--------------|
| < 8% (Easy)                | 8         | Huber           | +0           |
| 8-15% (Medium)             | 12        | Heteroscedastic | +0           |
| > 15% (Hard)               | 16        | Heteroscedastic | +10          |

**Extended epochs:** Multiply base epochs by 1.5

**Example for NVDA (1707 samples, 15.43% baseline MAE):**
- Context: 1024, Prediction: 64
- LoRA rank: 16, Loss: Heteroscedastic
- Base epochs: 30, Extended: 45
- Learning rate: 1e-4

### Phase 3: Hyperparameter Validation

The validation script (`validate_hyperparams_holdout.py`) tests **inference-time** hyperparameters on recent holdout data:

**Kronos inference params:**
- `num_samples`: [5, 10, 20, 50]
- `temperature`: [0.5, 0.7, 1.0, 1.2]
- `top_k`: [20, 50, None]
- `top_p`: [0.8, 0.9, 0.95, None]

**Toto inference params:**
- `temperature`: [0.5, 0.7, 1.0]
- `use_past_values`: [True, False]

**Process:**
1. Load training data up to holdout period
2. Use last N days as holdout test set (default: 7)
3. Generate predictions with each param combination
4. Calculate MAE on holdout data
5. Save best parameters per stock

**Output:**
```json
{
  "stock": "SPY",
  "kronos": {
    "status": "success",
    "best_params": {
      "num_samples": 20,
      "temperature": 0.7,
      "top_k": 50,
      "top_p": 0.9,
      "mae": 2.45,
      "mae_pct": 0.51
    }
  },
  "toto": {
    "status": "success",
    "best_params": {
      "temperature": 0.7,
      "mae": 2.12,
      "mae_pct": 0.44
    }
  }
}
```

### Phase 4: Comparison & Analysis

After retraining and validation:

```bash
# Compare retrained models
uv run python tototraining/compare_toto_vs_kronos.py --all
```

**Metrics tracked:**
- Price MAE (absolute error in dollars)
- Return MAE (percentage error)
- Inference latency
- Winner per stock
- Improvement over baseline

## Expected Results

### Training Improvements

From extended training with optimized configs:

| Stock Type | Baseline MAE% | After Retraining | Improvement |
|------------|---------------|------------------|-------------|
| Easy (SPY, MSFT, AAPL) | 5-6% | 3-4% | 30-40% |
| Medium (NVDA, AMD, META) | 12-15% | 8-11% | 20-30% |
| Hard (COIN, TSLA) | 20-24% | 15-19% | 15-25% |
| Extreme (UNIUSD, QUBT) | 30-70% | 25-60% | 10-20% |

### Inference Tuning Gains

From hyperparameter validation on holdout:

| Parameter Optimization | Expected Gain |
|------------------------|---------------|
| Optimal sampling params | 5-15% MAE reduction |
| Temperature tuning | 3-10% MAE reduction |
| Top-k/top-p selection | 2-8% MAE reduction |

**Combined effect:** 10-25% additional improvement from inference tuning

## Iteration Strategy

### 1. First Pass (Priority Stocks)

```bash
# Quick iteration on 11 priority stocks (~2-4 hours)
./run_full_retraining_pipeline.sh true
```

**Priority stocks:**
- SPY, QQQ (index ETFs - easiest)
- MSFT, AAPL, GOOG (mega-caps - stable)
- NVDA, AMD, META (growth - medium difficulty)
- TSLA, BTCUSD, ETHUSD (volatile - hardest)

### 2. Analyze Results

```bash
# Check what worked
cat hyperparameter_validation_results.json | jq '.[] | select(.kronos.status=="success") | {stock, mae: .kronos.best_mae}'

# Find stocks that still underperform
uv run python analyze_hyperparam_regression.py
```

### 3. Refine Poor Performers

For stocks still underperforming:

**Manual config adjustment in `retrain_all_stocks.py`:**
```python
# Example: TSLA needs even more training
if stock == "TSLA":
    epochs = 50  # Instead of 45
    adapter_r = 20  # Instead of 16
    lr = 5e-5  # Lower learning rate
```

Then retrain:
```bash
uv run python retrain_all_stocks.py --stocks TSLA
uv run python validate_hyperparams_holdout.py --stocks TSLA
```

### 4. Full Training

Once satisfied with priority stocks:
```bash
# Train all 24 stocks (~4-8 hours)
./run_full_retraining_pipeline.sh false
```

## File Outputs

After running the full pipeline:

```
├── retraining_results.json                          # Retraining summary
├── hyperparameter_validation_results.json           # Best inference params per stock
├── regression_analysis.json                         # Which models got worse
│
├── kronostraining/artifacts/stock_specific/
│   ├── AAPL/
│   │   ├── checkpoints/                            # Training checkpoints
│   │   ├── adapters/AAPL/adapter.pt               # LoRA adapter
│   │   └── metrics/evaluation.json                 # Training metrics
│   ├── NVDA/
│   └── ...
│
├── tototraining/stock_models/
│   ├── AAPL/
│   │   ├── AAPL_model/                            # Trained model
│   │   ├── training_config.json                    # Training config
│   │   └── training_metrics.json                   # Training metrics
│   ├── NVDA/
│   └── ...
│
├── hyperparams/
│   ├── kronos/                                     # Per-stock inference configs
│   │   ├── AAPL.json
│   │   └── ...
│   └── toto/
│       ├── AAPL.json
│       └── ...
│
└── comparison_results/
    ├── AAPL_comparison.txt                         # Detailed comparison
    ├── NVDA_comparison.txt
    └── comparison_summary_h64.json                 # Aggregate results
```

## Integration with Existing Systems

### Update Hyperparameter Configs

After validation, update configs:

```bash
# For each stock, copy best params from validation results
cat hyperparameter_validation_results.json | jq '.[] | {
  stock: .stock,
  kronos_params: .kronos.best_params,
  toto_params: .toto.best_params
}'

# Manually update hyperparams/kronos/{STOCK}.json
# Manually update hyperparams/toto/{STOCK}.json
```

### Use in Live Trading

```python
# Load retrained model
from src.models.toto_wrapper import TotoPipeline

# Use stock-specific model
model = TotoPipeline.from_pretrained(
    "tototraining/stock_models/SPY/SPY_model",
    device="cuda"
)

# Load validated inference params
import json
with open("hyperparams/toto/SPY.json") as f:
    config = json.load(f)

# Make prediction with optimal params
predictions = model.predict(
    context_data,
    prediction_length=64,
    temperature=config["temperature"]
)
```

## Troubleshooting

### Issue: Training fails with CUDA OOM

**Solution:** Reduce batch size and context length
```bash
# Edit retrain_all_stocks.py
# Change: batch_size = 4 → batch_size = 2
# Change: context_length = 1024 → context_length = 512
```

### Issue: Validation shows no improvement

**Possible causes:**
1. Not enough training epochs → Increase in `_get_*_config()` methods
2. Learning rate too high → Try lower LR (1e-5)
3. Overfitting → Add dropout, reduce LoRA rank
4. Data quality issues → Check training data

### Issue: Some stocks still regress

**Strategy:**
1. Check if it's truly a regression (use longer holdout)
2. Try different loss function (MSE vs Huber vs Heteroscedastic)
3. Increase LoRA rank (up to 32)
4. Train even longer (2x base epochs)

## Performance Monitoring

### During Training

```bash
# Watch Kronos training
tail -f kronostraining/artifacts/stock_specific/SPY/metrics/training.log

# Watch Toto training
tail -f tototraining/stock_models/SPY/training_output.txt

# Monitor GPU
watch -n 1 nvidia-smi
```

### After Training

```bash
# Compare training metrics across stocks
cat tototraining/stock_models/*/training_metrics.json | jq '{
  stock: .stock,
  final_val_loss: .final_val_loss,
  final_val_mae: .final_val_mae
}'

# Get overall statistics
cat retraining_results.json | jq '{
  kronos_success_rate: ((.kronos_successes | length) / (.stocks_processed | length)),
  toto_success_rate: ((.toto_successes | length) / (.stocks_processed | length))
}'
```

## Next Steps

After completing retraining:

1. **Update production configs** with validated hyperparameters
2. **Run backtests** with retrained models on longer periods
3. **Paper trade** for 1-2 weeks to verify real performance
4. **Monitor live performance** and iterate if needed
5. **Schedule periodic retraining** (monthly/quarterly)

## Summary

This pipeline provides:
- ✅ Systematic retraining from checkpoints with optimal configs
- ✅ Extended training for harder-to-predict stocks
- ✅ Inference-time hyperparameter optimization on holdout data
- ✅ Clear comparison between models and baseline
- ✅ Integration with existing test/deployment framework

**Expected outcome:** 20-40% improvement over baseline, with stock-specific models outperforming generic models by 10-25% on average.

---

*Generated: 2025-10-31*
*For stock-specific retraining and hyperparameter optimization*
