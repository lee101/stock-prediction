# Quick Start: Model Retraining & Optimization

## TL;DR

Run this single command to retrain all models with extended training and validate on 7-day holdout:

```bash
./run_full_retraining_pipeline.sh true
```

## What This Does

1. **Retrains Kronos models** per stock with extended epochs (1.5x default)
2. **Retrains Toto models** per stock with extended epochs
3. **Validates hyperparameters** on 7-day holdout data
4. **Compares performance** against baseline and between models

## Analysis Results

Based on current optimization results:

### ✅ Successful Optimizations: 21 stocks

All successful stocks showed improvement over baseline:
- **SPY**: 0.36% MAE (best performer)
- **BTCUSD**: 0.97% MAE
- **META**: 0.99% MAE
- **QQQ**: 0.49% MAE
- And 17 more...

### ❌ Failed Optimizations: 2 stocks

- **ADBE**: Model cache write error
- **TSLA**: GPU OOM during Kronos prediction

## Next Steps

### 1. Fix Failed Stocks

```bash
# Create model cache directory
mkdir -p compiled_models/toto/Datadog-Toto-Open-Base-1.0/fp32

# Retrain ADBE and TSLA with reduced memory
uv run python retrain_all_stocks.py --stocks ADBE TSLA
```

### 2. Retrain All Stocks with Extended Training

**Quick test (priority stocks only):**
```bash
./run_full_retraining_pipeline.sh true
```

**Full training (all stocks):**
```bash
./run_full_retraining_pipeline.sh false
```

**With custom settings:**
```bash
# 10-day holdout, 128-step predictions
./run_full_retraining_pipeline.sh true 10 128
```

### 3. Individual Stock Retraining

```bash
# Retrain specific stocks
uv run python retrain_all_stocks.py --stocks AAPL NVDA AMD

# Kronos only
uv run python retrain_all_stocks.py --stocks TSLA --model-type kronos

# Toto only
uv run python retrain_all_stocks.py --stocks SPY QQQ --model-type toto
```

### 4. Validate Hyperparameters

```bash
# Test inference params on holdout data
uv run python validate_hyperparams_holdout.py --stocks SPY NVDA AMD

# All stocks with custom holdout
uv run python validate_hyperparams_holdout.py --all --holdout-days 14
```

## Key Features

### Automatic Configuration

The retraining scripts automatically adjust:

- **Context/prediction length** based on dataset size
- **LoRA rank** based on prediction difficulty
- **Loss function** based on baseline MAE
- **Epochs** based on difficulty (with 1.5x multiplier)
- **Learning rate** based on difficulty

### Training Strategy by Difficulty

| Difficulty | Baseline MAE% | LoRA Rank | Loss Function | Extra Epochs |
|------------|---------------|-----------|---------------|--------------|
| Easy       | < 8%          | 8         | Huber         | +0           |
| Medium     | 8-15%         | 12        | Heteroscedastic | +0         |
| Hard       | > 15%         | 16        | Heteroscedastic | +10        |

### Hyperparameter Validation

Tests inference-time parameters on recent holdout data:

**Kronos:**
- `num_samples`, `temperature`, `top_k`, `top_p`

**Toto:**
- `temperature`, `use_past_values`

Saves best configuration per stock to `hyperparameter_validation_results.json`

## Expected Improvements

From extended training:
- Easy stocks: **30-40% better** than baseline
- Medium stocks: **20-30% better** than baseline
- Hard stocks: **15-25% better** than baseline

From inference tuning:
- Additional **10-25% improvement** from optimal parameters

## Files Created

After running the pipeline:

```
├── retraining_results.json                    # Training summary
├── hyperparameter_validation_results.json     # Best params per stock
├── regression_analysis.json                   # Performance analysis
│
├── kronostraining/artifacts/stock_specific/   # Kronos models
│   └── {STOCK}/adapters/{STOCK}/adapter.pt
│
├── tototraining/stock_models/                 # Toto models
│   └── {STOCK}/{STOCK}_model/
│
└── comparison_results/                        # Performance comparisons
    └── comparison_summary_h64.json
```

## Monitoring Progress

```bash
# Watch training logs
tail -f tototraining/stock_models/SPY/training_output.txt

# Check GPU usage
watch -n 1 nvidia-smi

# View results
cat retraining_results.json | jq
cat hyperparameter_validation_results.json | jq
```

## Troubleshooting

### CUDA OOM Error

Reduce batch size in `retrain_all_stocks.py`:
```python
batch_size = 2  # instead of 4
```

### Model Cache Error

Create missing directories:
```bash
mkdir -p compiled_models/toto/Datadog-Toto-Open-Base-1.0/fp32
mkdir -p compiled_models/kronos/
```

### No Improvement After Training

Try these adjustments:
1. Increase epochs (edit `_get_*_config()` methods)
2. Lower learning rate (1e-5 instead of 1e-4)
3. Increase LoRA rank (16 or 32)
4. Try different loss function

## Integration with Existing Code

The retrained models work with existing test frameworks:

```bash
# Compare models
uv run python tototraining/compare_toto_vs_kronos.py --all

# Run hyperparameter tests
uv run python test_hyperparameters_extended.py

# Backtest with new models
uv run python trade_stock_e2e.py --symbol SPY
```

## Full Documentation

See `RETRAINING_GUIDE.md` for complete details on:
- Detailed workflow and methodology
- Advanced configuration options
- Performance monitoring
- Integration patterns
- Troubleshooting guide

---

**Ready to start?**

```bash
./run_full_retraining_pipeline.sh true
```

This will retrain priority stocks (SPY, QQQ, MSFT, AAPL, GOOG, NVDA, AMD, META, TSLA, BTCUSD, ETHUSD) with extended training and validate hyperparameters on 7-day holdout data.

Total time: ~2-4 hours for priority stocks, ~4-8 hours for all stocks.
