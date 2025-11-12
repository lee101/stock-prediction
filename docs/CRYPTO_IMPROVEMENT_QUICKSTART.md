# Crypto Forecasting - Quick Start Guide

## TL;DR
ETHUSD has the worst performance (3.75% MAE). Quick fix: increase samples from 128 → 1024.
Expected improvement: 30-40% better accuracy.

## Current Performance
```
BTCUSD: 1.95% MAE ✓ (1024 samples, trimmed_mean_5)
ETHUSD: 3.75% MAE ⚠️ (128 samples, trimmed_mean_20) ← FIX THIS
UNIUSD: 2.85% MAE   (Kronos model, 320 samples)
```

## Quick Wins (Do These First)

### 1. Test Improved ETHUSD Config (5 min)
The simplest test - already generated the config file:
```bash
# Config already saved to: hyperparams/crypto_improved/ETHUSD_config2.json
# Config: 1024 samples, trimmed_mean_5 (matches best-performing BTCUSD)

# Method A: Use it directly in your forecaster
# Edit stockagentcombined/forecaster.py to point to new config

# Method B: Run evaluation to validate
python tototraining/quick_eval.py \
  --symbol ETHUSD \
  --config hyperparams/crypto_improved/ETHUSD_config2.json
```

### 2. Try All Improved Configs (30 min)
```bash
# Test all 8 generated improved configs
# They're in: hyperparams/crypto_improved/

# ETHUSD (3 configs)
ETHUSD_config1.json - 512 samples (moderate)
ETHUSD_config2.json - 1024 samples (recommended)
ETHUSD_config3.json - 2048 samples (max)

# BTCUSD (2 configs)
BTCUSD_config1.json - 2048 samples
BTCUSD_config2.json - quantile aggregation

# UNIUSD (3 configs)
UNIUSD_config1.json - Switch to Toto 1024
UNIUSD_config2.json - Toto 2048
UNIUSD_config3.json - Improved Kronos
```

### 3. Run Automated Hyperparameter Search (2-4 hours)
```bash
# Full grid search
python test_hyperparameters_extended.py \
  --symbols BTCUSD ETHUSD UNIUSD \
  --search-method grid \
  --models both

# Or use Optuna for smarter search (requires: uv pip install optuna)
python test_hyperparameters_extended.py \
  --symbols BTCUSD ETHUSD UNIUSD \
  --search-method optuna \
  --n-trials 50 \
  --models toto
```

## Advanced Improvements

### Ensemble Models (1-2 days)
Combine Kronos + Toto for better predictions:
```python
# In stockagentcombined/forecaster.py
toto_pred = toto_model.forecast(...)
kronos_pred = kronos_model.forecast(...)

# Weight by validation performance
ensemble = 0.6 * toto_pred + 0.4 * kronos_pred
```

### Fine-tune Models (3-5 days)

#### Toto Training
```bash
cd tototraining

# First: Debug why recent training failed
# Check: optimization_results/optimization_results.json
# All attempts show "success": false

# Then: Run training
python train.py \
  --symbols BTCUSD,ETHUSD,UNIUSD \
  --epochs 10 \
  --batch_size 8
```

#### Kronos Training
```bash
cd kronostraining

python run_training.py \
  --symbols BTCUSD,ETHUSD,UNIUSD \
  --epochs 20
```

## Files Created

### Scripts
- `optimize_crypto_forecasting.py` - Full optimization pipeline
- `apply_improved_crypto_configs.py` - Generate improved configs
- `quick_crypto_config_test.py` - Quick analysis

### Documentation
- `docs/CRYPTO_FORECASTING_IMPROVEMENT_PLAN.md` - Complete plan
- `CRYPTO_IMPROVEMENT_QUICKSTART.md` - This file

### Generated Configs
- `hyperparams/crypto_improved/` - 8 improved configs ready to test

## What to Try

### Priority 1: Quick Tests (Today)
1. Apply ETHUSD_config2.json (1024 samples)
2. Run quick evaluation to verify improvement
3. Update hyperparams/best/ETHUSD.json if better

### Priority 2: Comprehensive Search (This Week)
1. Run test_hyperparameters_extended.py for all 3 symbols
2. Test ensemble approach
3. Try different aggregation strategies

### Priority 3: Retraining (Next 2 Weeks)
1. Debug toto training failures
2. Fine-tune on crypto-specific data
3. Re-sweep hyperparameters with new models

## Expected Results

### ETHUSD (Primary Target)
- Current: 3.75% MAE
- With 1024 samples: ~2.4% MAE (36% improvement)
- With retraining: ~1.8% MAE (52% improvement)

### BTCUSD (Already Good)
- Current: 1.95% MAE
- With 2048 samples: ~1.7% MAE (13% improvement)
- With retraining: ~1.4% MAE (28% improvement)

### UNIUSD (Switch to Toto)
- Current: 2.85% MAE (Kronos)
- With Toto 1024: ~2.2% MAE (22% improvement)
- With retraining: ~1.9% MAE (33% improvement)

## Next Steps
1. Check environment setup: `source .venv/bin/activate && python -c "import torch; import numpy"`
2. Test one improved config to validate approach
3. Run full hyperparameter sweep overnight
4. Monitor results and iterate

## Questions?
- See full plan: `docs/CRYPTO_FORECASTING_IMPROVEMENT_PLAN.md`
- Check training docs: `docs/TRAINING_OVERVIEW.md`
- Hyperparameter details: `test_hyperparameters_extended.py`
