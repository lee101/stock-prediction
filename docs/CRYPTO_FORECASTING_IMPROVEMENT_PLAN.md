# Crypto Forecasting Improvement Plan

## Current Performance Baseline

### BTCUSD (Best Performer)
- **Model**: Toto
- **Test pct_return_mae**: 1.95% ✓
- **Config**: 1024 samples, trimmed_mean_5, samples_per_batch=128
- **Latency**: 6.6s
- **Status**: Good baseline, room for small improvements

### ETHUSD (Needs Most Improvement)
- **Model**: Toto
- **Test pct_return_mae**: 3.75% ⚠️
- **Config**: 128 samples, trimmed_mean_20, samples_per_batch=32
- **Latency**: 3.2s
- **Status**: **PRIMARY TARGET FOR IMPROVEMENT**
- **Issue**: Too few samples (8x less than BTCUSD), overly aggressive trimming

### UNIUSD (Moderate)
- **Model**: Kronos
- **Test pct_return_mae**: 2.85%
- **Config**: 320 samples, temp=0.3, top_p=0.78, top_k=28
- **Latency**: 7.2s
- **Status**: Consider switching to Toto model

## Improvement Strategy

### Phase 1: Quick Wins (Hyperparameter Tuning)

#### 1.1 ETHUSD - Increase Samples (Expected: 30-40% improvement)
```json
Priority 1: Match BTCUSD config
{
  "num_samples": 1024,
  "aggregate": "trimmed_mean_5",
  "samples_per_batch": 128
}
Expected MAE: ~2.4% (36% improvement)
```

#### 1.2 BTCUSD - Push Lower (Expected: 5-15% improvement)
```json
Option A: More samples
{
  "num_samples": 2048,
  "aggregate": "trimmed_mean_5",
  "samples_per_batch": 128
}
Expected MAE: ~1.7%
```

#### 1.3 UNIUSD - Switch to Toto (Expected: 15-25% improvement)
```json
{
  "model": "toto",
  "num_samples": 1024,
  "aggregate": "trimmed_mean_5",
  "samples_per_batch": 128
}
Expected MAE: ~2.2%
```

### Phase 2: Advanced Techniques (10-20% additional improvement)

#### 2.1 Ensemble Models
Combine Kronos + Toto predictions:
```python
ensemble_pred = 0.6 * toto_pred + 0.4 * kronos_pred
```
Weight by validation performance.

#### 2.2 Aggregation Optimization
Test alternatives to trimmed_mean:
- `quantile_0.50` (median) - robust to outliers
- `trimmed_mean_3` - less aggressive trimming
- `winsorized_mean` - cap outliers instead of removing

#### 2.3 Context Length Tuning
Currently using 128-288 context window:
- Try 384 or 512 for crypto (high autocorrelation)
- Balance with memory constraints

### Phase 3: Model Retraining (20-30% additional improvement)

#### 3.1 Toto Fine-tuning
```bash
# Location: tototraining/
cd tototraining

# Edit train.py to focus on crypto symbols
python train.py \
  --symbols BTCUSD,ETHUSD,UNIUSD,SOLUSD \
  --epochs 10 \
  --learning_rate 0.0001 \
  --batch_size 8 \
  --context_length 4096 \
  --prediction_length 64
```

**Training Configuration** (tototraining/optimization_results/):
- Recent attempts failed - need to debug first
- Try different loss functions: huber, heteroscedastic
- Adjust learning rate: 0.0001 to 0.0005
- Monitor for overfitting on small crypto dataset

#### 3.2 Kronos Fine-tuning
```bash
# Location: kronostraining/
cd kronostraining

python run_training.py \
  --data_dir ../trainingdata \
  --symbols BTCUSD,ETHUSD,UNIUSD \
  --epochs 20 \
  --learning_rate 1e-4 \
  --batch_size 16
```

### Phase 4: Full Re-sweep
After retraining, re-run hyperparameter optimization:
```bash
python test_hyperparameters_extended.py \
  --symbols BTCUSD ETHUSD UNIUSD \
  --search-method optuna \
  --models both \
  --n-trials 100 \
  --output-dir hyperparams_optimized_crypto
```

## Implementation Scripts

### Generated Files
1. **optimize_crypto_forecasting.py** - Full optimization pipeline
2. **apply_improved_crypto_configs.py** - Generate improved configs
3. **quick_crypto_config_test.py** - Quick analysis script

### Improved Configs Generated
- `hyperparams/crypto_improved/ETHUSD_config1.json` - 512 samples
- `hyperparams/crypto_improved/ETHUSD_config2.json` - 1024 samples (recommended)
- `hyperparams/crypto_improved/ETHUSD_config3.json` - 2048 samples
- `hyperparams/crypto_improved/BTCUSD_config1.json` - 2048 samples
- `hyperparams/crypto_improved/BTCUSD_config2.json` - quantile aggregation
- `hyperparams/crypto_improved/UNIUSD_config1.json` - Toto 1024
- `hyperparams/crypto_improved/UNIUSD_config2.json` - Toto 2048
- `hyperparams/crypto_improved/UNIUSD_config3.json` - Improved Kronos

## Execution Plan

### Immediate Actions (Today)
```bash
# 1. Test improved ETHUSD config
python test_hyperparameters_extended.py \
  --symbols ETHUSD \
  --config-file hyperparams/crypto_improved/ETHUSD_config2.json

# 2. Quick eval all improved configs
for symbol in BTCUSD ETHUSD UNIUSD; do
  python tototraining/quick_eval.py --symbol $symbol
done
```

### Short-term (This Week)
1. Run full hyperparameter sweep for all 3 crypto assets
2. Test ensemble approach
3. Debug toto training failures
4. Begin crypto-specific fine-tuning

### Medium-term (Next 2 Weeks)
1. Complete Toto model retraining on crypto data
2. Complete Kronos model retraining
3. Re-sweep hyperparameters with new models
4. Update production configs in `hyperparams/best/`

### Long-term (Ongoing)
1. Monitor live trading performance
2. Iterate on training data quality
3. Experiment with additional crypto assets
4. Continuous hyperparameter optimization

## Expected Outcomes

### Conservative Estimates
- ETHUSD: 3.75% → 2.5% (33% improvement)
- BTCUSD: 1.95% → 1.7% (13% improvement)
- UNIUSD: 2.85% → 2.3% (19% improvement)

### Optimistic Estimates (with retraining)
- ETHUSD: 3.75% → 1.8% (52% improvement)
- BTCUSD: 1.95% → 1.4% (28% improvement)
- UNIUSD: 2.85% → 1.9% (33% improvement)

## Risk Factors
1. **Overfitting**: Small crypto datasets may cause overfitting during retraining
2. **Latency**: Higher sample counts increase inference time
3. **Market Changes**: Crypto volatility may invalidate historical patterns
4. **GPU Memory**: Large models may hit memory constraints

## Monitoring
Track these metrics after each change:
- `validation/pct_return_mae` - Generalization performance
- `test/pct_return_mae` - Final evaluation metric
- `latency_s` - Inference speed for trading
- Live trading PnL - Ultimate success metric

## Resources
- Hyperparameter optimization: `test_hyperparameters_extended.py`
- Toto training: `tototraining/train.py`, `tototraining/optimize_training.py`
- Kronos training: `kronostraining/run_training.py`
- Quick evaluation: `tototraining/quick_eval.py`
- Comparison tools: `tototraining/compare_toto_vs_kronos.py`
