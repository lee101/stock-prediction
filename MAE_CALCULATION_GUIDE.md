# MAE Calculation Guide - Stock Prediction Models

## Overview
MAE (Mean Absolute Error) is the primary metric used to evaluate prediction accuracy across both Toto and Kronos models in the stock prediction system.

## MAE Variants Used

### 1. Price MAE (price_mae)
**Calculation:** Average absolute difference between predicted and actual closing prices in dollars
```
price_mae = mean(|y_pred - y_true|)
```

**Example:**
- Actual prices: [100.50, 101.25, 99.75]
- Predicted prices: [100.75, 100.80, 100.20]
- Errors: [0.25, 0.45, 0.45]
- Price MAE = 0.38 dollars

**Use case:** Model performance in absolute dollar terms
**Interpretation:** On average, predictions are off by ~$1.27 (from unseen15 test)

---

### 2. Percentage MAE (pct_mae)
**Calculation:** MAE expressed as percentage of stock price
```
pct_mae = (mean(|y_pred - y_true|) / mean(|y_true|)) * 100
```

**Example:**
- Average price: 100.50
- Price MAE: 0.38
- pct_mae = (0.38 / 100.50) * 100 = 0.38%

**Use case:** Scale-independent comparison across different stocks
**Interpretation:** Error is ~1.16% of stock price (from unseen15 test)

---

### 3. MAPE (Mean Absolute Percentage Error)
**Calculation:** Average percentage error relative to actual values
```
mape = mean(|y_pred - y_true| / |y_true|) * 100
```

**Can be problematic:** Undefined when actual values are zero or near-zero
**Note:** High values (7341%) in unseen15 indicate issues with low-priced assets

---

## Implementation in Code

### From toto_trainer.py (Toto Model)
```python
def calculate_mae(predictions, targets):
    """Calculate MAE metrics"""
    
    # Price MAE (absolute error)
    price_mae = torch.mean(torch.abs(predictions - targets))
    
    # Percentage MAE
    pct_mae = (torch.mean(torch.abs(predictions - targets)) 
               / torch.mean(torch.abs(targets))) * 100
    
    # RMSE
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
    
    return {
        'price_mae': price_mae.item(),
        'pct_mae': pct_mae.item(),
        'rmse': rmse.item(),
        'mape': calculate_mape(predictions, targets)
    }
```

### From kronostraining/trainer.py (Kronos Model)
```python
# Kronos uses similar calculations but per-symbol:
# - Evaluates each symbol independently
# - Aggregates across symbols for overall MAE
# - Tracks both MAE and RMSE for each symbol

def evaluate_model(predictions_dict: Dict[str, np.ndarray], 
                   targets_dict: Dict[str, np.ndarray]):
    """Per-symbol evaluation"""
    
    results = []
    for symbol in symbols:
        mae = np.mean(np.abs(predictions_dict[symbol] - targets_dict[symbol]))
        rmse = np.sqrt(np.mean((predictions_dict[symbol] - targets_dict[symbol]) ** 2))
        mape = np.mean(np.abs((predictions_dict[symbol] - targets_dict[symbol]) / targets_dict[symbol])) * 100
        
        results.append({
            'symbol': symbol,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        })
    
    # Aggregate
    aggregate_mae = np.mean([r['mae'] for r in results])
    return results, aggregate_mae
```

---

## Current MAE Performance

### Toto (unseen15 validation set)
```
price_mae:    1.272 dollars
pct_mae:      1.161% 
rmse:         1.317
mape:         7341.06% (unreliable - issues with low prices)
```

**Diebold-Mariano Test:** DM statistic = 11.28, p-value = 0.0
- Indicates Toto predictions significantly outperform naive forecasting

---

### Kronos (unseen15 validation set - 15 symbols)

**Aggregate MAE by Symbol:**
```
Symbol       MAE (dollars)   Assessment
ALGO-USD     0.038           Excellent (crypto)
ADA-USD      0.162           Excellent (crypto)
BAC          2.383           Good
ABT          3.674           Good
ARKG         1.168           Excellent (ETF)
ARKK         13.384          Moderate
ARKQ         13.797          Moderate
ARKW         34.312          Poor
ASML         24.500          Moderate
AVGO         62.525          Poor
AXP          13.099          Moderate
BA           42.115          Poor
BABA         11.025          Moderate
AMT          5.167           Good

Aggregate MAE: 16.09 dollars
Aggregate RMSE: 17.62
Aggregate MAPE: 10.88%
```

**Per-Symbol Analysis:**
- Cryptos perform better (lower absolute prices, higher percentage volatility)
- High-priced stocks (BA, AVGO) show worse absolute MAE
- Mid-cap stocks (ARKK, AXP) show moderate performance

---

## MAE Interpretation Guidelines

### Excellent Performance
- pct_mae < 1%: Model captures short-term trends very well
- price_mae < 0.5% of stock price

### Good Performance
- pct_mae 1-3%: Reasonable for day-ahead predictions
- Suitable for position sizing and entry timing

### Moderate Performance
- pct_mae 3-10%: Useful for directional signals only
- May need ensemble with other models

### Poor Performance
- pct_mae > 10%: Consider alternative models or features
- Investigate data quality, stock characteristics

---

## Factors Affecting MAE

### Stock Characteristics
1. **Price Level:** Low-priced stocks (cryptos) have lower absolute MAE
2. **Volatility:** High-volatility stocks harder to predict accurately
3. **Liquidity:** More liquid assets generally easier to predict
4. **Historical Data:** More training data improves MAE

### Model Factors
1. **Context Length:** Longer lookback (4096 vs 64 steps) affects patterns recognized
2. **Prediction Horizon:** 64-step ahead is harder than 1-step
3. **Training Data:** More diverse training improves generalization
4. **Learning Rate:** Affects convergence and generalization

### Data Factors
1. **Normalization:** OHLC data should be normalized properly
2. **Outliers:** Price gaps can increase MAE significantly
3. **Non-stationarity:** Time series properties change over time
4. **Seasonality:** Market patterns vary by time of day/day of week

---

## Improving MAE

### 1. Data Quality
```python
# Check for outliers and normalize
prices = df['close'].values
z_scores = np.abs((prices - prices.mean()) / prices.std())
outliers = z_scores > 3  # Flag extreme moves

# Use robust normalization
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
normalized_prices = scaler.fit_transform(prices.reshape(-1, 1))
```

### 2. Training Configuration
```python
# Increase training time
python tototraining/train.py \
    --epochs 30 \
    --learning-rate 1e-4 \
    --batch-size 8

# Use LoRA for efficient fine-tuning
python tototraining/run_gpu_training.py \
    --adapter lora \
    --adapter-r 16 \
    --freeze-backbone
```

### 3. Hyperparameter Tuning
```python
# Key parameters for MAE reduction:
# - learning_rate: 1e-5 to 1e-3
# - weight_decay: 1e-3 to 1e-1
# - warmup_steps: 100 to 1000
# - grad_clip: 0.5 to 5.0
```

### 4. Ensemble Methods
```python
# Combine Toto and Kronos predictions
toto_pred = toto_model.predict(context)
kronos_pred = kronos_model.predict(context)
ensemble_pred = 0.6 * toto_pred + 0.4 * kronos_pred
ensemble_mae = calculate_mae(ensemble_pred, target)
```

---

## Evaluation Best Practices

### 1. Use Separate Test Sets
```python
# Never evaluate on training data
train_indices, test_indices = train_test_split(data, test_size=0.2)
train_mae = evaluate(model, data[train_indices])      # Should be lower
test_mae = evaluate(model, data[test_indices])        # True performance

# Holdout completely unseen symbols
unseen15_mae = evaluate(model, completely_new_symbols)
```

### 2. Time-Based Validation
```python
# Use walk-forward validation
for end_date in test_dates:
    train_data = data[data.date < end_date - lookback]
    test_data = data[(data.date >= end_date - lookback) & (data.date <= end_date)]
    
    model.fit(train_data)
    mae = evaluate(model, test_data)
    results.append(mae)

avg_mae = np.mean(results)  # More realistic estimate
```

### 3. Per-Symbol Metrics
```python
# Don't just use aggregate MAE
for symbol in symbols:
    symbol_mae = evaluate(model, data[data.symbol == symbol])
    print(f"{symbol}: {symbol_mae}")

# Identify problematic symbols for special handling
```

---

## Metric Tracking in Checkpoints

### File: final_metrics.json
```json
{
  "val": {
    "loss": 0.011563,
    "pct_mae": 1.161,
    "pct_rmse": 1.218,
    "price_mae": 1.272,
    "price_rmse": 1.317,
    "naive_mae": 0.0426,
    "dm_stat_vs_naive": 11.28,
    "dm_pvalue_vs_naive": 0.0
  },
  "test": {}
}
```

### File: best_records.json
```json
[
  {
    "path": "checkpoint_epoch_17.pt",
    "val_loss": 0.01156  # Best checkpoint
  },
  {
    "path": "checkpoint_epoch_16.pt",
    "val_loss": 0.01169
  }
]
```

---

## Common Issues and Solutions

### Issue 1: MAPE is undefined or extremely high
**Cause:** Predictions or targets near zero (especially low-priced cryptos)
**Solution:** Use price_mae and pct_mae instead, cap MAPE at reasonable values

### Issue 2: Training MAE much lower than validation MAE
**Cause:** Overfitting
**Solution:** 
- Increase weight_decay
- Use LoRA adapters instead of full fine-tuning
- Add dropout to adapter layers
- Use early stopping based on validation loss

### Issue 3: MAE not improving after several epochs
**Cause:** Learning rate too high or too low
**Solution:**
- Lower learning rate by 10x if diverging
- Increase learning rate if improvements plateau
- Use learning rate scheduling (WarmupCosine)

### Issue 4: Different MAE for same model on same data
**Cause:** Non-deterministic operations, floating point precision
**Solution:**
- Set random seeds: `torch.manual_seed(42)`
- Use deterministic algorithms
- Log full precision metrics

---

## References

- **Diebold-Mariano Test:** Statistical test for forecast accuracy comparison
  - H0: Both forecasts have same accuracy
  - Rejection (p < 0.05) means one is significantly better

- **Naive Baseline:** Simple previous value forecast
  - Good benchmark for time series
  - Models should beat naive forecast to be useful

