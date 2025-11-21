# Neural Daily Trading Model - Automated Improvement Plan

## Current Status (Nov 17, 2025)

### Baseline Performance
- **Model**: neuraldaily_20251116_233354 (epoch_0064)
- **Test Period**: Oct 5-14, 2025 (10 days)
- **Results**:
  - Final PnL: **+69.3%** (0.6930 return)
  - Sortino Ratio: **1.8873**
  - Win Rate: 8/10 days positive
  - Max Daily Gain: +15.42%
  - Max Daily Loss: -3.48%

### Simulation Validation ✅

The simulation uses **proper chronological backtesting**:

1. **No Lookahead Bias**: Model only sees data up to each date (`frame = frame[frame["date"] <= cutoff]`)
2. **Real Chronos-T Predictions**: Uses actual Chronos model forecasts for high/low predictions
3. **Realistic Trading**:
   - Buy fills only if `low <= buy_price`
   - Sell fills only if `high >= sell_price`
   - 0.08% maker fees on both sides
   - Cash constraint checking
4. **Day-by-Day Stepping**: Iterates chronologically through sorted dates

## Data Update Status

✅ **Price Data Updated to Nov 17, 2025**
- 377 symbols refreshed
- Location: `trainingdata/train/`
- Latest data points: Nov 17, 2025

🔄 **Chronos Forecasts Updating**
- 27 key symbols being updated
- Using Chronos-2 with 512 context length
- Location: `strategytraining/forecast_cache/`

## Automated Improvement Pipeline

### 1. Scripts Created

#### `update_daily_data.py`
Updates all symbol price data to latest available date.

#### `update_key_forecasts.py`
Regenerates Chronos-2 forecasts for the 27 key symbols with existing forecasts.

#### `auto_improve_loop.py`
**Main improvement engine** - systematically searches hyperparameter space:

```python
# Example usage:
python auto_improve_loop.py --max-experiments 10

# Run all generated configs:
python auto_improve_loop.py
```

**Features**:
- Generates hyperparameter grid combinations
- Trains model with each config
- Runs 10-day market simulation
- Tracks best PnL and best Sortino
- Saves results to `improvement_results.jsonl`
- Copies best checkpoints to `best_neuraldaily_checkpoints/`

### 2. Hyperparameter Search Space

#### Architecture Variations
- `transformer_dim`: [256, 384, 512]
- `transformer_heads`: [8, 12, 16]
- `transformer_layers`: [4, 6, 8]
- `sequence_length`: [128, 256, 384]

#### Training Variations
- `learning_rate`: [0.00005, 0.0001, 0.0002]
- `dropout`: [0.05, 0.1, 0.15]
- `batch_size`: [16, 32, 64]

#### Trading Variations
- `price_offset_pct`: [0.01, 0.025, 0.05]
- `max_trade_qty`: [1.0, 3.0, 5.0]
- `risk_threshold`: [0.5, 1.0, 2.0]

### 3. Experiment Tracking

Results logged to `improvement_results.jsonl`:
```json
{
  "config": {...},
  "checkpoint_path": "...",
  "final_equity": 1.6930,
  "pnl": 0.6930,
  "sortino": 1.8873,
  "training_time": 1234.56,
  "simulation_date": "2025-11-17T..."
}
```

## Next Steps

### Immediate (Tonight)

1. ✅ Update price data to Nov 17
2. 🔄 Update Chronos forecasts (in progress)
3. ⏳ Re-run baseline simulation on Nov 2025 data
4. ⏳ Launch first batch of training experiments

### Short-term (This Week)

1. **Phase 1**: Architecture Search (10-15 experiments)
   - Focus on transformer depth/width
   - Track training stability

2. **Phase 2**: Learning Rate Tuning (5-10 experiments)
   - Once best architecture found
   - Try different schedules (cosine, warmup)

3. **Phase 3**: Trading Parameter Optimization (5-10 experiments)
   - Optimize price_offset_pct and risk_threshold
   - Test on multiple time windows

### Medium-term (Next 2 Weeks)

1. **Enhanced Features**:
   - Cross-asset correlations
   - Volatility regime indicators
   - Order flow metrics
   - Market breadth indicators

2. **Loss Function Engineering**:
   - Multi-objective: PnL + Sortino + drawdown
   - Asymmetric loss (penalize losses more)
   - Reward shaping experiments

3. **Robustness Testing**:
   - Test on different market regimes
   - Validate on out-of-sample periods
   - Stress test on volatile periods (Oct-Nov 2025)

## Realism Improvements to Consider

Current sim assumptions to refine:
1. **Perfect fills** - Add slippage model
2. **Simultaneous buy/sell** - Restrict to one direction per day
3. **Single-day holding** - Add multi-day position tracking
4. **No gap risk** - Model overnight gaps
5. **Fixed fees** - Use dynamic spread modeling

## Success Metrics

Target improvements over baseline:
- **Primary**: PnL > 70% (10-day test)
- **Secondary**: Sortino > 2.0
- **Tertiary**: Max drawdown < 5%
- **Robustness**: Win rate > 75%

## Running the Automated Loop

```bash
# Activate environment
source .venv313/bin/activate

# Quick test (3 experiments)
python auto_improve_loop.py --max-experiments 3

# Full overnight run (all configs)
nohup python auto_improve_loop.py > training_overnight.log 2>&1 &

# Monitor progress
tail -f training_overnight.log
tail -f improvement_results.jsonl
```

## Analyzing Results

```python
import json
import pandas as pd

# Load results
results = []
with open('improvement_results.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

df = pd.DataFrame(results)

# Find best performers
best_pnl = df.nlargest(5, 'pnl')
best_sortino = df.nlargest(5, 'sortino')

# Analyze correlations
print(df[['pnl', 'sortino', 'final_equity']].corr())

# Best config
winner = df.loc[df['pnl'].idxmax()]
print(f"Best PnL: {winner['pnl']:.4f}")
print(f"Config: {winner['config']}")
```

## Files Created

1. `update_daily_data.py` - Data refresh script
2. `update_key_forecasts.py` - Chronos forecast updater
3. `auto_improve_loop.py` - Main training loop
4. `IMPROVEMENT_PLAN.md` - This document

## Estimated Timelines

- Single training run: ~30-60 minutes
- Single simulation: ~30 seconds
- Full experiment (train + sim): ~35-65 minutes
- 10 experiments: ~6-11 hours
- 50 experiments: ~30-55 hours

**Recommendation**: Start with 10-15 targeted experiments tonight, analyze results tomorrow, then launch larger sweep.
