# Experiments Directory

This folder contains experimental results, backtest data, and model comparisons for the NeuralDaily trading system.

## Directory Structure

### walk_forward/
Walk-forward validation results testing weekly retraining performance.
- **Best Result**: Standard 8-week: 62.79% total return, 62.5% positive weeks
- See `walk_forward/README.md` for detailed results

### model_sizes/
Model architecture experiments comparing different transformer configurations.

### neural_strategies/
Neural strategy comparison experiments.

### Sizing Strategy Files
- `comprehensive_sizing_results.json` - Full position sizing comparison
- `leverage_sizing_stocks_results.json` - Leverage impact on stocks
- `sizing_strategy_results.json` - Basic sizing strategy comparison
- `top5_sizing_marketsim_results.json` - Top 5 strategies in market simulator

## Current Model Performance (V10/V11)

### V10 Extended Training Results
- **Checkpoint**: `neuraldailyv4/checkpoints/v10_extended/epoch_0020.pt`
- **90-day Backtest**:
  - Total Return: 90.30%
  - Sharpe Ratio: 0.660
  - Sortino Ratio: 6.01
  - Win Rate: 75.8%
  - TP Rate: 63.2%
  - Total Trades: 95

### Walk-Forward Validation (8 weeks)
- Total Return: 62.79%
- Avg Weekly Return: 7.85%
- Positive Weeks: 5/8 (62.5%)
- **Verdict**: PASSED

## V11 Improvements (2025-12-26)

### Bug Fix
- Fixed position regularization using wrong variable (`avg_confidence` instead of `position_size`)

### New Features
1. **Utilization Penalty** - Encourages using more portfolio capacity
2. **Holiday Awareness** - 3 new features for trading day detection
3. **Crypto During Holidays** - Auto-enables crypto when NYSE closed

### Expected Impact
- Larger position sizes (regularization now works correctly)
- Better portfolio utilization
- No more 0-trade weeks during holidays

## Paper Trading Status

### Current Session (Started 2025-12-26)
- **Model**: V10 Extended
- **Positions**: 10 active
- **Filled**: AMD @ $204.95 (TP: $216.85)
- **Pending Entry**:
  - ADBE @ $337.87 (TP: $356.00)
  - QCOM @ $167.06 (TP: $176.03)
  - GOOGL @ $294.42 (TP: $311.55)
  - TSLA @ $464.35 (TP: $489.26)
  - AMAT @ $247.18 (TP: $260.45)
  - PFE @ $23.96 (TP: $25.24)
  - LRCX @ $164.90 (TP: $173.75)
  - NVDA @ $174.50 (TP: $184.67)
  - TGT @ $90.39 (TP: $95.38)

## Running Experiments

### Walk-Forward Validation
```bash
python walk_forward_validation.py --weeks 8 --epochs 30 --config standard
```

### Position Size Tuning
```bash
python tune_position_size.py --checkpoint neuraldailyv4/checkpoints/v10_extended/epoch_0020.pt
```

### Paper Trading
```bash
python paper_trade.py
```

---
*Last updated: 2025-12-27*
