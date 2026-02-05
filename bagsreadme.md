# Bags.fm Trading Strategies

This document covers running the Bags.fm trading bots for CODEX and other Solana tokens.

## Status (2026-02-05)

We no longer run Bags.fm live trading from this repo by default. All `--live` entrypoints are guarded behind `BAGSFM_ENABLE_LIVE_TRADING=1` to prevent accidental execution.

## Neural Trader (bagsneural)

Neural network-based trading using a trained model for buy/sell signals.

### Training

```bash
# Train on CODEX data
python bagsneural/run_train.py \
  --ohlc bagstraining/ohlc_data.csv \
  --mint HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS \
  --context 16 \
  --horizon 3 \
  --epochs 100
```

### Backtesting

```bash
# Full backtest
python bagsneural/run_backtest.py \
  --mint HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS \
  --checkpoint bagsneural/checkpoints/bagsneural_HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS_best.pt \
  --context 16 \
  --buy-threshold 0.46 \
  --sell-threshold 0.42

# Out-of-sample test (last 30%)
python bagsneural/run_backtest.py \
  --mint HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS \
  --checkpoint bagsneural/checkpoints/bagsneural_HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS_best.pt \
  --context 16 \
  --test-split 0.3 \
  --auto-thresholds
```

### Live Trading

```bash
# Dry run (default)
python run_neural_trader.py --dry-run

# Live trading (guarded; requires SOLANA_PRIVATE_KEY in env_real.py)
BAGSFM_ENABLE_LIVE_TRADING=1 python run_neural_trader.py --live --max-position 0.5

# With custom thresholds
BAGSFM_ENABLE_LIVE_TRADING=1 python run_neural_trader.py --live \
  --buy-threshold 0.46 \
  --sell-threshold 0.42 \
  --interval 10
```

### Model Architecture

- Input: 16 bars of OHLC data (48 features: open/high/low normalized returns)
- Hidden: [32, 16] with ReLU + dropout
- Output: signal probability + position size

### Current Performance (Jan 19-24, 2026)

| Metric | In-Sample | Out-of-Sample (30%) |
|--------|-----------|---------------------|
| Return | +235.56% | +25-28% |
| Max DD | 26.25% | 17-25% |
| Win Rate | 59% | 70-87% |
| Trades | 45 | 17-20 |

Buy & hold benchmark: +212%

---

## Direct Chronos2 Forecaster (bagsdirect)

Uses Chronos2 time-series forecasting with scipy-optimized buy/sell thresholds.

### Training (Threshold Optimization)

```bash
python -m bagsdirect.optimizer \
  --ohlc bagstraining/ohlc_data.csv \
  --mint HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS \
  --train-split 0.7 \
  --prediction-length 6
```

### Backtesting

```bash
# Test on held-out 30%
python -m bagsdirect.backtest \
  --mint HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS \
  --test-split 0.3

# Full data backtest
python -m bagsdirect.backtest \
  --mint HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS \
  --full
```

### Optimized Parameters (Jan 2026)

```
Buy threshold: 6.48% predicted return
Sell threshold: -8.22% predicted return
Upside ratio min: 2.12
Hold bars: 6
```

### Performance Comparison (Test Period: -65% crash)

| Strategy | Return | Alpha | Trades | Win Rate |
|----------|--------|-------|--------|----------|
| Neural | +25.4% | +90.8% | 17 | 87.5% |
| Chronos2 Direct | +3.2% | +68.6% | 3 | 66.7% |
| Buy & Hold | -65.4% | - | - | - |

---

## Data Collection

```bash
# Collect OHLC data continuously
python collect_bags_data.py --interval 10

# Data stored in bagstraining/ohlc_data.csv
```

---

## Environment Setup

Required in `env_real.py`:
```python
BAGS_API_KEY = "your-bags-api-key"
HELIUS_API_KEY = "your-helius-key"  # Optional, reduces rate limits
SOLANA_PRIVATE_KEY = "base58-encoded-private-key"
SOLANA_PUBLIC_KEY = "your-wallet-address"
```

## RPC Configuration

The system uses Helius RPC (if configured) with automatic fallback to public mainnet:
- Rate limiting with exponential backoff
- Automatic endpoint rotation on 429 errors
- Balance caching to reduce RPC calls
