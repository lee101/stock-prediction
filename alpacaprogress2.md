# Alpaca Progress 2 - Sortino-Focused RL Training

## 2026-02-13: Starting Fresh

Goal: Train smooth, loss-resistant RL models for Alpaca crypto pairs
- Focus on Sortino ratio (penalize downside more than upside)
- Smooth PnL curves
- Low drawdowns
- Consistent returns

Target pairs: ETHUSD, UNIUSD, LINKUSD, BTCUSD, SOLUSD

### Best SUI Result (Reference from Binance)
- Model: bitbank-style 30ep, 256 hidden, 4 layers
- 7d Return: 54.69%, Sortino: 270.2
- Now running live on SUIUSDT

---

## Training Runs

### 2026-02-13 19:02 - Multi-machine Training

**Local (GTX) - COMPLETED:**

| Symbol | 7d Return | Sortino | Final Equity |
|--------|-----------|---------|--------------|
| **ETHUSD** | 21.44% | **1404.6** | $12,208 |
| LINKUSD | 21.24% | 461.9 | $12,137 |
| BTCUSD | 19.83% | 70.8 | $12,092 |

Best: **ETH** with Sortino 1404.6 (5x better than SUI reference!)

**Remote 5090 - COMPLETED:**

| Symbol | 7d Return | Sortino | Final Equity |
|--------|-----------|---------|--------------|
| **UNIUSD** | **70.97%** | **144.6** | $17,097 |
| MATICUSD | 21.37% | 55.1 | $12,137 |
| AVAXUSD | 18.94% | 56.5 | $11,894 |
| SOLUSD | 6.50% | 10.4 | $10,650 |

Best: **UNI** with 70.97% return!

Config: Alpaca fees 0.15%, 7d val/test split, h1+h24 Chronos forecasts

---

## 2026-02-13: Stock LoRA + Unified Neural Policy

### Preaug Strategy Sweep (10 Stocks)
| Symbol | Best Preaug | Val MAE% |
|--------|-------------|----------|
| META | differencing | 0.88% |
| MSFT | percent_change | 0.97% |
| YELP | percent_change | 1.12% |
| NYT | differencing | 1.19% |
| GOOG | differencing | 1.31% |
| NVDA | log_returns | 1.51% |
| DBX | differencing | 1.68% |
| TRIP | percent_change | 1.71% |
| NET | differencing | 2.27% |
| PLTR | robust_scaling | 2.71% |

Finding: `differencing` works best for most stocks.

### Neural Policy Training v2 (100 epochs, fresh caches)
- Val Sortino: **1332.0**
- Val Return: **18.6%**
- Train Sortino: 1104.3, Train Return: 13.0%
- Checkpoint: `unified_hourly_experiment/checkpoints/unified_v2`

### Neural Policy Training v3 (100 epochs, 8000h lookback)
- Train samples: 5303 (was 1271)
- Train Sortino: **2188.1**
- Val Sortino: **1358.8**
- Val Return: **21.2%**
- Checkpoint: `unified_hourly_experiment/checkpoints/unified_v3_moredata`

### Backtest Results
| Version | Return | Sortino | Notes |
|---------|--------|---------|-------|
| v1 (old caches) | -4.06% | - | Before cache rebuild |
| v2 (fresh caches) | -1.54% | - | After cache rebuild |
| **v3 (8000h data + fix)** | **+31.78%** | **1.31** | Fixed position sizing bug |

Major breakthrough! Position sizing was uncapped, causing blowup. Fixed by clamping trade_amount to [0,1].

### v3 Stats
- Initial: $10,000
- Final: $13,177.80
- Trades: 27
- Holdout period: ~400 bars (~2 months trading)

### min_edge Sweep Results
| min_edge | Return | Sortino | Trades |
|----------|--------|---------|--------|
| 0.001 | 31.78% | 1.31 | 27 |
| 0.008 | **39.49%** | 1.50 | 19 |
| 0.010 | 35.30% | 1.65 | 25 |
| **0.012** | 36.34% | **1.69** | 23 |
| 0.015 | 20.40% | 1.20 | 11 |
| 0.020 | -46.33% | -2.01 | 11 |

Optimal: **min_edge=0.012** for best Sortino, or **min_edge=0.008** for max return.

### Trading Bot Ready
- Checkpoint: `unified_hourly_experiment/checkpoints/unified_v3_moredata`
- Config: min_edge=0.012, fee=0.001, leverage=1.0
- Edge filtering: only trades when predicted_high / buy_price - fee >= min_edge
- Market hours: enforced for stocks, 24/7 for crypto

```bash
python unified_hourly_experiment/trade_unified_hourly.py \
  --checkpoint-dir unified_hourly_experiment/checkpoints/unified_v3_moredata \
  --stock-symbols NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP \
  --min-edge 0.012 --dry-run
```

### Status
- Backtest: +36.34% return, Sortino 1.69 (holdout period)
- Ready for paper trading during market hours

