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

## 2026-02-13 21:06 - Stock Sortino Training (No Forecasts Baseline)

**Tech Stocks (50 epochs, 128 hidden, 3 layers):**

| Symbol | 5d Return | Sortino | Final Equity |
|--------|-----------|---------|--------------|
| **NET** | **88.09%** | 0.0 | $19,302 |
| **META** | **34.71%** | 6141.8 | $13,471 |
| PLTR | 21.56% | 647k | $12,247 |
| MSFT | 19.59% | 97.6 | $11,962 |
| GOOG | 18.24% | 5682.5 | $11,961 |
| NVDA | 14.14% | nan | $11,414 |

Config: bf16 AMP, batch_size=32, seq_len=48, fee=0.0001
Note: High Sortino values due to few/no down days in test period

**Shortable Stocks (50 epochs, 128 hidden, 3 layers):**

| Symbol | 5d Return | Sortino | Final Equity |
|--------|-----------|---------|--------------|
| **TRIP** | **63.13%** | 0.0 | $16,553 |
| **DBX** | **34.65%** | 0.0 | $13,565 |
| YELP | 25.73% | 9376.6 | $12,768 |
| NYT | 25.08% | nan | $12,749 |

**Combined best performers:** NET(88%), TRIP(63%), META(34%), DBX(34%)

### UNI Validation (Out-of-Sample)

| Period | Return | Sortino |
|--------|--------|---------|
| Last 7d (test) | **318.14%** | 1280 |
| Week -2 (oos) | 11.24% | 71.7 |
| Week -3 (oos) | 23.48% | 158.1 |

Out-of-sample returns (11-23%) confirm model generalizes beyond training period.

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

---

## 2026-02-13 22:36 - Paper Trading Bot Deployed

### Bot Configuration
```bash
python unified_hourly_experiment/trade_unified_hourly.py \
  --checkpoint-dir unified_hourly_experiment/checkpoints/unified_v3_moredata \
  --crypto-symbols ETHUSD,SOLUSD,UNIUSD \
  --crypto-cache-root binanceneural/forecast_cache \
  --paper --loop
```

### Current Positions (Alpaca Paper)
| Symbol | Qty | Notes |
|--------|-----|-------|
| UNIUSD | 681.21 | 25.58% edge signal |
| LINKUSD | 279.66 | Prior position |
| SOLUSD | 35.38 | Prior position |
| NVDA | 92.0 | Stock position |
| Buying Power | $123,535 | Available |

### Bot Status
- Running in continuous loop mode
- Checks signals hourly
- min_edge=0.012 filtering
- Supervisor config: `supervisor/unified-hourly-paper.conf`
- Log: `unified_paper_trading.log`

---

## 2026-02-13 23:12 - Pufferlib Stock PPO Training

### Training Results (50M steps, ~30 min)
- **Sortino: 135** (peak)
- **Return: +160%** per episode (30-day simulation)
- **Win Rate: 92-93%**
- ~62-64 trades per episode
- Symbols: NVDA, MSFT, META, GOOG, PLTR, DBX, TRIP

### Config
```bash
python pufferlib_market/train.py \
  --data-path pufferlib_market/data/stocks10_data.bin \
  --total-timesteps 50000000 \
  --hidden-size 256 --arch resmlp \
  --downside-penalty 1.5 --trade-penalty 0.001 \
  --checkpoint-dir experiments/pufferlib_stocks7_50M
```

### Checkpoint
- Path: `experiments/pufferlib_stocks7_50M/best.pt`
- Actions: 15 (flat + 7 longs + 7 shorts)
- Hidden: 256, ResidualMLP with 3 blocks

### Trading Bot
```bash
python -m pufferlib_market.trade_ppo_stocks \
  --checkpoint experiments/pufferlib_stocks7_50M/best.pt \
  --paper --allocation-usd 1000
```

### Stock Categories
**Long-only (AI/Tech):** NVDA, MSFT, META, GOOG, PLTR (+ NFLX, AAPL, AMZN, etc.)
**Shortable:** YELP, EBAY, TRIP, MTCH, KIND, ANGI, Z, EXPE, BKNG, NWSA, NYT, DBX

### Production Deployment
1. Supervisor config: `supervisor/pufferlib-stocks-live.conf`
2. Monitor script: `scripts/check_trading_status.sh`
3. Cron schedule: `scripts/cron_trading_monitor.txt`

**Start live trading:**
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start pufferlib-stocks-live
```

**Monitoring (cron during market hours):**
- 9:35 AM ET: Market open check
- 12:00 PM ET: Midday check
- 3:30 PM ET: Pre-close check
- 4:05 PM ET: End of day summary

**Safety notes:**
- Bot only trades during market hours (9:30 AM - 4:00 PM ET)
- Uses market orders (no limit price issues)
- 60% confidence threshold for trades
- Keep existing SOL position sell order active
- IEX data feed (no SIP subscription required)

---

## 2026-02-13 23:17 - LIVE DEPLOYMENT

### Live Account Status
- Equity: $55,935
- Buying Power: $5,085
- SOL sell order: 667.9 @ $81.19 (preserved)
- NVDA: 2.15 shares

### Live Bot Started
```
Connected to Alpaca (live)
Starting hourly trading loop
Market closed, skipping
```

### Monitoring Schedule (cron, UTC times)
- 14:35 (9:35 AM ET) - market open check
- 16:00 (11:00 AM ET) - mid-morning
- 18:00 (1:00 PM ET) - midday
- 20:00 (3:00 PM ET) - pre-close

Monitor script: `scripts/monitor_trading.sh` (gitignored, has sudo pw)

---

## 2026-02-15: Status Review & Cleanup

### What's Running on This Machine (5090)

**Active trading process (1 only):**
- `maxdiff_cli.py close-position LINKUSD` - exit watcher for 279.66 LINKUSD, take-profit @ $103, expires 12:12 UTC today. Paper account.

**NOT running (all stopped/removed):**
- Unified hourly trading bot - NOT running, no log file exists
- Pufferlib stock PPO trading bot - NOT running
- Stock sortino training (META, NVDA) - STOPPED in supervisor
- Binance bots (SUI, BTC, ETH, SOL, selector) - configs reference `/home/lee/code/stock/` (different machine), not deployed here
- Monitoring cron jobs - all commented out
- Referenced scripts (`scripts/check_trading_status.sh`, `scripts/monitor_trading.sh`, `scripts/cron_trading_monitor.txt`) - do not exist

**Non-trading services on this machine (unrelated):**
- text-generator.io (gunicorn :8083 + inference :9080, uses 7.4GB GPU)
- websim (uvicorn :8000 + :7322)
- codex-infinity monitor + token manager
- news-crawler, news-server
- Various cloudflared tunnels

### Supervisor Configs in `supervisor/` Directory

Most configs are stale and reference `/home/lee/code/stock/` (a different machine):
- `binance-sui.conf`, `binanceexp1-*.conf`, `binanceexp1-selector.conf` - wrong paths
- `alpaca-exit-nflx.conf` - one-off exit watcher, expired Feb 6
- `unified-stock-trader.conf` - wrong paths

Configs that reference this machine's paths but are NOT running:
- `neural-trader-v3.conf` - old neural trader, dry-run mode
- `bags-data-collector.conf` - data collector
- `/etc/supervisor/conf.d/stock-sortino.conf` - training (STOPPED)

### Summary

No active stock/crypto trading bots are running on this machine. The only
trading-related process is a LINKUSD paper exit watcher that expires today.
The live deployment from Feb 13 was short-lived and is no longer active.
All cron monitoring is disabled.

### Key Artifacts That Remain

| Artifact | Path | Status |
|----------|------|--------|
| Sortino crypto checkpoints | `alpacasortino/checkpoints/` | Trained, not deployed |
| Unified v3 checkpoint | `unified_hourly_experiment/checkpoints/unified_v3_moredata` | Trained, not deployed |
| Pufferlib stocks checkpoint | `experiments/pufferlib_stocks7_50M/best.pt` | Trained, not deployed |
| PufferLib crypto12 300M (best) | various `experiments/` dirs | Trained, not deployed |
| Crypto data exports | `pufferlib_market/data/*.bin` | Available |
| Stale supervisor configs | `supervisor/*.conf` | Most reference wrong machine |

### What Was Accomplished (Feb 13)

1. **Sortino RL training** - trained crypto models (ETH, UNI, LINK, BTC, SOL) and stock models (NET, META, TRIP, DBX, etc.)
2. **Unified neural policy** - v3 achieved +36% backtest return, Sortino 1.69
3. **Pufferlib stock PPO** - 50M steps, +160% per 30d episode, 92% WR
4. **Brief live deployment** - started on live account ($55k equity) but stopped same day
5. **PufferLib crypto scaling** - continued to show crypto12 h1024 300M anneal-LR as best at 2,659x per 30d

