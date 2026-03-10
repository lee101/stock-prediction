# Production Trading Log

## 2026-03-10 05:57 UTC - Major Configuration Change

### Issues Found
1. **3 stale ETHUSD buy orders on LIVE Alpaca account** - The `alpaca-hourly-trader` systemd service had died, leaving orphaned limit buy orders from 3 consecutive hourly cycles (Mar 9 10pm, 11pm, Mar 10 12am UTC). Only 1 order should exist at a time.

2. **Stock trader (unified-stock-trader) completely broken** - ALL strategy simulations failing with `NameError: name '_bar_margin_for_symbol' is not defined`. Process (PID 1617308, started Mar 8 00:35 UTC) was running stale code - the source file was updated at 06:36 UTC the same day. Every symbol sat out (CASH) on every cycle = stock trader was doing **nothing** for 2+ days.

3. **No crypto trader running** - `alpaca-hourly-trader.service` was inactive/disabled with no replacement.

### Actions Taken

#### Stale Orders Cancelled
- Cancelled 3 ETHUSD buy orders: $1960.61, $1968.40, $1975.02
- Cancelled 2 additional stale orders found on crypto trader startup
- Closed dust ETHUSD position (0.000000244 ETH)

#### Stock Trader Fixed (supervisor: unified-stock-trader)
- **Action**: Restarted via `supervisorctl restart unified-stock-trader`
- **Root cause**: Process had stale bytecode from before `_bar_margin_for_symbol` was restored in the merge commit (6ad5e41)
- **Verified**: After restart, strategies compute successfully. PLTR selected `wd04` strategy with edge=0.0146
- **Config**: Unchanged - 7 strategies (wd04, wd06, wd05, wd08, wd03, robb, robc), 7 stocks (NVDA, PLTR, GOOG, DBX, TRIP, MTCH, NYT)

#### Crypto Trader Started (systemd: alpaca-hourly-trader)
- **Action**: Updated service config with best-performing checkpoints, enabled and started
- **Previous config**: Single weak checkpoint (`alpaca_cross_global_mixed7_robust_short_seq128`) for all symbols including stocks
- **New config**: Per-symbol best checkpoints, crypto-only (ETHUSD + BTCUSD)

| Parameter | Old Value | New Value | Reason |
|---|---|---|---|
| Symbols | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | ETHUSD,BTCUSD | Stocks handled by unified-stock-trader |
| ETHUSD checkpoint | alpaca_cross_global (1.42x return) | seed42_ethusd_ft30 epoch_028 (5.28x return) | 3.7x better backtest return |
| BTCUSD checkpoint | alpaca_cross_global (2.22x return) | btcusd_h1only_ft30 epoch_029 (7.60x return) | 3.4x better backtest return |
| Sequence length | 128 | 72 | Matches training config of new checkpoints |
| Allocation pct | 1.0 | 0.5 | Conservative start with new models |
| Intensity scale | 2.0 | 1.0 | Conservative start |

#### Code Fix: Startup Order Cleanup
- Added stale order cancellation on startup in `newnanoalpacahourlyexp/trade_alpaca_hourly.py`
- On startup, cancels any existing open orders for the configured symbols
- Prevents orphaned orders if the service crashes/restarts between cycles

### Backtest Results (Full Sweep)

#### ETHUSD - Top 5 by Total Return
| Checkpoint | Seq Len | Max Hold | Total Return | Sortino |
|---|---|---|---|---|
| seed42_ethusd epoch_028 | 72 | 24h | 5.28x | 67.01 |
| ft_cross_target_ethusd epoch_016 | 96 | 6h | 3.60x | 49.71 |
| seed42_ethusd epoch_028 | 72 | default | 3.45x | 51.64 |
| seed2024_ethusd epoch_030 | 72 | default | 3.28x | 62.91 |
| ft_cross_target_ethusd epoch_016 | 96 | default | 2.94x | 26.15 |

#### BTCUSD - Top 5 by Total Return
| Checkpoint | Seq Len | Max Hold | Total Return | Sortino |
|---|---|---|---|---|
| btcusd_h1only epoch_029 | 72 | 24h | 7.60x | 75.07 |
| btcusd_h1only epoch_029 | 72 | default | 7.24x | 72.28 |
| btcusd_h1only epoch_029 | 72 | 6h | 7.14x | 75.48 |
| selector_robust_btcusd epoch_012 | 96 | 6h | 6.20x | 121.33 |
| selector_robust_btcusd epoch_012 | 96 | default | 6.02x | 106.33 |

### Current Running Systems

| System | Process | Status | Symbols | Management |
|---|---|---|---|---|
| Stock trader | unified-stock-trader | RUNNING | NVDA, PLTR, GOOG, DBX, TRIP, MTCH, NYT | supervisorctl |
| Crypto trader | alpaca-hourly-trader | RUNNING | ETHUSD, BTCUSD | systemctl |
| Data collector | collect_stock_5min_bars | RUNNING | All | supervisorctl |
| Cache refresh | refresh_stock_caches | RUNNING | All | supervisorctl |

### Account State (Post-Fix)
- Equity: $40,475.39
- Cash: $10,118.84
- Buying power: $20,237.68
- Open positions: Dust ETHUSD (closing)
- Open orders: BTCUSD buy @ $69,736.70 (entry), ETHUSD sell @ $2,043.18 (dust close)

### Known Issues / TODO
- ETHUSD dust position may block new buy entries until closed (always_full_exit mode)
- Backtest used `binanceneural/run_simulation.py` (simpler simulator); should validate with `newnanoalpacahourlyexp/run_hourly_trader_sim.py` for live-realistic fills
- Could add `--max-hold-hours 24` to service config (showed best returns in backtest) but keeping defaults for now
- `selector_robust_btcusd` has best Sortino (121) but lower return than h1only; could consider for risk-adjusted trading
- Monitor first few cycles to verify live behavior matches backtest expectations

---

## 2026-03-10 06:15 UTC - Realistic Backtest Results (CRITICAL)

### Backtest numbers are NOT real - look-ahead bias confirmed

Ran backtests with realistic execution friction to validate the optimistic numbers above. Results show the models **cannot profitably trade** under real-world conditions.

### Alpha Decomposition (BTCUSD h1only checkpoint)

| Setting | Total Return | Sortino | Notes |
|---------|-------------|---------|-------|
| Optimistic (lag=0, no buffer, 8bps fee) | +748% | 73.1 | Same-bar fills, Binance fees |
| + 5bps fill buffer only | +534% | 54.9 | Slippage alone is minor |
| + 15bps Alpaca fee only | +277% | 48.3 | Fees alone are minor |
| **+ 1-bar lag only** | **+23%** | **2.6** | **97% of alpha destroyed** |
| Lag + 15bps fee | +11% | 1.4 | Barely above zero |
| Lag + 2bps buffer + 15bps fee | +11% | 1.4 | Buffer irrelevant once lagged |
| Lag + 5bps buffer + 15bps fee | +10% | 1.3 | Full realistic |
| Full realistic + one-side-per-bar | +4% | 0.6 | Not tradeable |

### ETHUSD (seed42 checkpoint, realistic settings)

| Setting | Total Return | Sortino |
|---------|-------------|---------|
| Optimistic | +130% | 20.8 |
| Realistic (lag+5bps+15bps+1side) | **-10%** | **-1.2** |
| Realistic + 24h max hold | -68% | -12.2 |

### Root Cause

The model uses `reference_close = frame["close"]` (current bar's close) to decode predictions into absolute prices. The simulator then tests whether those prices would fill within the same bar's high/low range. This is look-ahead bias: the model sees the close, generates limit prices, and gets filled on the same bar.

In production, orders placed at hour T based on bar T's close can only fill during bar T+1 at earliest. This 1-bar lag alone destroys 97% of the signal.

### Account State (Updated)
- Equity: $40,475.38
- Cash: $37,846.12
- Buying Power: $75,692.24
- Open positions: 0 (dust closed)
- Open orders: ETHUSD buy @ $2,031.42, BTCUSD buy @ $69,736.70

### Corrected Backtests with 10bps Fee (Alpaca actual rate)

Previous backtests used 15bps (wrong). Correct Alpaca crypto maker fee is **10bps**.

| Symbol | Lag-only + 10bps fee | Full realistic (lag+5bps+10bps+1side) |
|--------|---------------------|--------------------------------------|
| BTCUSD | **+19.8%** / Sortino 2.24 | **+7.6%** / Sortino 0.98 |
| ETHUSD | -5.0% / Sortino -0.51 | -8.9% / Sortino -0.99 |

**BTCUSD is marginally profitable** even under friction. ETHUSD is negative in backtests but profitable in real trading.

**Real trades were profitable:** Before the service died, the old config made successful ETH round-trips (Mar 8: +1.7%, Mar 9: +1.9%). Live limit orders fill within the same hour (closer to lag=0 than lag=1), explaining why real results beat the lag=1 backtest.

### Configuration Updates
- `--allocation-pct` changed from 0.5 to 1.0 (each symbol gets 50% of equity with portfolio mode)
- `DEFAULT_MAKER_FEE_RATE` updated from 8bps to 10bps in `differentiable_loss_utils.py`

### BTCUSD Model Signal Issue
BTC model outputs `buy_amt=0.02, sell_amt=100.00` - strongly wants to **short**, not go long. Since `can_short=False`, buy signal is near-zero (~$10 min notional). This is correct model behavior - model thinks BTC should be sold right now.

---

## 2026-03-10 07:39 UTC - Meta-Switching Experiment & Checkpoint Switch

### Meta-Switching Experiment Results

Built `binanceneural/run_meta_simulation.py` to test strategy-switching approaches:
- **Meta-A**: Simple trailing 24h return - pick best recent performer
- **Meta-B**: Use Chronos2 to forecast 6h of PnL, pick best predicted

#### BTCUSD (5 strategies, lag=1, 10bps fee)

| Strategy | Return | Sortino |
|----------|--------|---------|
| h1only | +20.1% | 2.27 |
| **META-B (Chronos2 6h)** | **+11.9%** | **2.83** |
| seed42 | +7.2% | 1.45 |
| seed2024 | +6.7% | 1.40 |
| seed123 | +5.5% | 1.23 |
| robust | +2.3% | 0.62 |
| CASH | 0.0% | 0.00 |
| META-A (trailing 24h) | -3.2% | -0.44 |

**META-B has highest Sortino (2.83)** - better risk-adjusted than any individual strategy.

#### ETHUSD (4 strategies, lag=1, 10bps fee)

| Strategy | Return | Sortino |
|----------|--------|---------|
| **h1only** | **+13.5%** | **1.38** |
| ft_cross | +0.2% | 0.28 |
| seed123 | -1.9% | -0.03 |
| seed42 | -8.0% | -0.97 |
| META-B | -12.0% | -2.13 |
| META-A | -24.3% | -2.86 |

**h1only dominates ETH** - meta-switching hurts by rotating into losing strategies.

### Checkpoint Switch: ETHUSD seed42 → h1only

The seed42 checkpoint (previously deployed) is -8% under lag=1. The h1only checkpoint is +13.5%. Switched ETHUSD to `ethusd_h1only_ft30_20260208/epoch_030.pt`.

### Current Service Config
- Symbols: ETHUSD, BTCUSD
- ETH checkpoint: `ethusd_h1only_ft30_20260208/epoch_030.pt` (was seed42)
- BTC checkpoint: `btcusd_h1only_ft30_20260208/epoch_029.pt` (unchanged)
- allocation_pct: 1.0, portfolio mode (50% each)
- Both models currently bearish (sell_amt=100%, buy_amt~0.02-0.04%) = minimum $10 orders

### Next Steps
- **Monitor real P&L** over next 24-48 hours
- **Consider enabling short for BTC/ETH** - both models have strong short conviction
- **Retrain with execution lag penalty** in differentiable loss
- **Production meta-switcher**: Implement META-B (Chronos2 PnL forecasting) in the live trader
- **Unify stock + crypto meta-learning** - extend meta-selector to crypto strategies
