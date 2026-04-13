# XGBoost Hourly Open-to-Close Strategy

## Overview

An XGBoost classifier trained to predict hourly open-to-close direction across US stocks.
Strategy: buy at the open of each hourly bar, sell at the close of the same bar, using realistic spread + commission costs.

## Data

### MKTD v2 Binary Format

The hourly pipeline reads from `pufferlib_market/data/*_hourly_*_v2_*.bin` files.

**CRITICAL LIMITATION**: The available MKTD files (e.g. `stocks13_hourly_forecast_mktd_v2_20260214.bin`) are **RL training observation tensors**, not raw OHLCV bars.

- Shape: `[num_steps, num_symbols, n_features]` of `float32`
- Timestamps (feature index 0) are **RL-normalized values** (garbage floats like 0.3, 1.7, etc.), NOT unix timestamps
- Parsing these as `int64` seconds causes year 2398+ overflow

The `mktd_reader.py` guards against this:
```python
valid_ts = np.isfinite(ts_float) & (ts_float > 1.26e9) & (ts_float < 2.1e9)
```
RL tensor files have no valid timestamps → all rows dropped → empty DataFrames → `run_hourly.py` exits with an error.

### Genuine Hourly OHLCV

Genuine hourly OHLCV data in MKTD v2 format would have:
- Unix timestamps in range ~1.26e9 (year 2010) to ~2.1e9 (year 2036)
- OHLCV values in real dollar/share units (open ~$50–500 for typical stocks)

Until such a file is available, the hourly strategy cannot run end-to-end.

## Features (Hourly)

All features computed with strict no-lookahead: `feature[H]` uses only data available at end of bar H-1.

| Feature | Description |
|---------|-------------|
| `ret_1h` | 1-bar return using `close.shift(1)` |
| `ret_4h` | 4-bar return using `close.shift(1)` |
| `ret_8h` | 8-bar return using `close.shift(1)` |
| `rsi_14` | Wilder's RSI(14) on prior-bar close series |
| `vol_4h`, `vol_8h` | Annualized realized volatility (scaled to 252×6.5 bars/year) |
| `atr_4h` | Average True Range over 4 bars, fraction of price |
| `spread_bps` | Volume-based spread tiers, scaled to hourly bar dollar volume |
| `dolvol_4h_log` | Log of 8-bar rolling average dollar volume |
| `hour_of_day` | Hour within the trading day (14–21 UTC / 9–16 ET) |
| `day_of_week` | Calendar seasonality (0=Mon, 4=Fri) |
| `last_close_log` | Log of prior-bar close |

**Hourly spread tiers** (volume-based, daily tiers ÷ 6.5):
- `>$1.54B/bar → 2 bps`, `>$77M → 3 bps`, `>$15M → 7 bps`, `>$7.7M → 12 bps`, `>$1.54M → 25 bps`, else 50 bps

## Model

Same XGBClassifier as daily, but trained on hourly features:

```python
XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    objective="binary:logistic",
)
```

Target: `target_oc_up = 1` if `(close - open) / open > 0` for that hourly bar.

## Cost Model

Per hourly trade:
```
spread_bps  = volume-based tier (2–50 bps, hourly-scaled)
gross_lev   = leverage × (close - open) / open
cost        = leverage × (spread + 2×commission) / 10000
              + (leverage-1) × 6.25% / 252 / 6.5   (margin prorated per hour)
net         = gross_lev - cost
```

## Limitations vs Daily Strategy

| Dimension | Daily | Hourly |
|-----------|-------|--------|
| Universe | 846 stocks (all with CSVs) | ~13 stocks (MKTD file) |
| Data available | ✓ (trainingdata/*.csv) | ✗ (MKTD = RL tensors) |
| Chronos2 signals | ✓ (cc_return cache) | ✗ (no hourly Chronos) |
| Backtested | ✓ (Jan–Apr 2026) | ✗ (blocked by data) |

The daily strategy (`run_daily.py`) is fully operational with 846 stocks. The hourly strategy is ready in code but requires genuine hourly OHLCV data in MKTD v2 format.

## Running

```bash
# Hourly backtest (requires genuine hourly MKTD file)
python -m xgbnew.run_hourly \
    --mktd-file path/to/genuine_hourly_ohlcv_v2.bin \
    --top-n 2 --leverage 1.0 \
    --output-dir analysis/xgbnew_hourly

# With 2x leverage
python -m xgbnew.run_hourly --mktd-file <file> --top-n 2 --leverage 2.0

# Verbose (see training progress and feature importances)
python -m xgbnew.run_hourly --mktd-file <file> -v
```

## Files

```
xgbnew/
  features.py         — HOURLY_FEATURE_COLS + build_features_for_symbol_hourly()
  mktd_reader.py      — MKTD v2 binary reader (timestamp overflow guard included)
  run_hourly.py       — Hourly pipeline (load → features → train → backtest)
  model.py            — XGBStockModel (shared with daily)
  backtest.py         — BacktestConfig, DayResult, etc. (shared with daily)
```

## Expected Performance (Hypothetical)

With genuine hourly OHLCV data, we would expect:
- Lower per-trade returns (hourly bars have smaller moves than daily)
- Higher trade frequency (6.5× more bars per day than daily)
- Similar Sharpe if the model captures intraday patterns (momentum, mean-reversion)
- Higher transaction costs as a % of return (need tighter spreads to be viable)

The daily strategy (+203% total, Sharpe 13.9 over 66 days) is a stronger result
because it has 846 stocks to select from vs ~13 in any one MKTD hourly file.
