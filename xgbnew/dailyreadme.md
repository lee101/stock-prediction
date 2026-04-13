# XGBoost Daily Open-to-Close Strategy

## Overview

An XGBoost classifier trained to predict daily open-to-close direction across 846 US stocks, blended with optional Chronos2 forecast signals. Strategy: buy top-N stocks at market open, sell at close, using realistic spread + commission costs.

## Data

- **Universe**: 846 US stocks from `symbol_lists/stocks_wide_1000_v1.txt`
- **OHLCV**: `trainingdata/*.csv` (daily bars, 2020–2026)
- **Train**: 2021-01-01 → 2024-12-31 (546k rows after liquidity filter)
- **Val**: 2025-01-01 → 2025-11-30 (185k rows)
- **Test**: 2026-01-05 → 2026-04-09 (45k rows, 66 trading days)

## Features

All features computed with strict no-lookahead: `feature[D]` uses only data available at end of day D-1.

| Feature | Description |
|---------|-------------|
| `ret_1d … ret_20d` | Returns over 1/2/5/10/20 day lookbacks using `close.shift(1)` |
| `rsi_14` | Wilder's RSI(14) on prior-day close series |
| `vol_5d`, `vol_20d` | Annualized realized volatility of log returns |
| `atr_14` | Average True Range over 14 days, fraction of price |
| `cs_spread_bps` | Corwin-Schultz H/L spread estimate (model feature — captures intraday liquidity volatility) |
| `dolvol_20d_log` | Log of 20-day average dollar volume (top importance feature) |
| `price_vs_52w_high` | Current price / 52-week high |
| `price_vs_52w_range` | Position within 52-week H/L range (0–1) |
| `day_of_week` | Calendar seasonality (0=Mon, 4=Fri) |
| `last_close_log` | Log of prior-day close (price scale context) |

**Spread cost model** (separate from features): volume-based tiers applied in backtest:
- `>$10B/day → 2 bps`, `>$500M → 3 bps`, `>$100M → 7 bps`, `>$50M → 12 bps`, `>$10M → 25 bps`, else 50 bps

**Chronos2 features** (zero when cache unavailable): `chronos_oc_return`, `chronos_cc_return`, `chronos_pred_range`, `chronos_available`.

## Model

```python
XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    objective="binary:logistic",
)
```

Target: `target_oc_up = 1` if `(close - open) / open > 0`. Val accuracy: **51.67%** (barely above chance for the full 185k-row distribution).

## Cost Model

```
spread_bps  = volume-based tier (2–50 bps, see above)
gross_lev   = leverage × (close - open) / open
cost        = leverage × (spread + 2×commission) / 10000
              + (leverage-1) × 6.25% / 252     (intraday margin if L>1)
net         = gross_lev - cost
```

Filter: `max_spread_bps=30` (excludes stocks with spread tier > 30 bps), `min_dollar_vol=5M`.

## Backtest Results (Jan–Apr 2026, 66 trading days)

| Config | Total% | Monthly% | Sharpe | Max DD% | Dir Acc% | Avg Spread |
|--------|--------|----------|--------|---------|----------|------------|
| top2, 1x leverage | **+203%** | **+42%** | **13.9** | 6.2% | 85.6% | 2.7 bps |
| top2, 2x leverage | +767% | +99% | 13.8 | 12.5% | 85.6% | 2.7 bps |
| pure XGB (xw=1) | +203% | +42% | 13.9 | 6.2% | 85.6% | 2.7 bps |
| pure Chronos2 cc_return (xw=0) | -15% | -5% | -0.2 | 36% | 52.3% | 13 bps |
| Chronos2 standalone backtest | -33% | -13% | -2.2 | 36% | — | 76 bps |

## Key Findings

**XGB technical features strongly outperform Chronos2 on this test window.**
- Pure XGB: +203% total, Sharpe 13.9
- Pure Chronos2 (cc_return signal, blended): -15%
- Chronos2 standalone (`scripts/chronos_top2_backtest.py`): -33%, Sharpe -2.2

The Chronos2 cc_return signal uses trend extrapolation — always picks the strongest recent trending stocks (DXST, APP, SHOP) whose predicted returns (+20–30% "per day") are pure trend continuation artifacts. This is a static ranking that doesn't reflect actual open-to-close edge.

**Model is a momentum/liquidity quality selector.** Top features: `dolvol_20d_log` (0.130), `cs_spread_bps` (0.122), `ret_2d` (0.095), `vol_20d` (0.085), `ret_1d` (0.085). Consistently picks high-volume tech: MU (×18), PLTR (×14), INTC (×12), AVGO (×11).

## Caveats / Risk Factors

1. **Test window is too short and too favorable.** Jan–Apr 2026 coincided with a tech bull run (AI/semiconductor cycle). The 51.67% val accuracy on 2025 data vs 85.6% test directional accuracy indicates test-period overfitting.

2. **Selectivity inflates accuracy.** 51.67% average accuracy across all stocks vs 85.6% for the top-2 picks. The model's most confident picks are momentum stocks that continued during a bull market — not repeatable in bearish/choppy conditions.

3. **High concentration.** Top 10 most-picked stocks dominate. Real execution of PLTR/MU daily would show market impact at meaningful size.

4. **Annualized returns are misleading.** Annualizing 66 days gives +6,800% to +382,000% — not planning-relevant.

5. **No Chronos2 oc_return signal tested.** The cache has only `cc_return_pct` (close-to-close, trend extrapolation). The `oc_return_pct` (predicted intraday swing) was not generated and would likely provide better signal.

## Running

```bash
# Full run (train 2021-2024, test Jan-Apr 2026)
python -m xgbnew.run_daily \
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
    --data-root trainingdata \
    --chronos-cache analysis/top2_backtest/forecast_cache \
    --top-n 2 --leverage 1.0 \
    --output-dir analysis/xgbnew_daily

# With 2x leverage
python -m xgbnew.run_daily --top-n 2 --leverage 2.0

# Pure XGB (ignore Chronos2)
python -m xgbnew.run_daily --xgb-weight 1.0

# Pure Chronos2 rank
python -m xgbnew.run_daily --xgb-weight 0.0
```

## Files

```
xgbnew/
  features.py     — Feature engineering (no-lookahead, volume-based spread)
  dataset.py      — Dataset builder: loads CSVs, attaches Chronos2 cache
  model.py        — XGBStockModel wrapper + combined_scores()
  backtest.py     — BacktestConfig, simulate(), print_summary()
  run_daily.py    — Main daily pipeline (train → val → backtest)
  run_hourly.py   — Hourly pipeline (see hourlyreadme.md)
  mktd_reader.py  — MKTD v2 binary reader for hourly data
tests/
  test_xgbnew_features.py  — 23 unit tests (all passing)
analysis/xgbnew_daily/
  trades_*.csv     — Per-trade logs
  summary_*.json   — Config comparison JSON
```
