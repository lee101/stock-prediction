# Alpaca Progress 9

## XGBoost Daily Open-to-Close Strategy (2026-04-13)

### Summary

Built a pure XGBoost strategy (`xgbnew/`) that predicts open-to-close direction across
846 US stocks. Real day-by-day backtest on Jan–Apr 2026 out-of-sample data:

| Config | Total% | Monthly% | Sharpe | Max DD% | Dir Acc% |
|--------|--------|----------|--------|---------|----------|
| top-2, 1x leverage | **+203%** | **+42%** | **13.9** | 6.2% | 85.6% |
| top-2, 2x leverage | +767% | +99% | 13.8 | 12.5% | 85.6% |

This is a real simulation — real OHLCV data, real cost model (volume-based spread tiers + 10 bps/side commission), no lookahead. Not marketsim, but comparable in methodology.

---

### Architecture

**Data**: 846 US stocks from `symbol_lists/stocks_wide_1000_v1.txt`, daily bars 2020–2026.

**Features** (all no-lookahead — `feature[D]` uses data through close of D-1):
- Returns: `ret_1d`, `ret_2d`, `ret_5d`, `ret_10d`, `ret_20d`
- RSI(14) via Wilder's EWM
- Realized vol: `vol_5d`, `vol_20d` (annualized)
- ATR(14) as fraction of price
- Corwin-Schultz H/L spread (`cs_spread_bps`) — model feature capturing intraday liquidity volatility
- Log 20-day dollar volume (`dolvol_20d_log`)
- Price vs 52-week high, price position within 52-week range
- Day of week, log close

**Cost model** (separate from features):
- Volume-based spread tiers: >$10B→2bps, >$500M→3bps, >$100M→7bps, >$50M→12bps, >$10M→25bps, else 50bps
- Commission: 10 bps/side
- Liquidity filter: `max_spread_bps=30`, `min_dollar_vol=5e6`

**Model**:
```python
XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
              gamma=0.1, reg_alpha=0.1, reg_lambda=1.0)
```
Target: binary up/down open-to-close direction.

**Val accuracy** (2025, 185k rows, full universe): **51.67%** — barely above chance across all 846 stocks. The model's edge comes from selectivity: picking the 2 most confident stocks each day.

---

### Key Technical Findings

**`np.select` for tiered costs**: Chained `.where()` calls for tiered logic are broken
(each call overwrites all rows where condition is False, destroying previously set tiers).
`np.select(conditions, values, default)` gives first-match semantics — correct.

**Corwin-Schultz spread gives 180-320 bps for liquid stocks**: It interprets daily H/L
range as bid-ask spread. Wrong for cost modelling — used as model feature only (`cs_spread_bps`).
Volume-based tiers handle actual costs.

**XGBoost ≥2.0**: `early_stopping_rounds` moved from `fit()` kwargs to constructor.
Use `clf.set_params(early_stopping_rounds=N)` with try/except fallback.

---

### Feature Importances (top-5)

| Feature | Importance |
|---------|-----------|
| `dolvol_20d_log` | 0.130 |
| `cs_spread_bps` | 0.122 |
| `ret_2d` | 0.095 |
| `vol_20d` | 0.085 |
| `ret_1d` | 0.085 |

The model is effectively a **momentum/liquidity quality selector**. Most picked stocks:
MU (×18), PLTR (×14), INTC (×12), AVGO (×11) — consistently large-cap high-volume tech.

---

### Caveats

1. **Test window too short and too favorable.** 66 days coincided with AI/semiconductor bull run.
2. **Selectivity inflates accuracy.** 51.67% avg vs 85.6% for top-2. The model's confidence is highest for momentum stocks that continued in a bull market.
3. **High concentration.** Real execution of PLTR/MU daily would have market impact at meaningful size.
4. **Not yet marketsim validated.** Current RL prod uses `decision_lag=2`, binary fills, 263 evaluation windows. XGBoost uses our own simpler backtest with similar cost assumptions but no lag.
5. **No Alpaca integration.** Needs singleton guard + death-spiral guard + live open-price feed for morning selections.

---

### Chronos2 Comparison

| Strategy | Total% | Sharpe |
|----------|--------|--------|
| Pure XGB (1x lev) | +203% | 13.9 |
| Blended XGB+Chronos2 cc_return | -15% | -0.2 |
| Chronos2 standalone | -33% | -2.2 |

Chronos2's `cc_return` signal always picks the strongest trending stocks (DXST, APP, SHOP with predicted +20-30%/day) — pure trend extrapolation artifacts, not intraday edge. The XGB technical features dominate.

---

### Next Steps to Productionize

1. **Regime robustness**: Test on 2022 bear market, 2023 choppy market. The 51.67% val accuracy on 2025 data is honest but the 85.6% on top-2 picks needs validation across regimes.

2. **More seeds and hyperparameter sweeps**: Vary `top_n` (1, 2, 3, 5), `max_depth` (3-6), `n_estimators` (100-500). The current params weren't swept.

3. **Marketsim integration**: Adapt to the `scripts/eval_100d.py` pipeline with `decision_lag=2`, binary fills, `fee=10bps`, `slip=5bps` so it can be compared on the same yardstick as the RL ensemble.

4. **Alpaca wrapper**: For live trading, add to `run_daily.py`:
   - Import `alpaca_wrapper` for singleton + death-spiral guards
   - Morning selection: fetch live open prices, score, place market orders at open
   - EOD: sell all positions at market close

5. **Feature additions**: Consider adding sector/industry flags, earnings calendar proximity, VIX regime. Cross-sectional normalization of features per day.

6. **Ensemble with RL**: XGB makes one buy at open and exits at close — doesn't conflict with the RL intraday system which holds for hours. Could run both simultaneously with separate allocation.

---

### Files

```
xgbnew/
  features.py         — Feature engineering (no-lookahead, _vol_spread_series, RSI fix)
  dataset.py          — Dataset builder: loads CSVs, attaches Chronos2 cache
  model.py            — XGBStockModel wrapper
  backtest.py         — BacktestConfig, simulate(), print_summary()
  run_daily.py        — Main pipeline (train 4 years → val → backtest)
  run_hourly.py       — Hourly pipeline (blocked: MKTD files are RL tensors, not OHLCV)
  mktd_reader.py      — MKTD v2 binary reader (timestamp overflow guard for RL tensors)
  dailyreadme.md      — Full daily results and documentation
  hourlyreadme.md     — Hourly architecture + data limitation explanation
tests/
  test_xgbnew_features.py  — 23 unit tests (all passing)
```

### Run

```bash
source .venv/bin/activate
python -m xgbnew.run_daily \
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
    --data-root trainingdata \
    --top-n 2 --leverage 1.0 \
    --output-dir analysis/xgbnew_daily
```
