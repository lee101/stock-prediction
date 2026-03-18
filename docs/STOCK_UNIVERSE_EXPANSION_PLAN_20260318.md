# Stock Universe Expansion Plan 2026-03-18

## Current Live State

- Supervisor stock trader is already running `PAPER=0` live in [`deployments/unified-stock-trader/supervisor.conf`](/nvme0n1-disk/code/stock-prediction/deployments/unified-stock-trader/supervisor.conf).
- Current live universe is `NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE`.
- The last 24 hours showed no new signal-driven stock executions, so expansion should focus on adding valid opportunities without skipping model quality gates.

## Requested Batch

- Manifest written to [`docs/stock_universe_candidates_20260318.json`](/nvme0n1-disk/code/stock-prediction/docs/stock_universe_candidates_20260318.json).
- Already live and cache-ready from the requested batch: `NVDA`, `TSLA`.
- History-ready but not yet cache-ready: `SOFI`, `INTC`, `MU`, `F`, `PFE`, `TTD`.
- Missing hourly stock history and therefore blocked on ingest first: `PLUG`, `ONDS`, `AAL`, `NOK`, `BMNR`, `NBIS`, `TME`, `MARA`, `BTG`, `OWL`, `RIG`, `ABEV`, `ITUB`, `HIMS`, `PATH`, `RCAT`, `RKLB`.

## Immediate Trial Order

1. `SOFI`
2. `INTC`
3. `MU`
4. `PFE`
5. `F`
6. `TTD`

This order prefers names with existing hourly history, good liquidity, and factor diversification before moving to thinner or shorter-history names.

## Execution Plan

### Phase 1: Data Readiness

- Ingest missing hourly histories with [`download_hourly_data.py`](/nvme0n1-disk/code/stock-prediction/download_hourly_data.py) before attempting any cache build or backtest.
- Refresh the history-ready batch first:

```bash
source .venv313/bin/activate
python download_hourly_data.py --only-stocks --symbols SOFI,INTC,MU,F,PFE,TTD
python download_hourly_data.py --only-stocks --symbols PLUG,ONDS,AAL,NOK,BMNR,NBIS,TME,MARA,BTG,OWL,RIG,ABEV,ITUB,HIMS,PATH,RCAT,RKLB
```

### Phase 2: Cache Build for Trial Candidates

- Use the current live cache root and horizon layout so trial symbols match production expectations.

```bash
source .venv313/bin/activate
python scripts/build_hourly_forecast_caches.py \
  --symbols SOFI,INTC,MU,F,PFE,TTD \
  --data-root trainingdatahourly/stocks \
  --forecast-cache-root unified_hourly_experiment/forecast_cache \
  --horizons 1,6,24 \
  --lookback-hours 5000 \
  --force-rebuild
```

### Phase 3: Single-Symbol Chronos2 Retrains

- Start with single-symbol LoRA baselines before any multivariate coupling.
- Use [`scripts/retrain_chronos2_hourly_loras.py`](/nvme0n1-disk/code/stock-prediction/scripts/retrain_chronos2_hourly_loras.py) per symbol and record validation MAE, test MAE, and downstream market-sim PnL.

```bash
source .venv313/bin/activate
python scripts/retrain_chronos2_hourly_loras.py \
  --symbol SOFI \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 1024 \
  --learning-rate 5e-5 \
  --num-steps 1500 \
  --save-name-suffix stockexp
```

- Repeat for `INTC`, `MU`, `PFE`, `F`, and `TTD`.

### Phase 4: Refit Re-Augmentation and Hyperparameters

- For each symbol, sweep context length, learning rate, and step count around the single-symbol baseline.
- Keep the first pass narrow: context `{512,1024}`, steps `{1000,1500,2500}`, learning rate `{1e-4,5e-5,2e-5}`.
- Promote only if lower MAE is accompanied by better or neutral 120-day market-sim PnL and no material drawdown regression.

### Phase 5: Correlation and Multivariate Groups

- Use rolling hourly return correlations over the latest 120 days, then fit covariate groups instead of hard-coding pairings.
- Current seed groups are:
  - `semis_and_ai`: `NVDA,MU,INTC,AAPL,MSFT,GOOG`
  - `growth_platforms`: `SOFI,TTD,TSLA,META,PLTR,GOOG`
  - `travel_and_consumer`: `AAL,BKNG,EXPE,TRIP,MTCH`
  - `value_and_defensive`: `F,PFE,NOK`
  - `international_adrs`: `TME,ABEV,ITUB`
  - `speculative_small_cap`: `ONDS,BMNR,NBIS,MARA,HIMS,PATH,RCAT,RKLB`
- Use the existing `--covariate-symbols` path in [`scripts/retrain_chronos2_hourly_loras.py`](/nvme0n1-disk/code/stock-prediction/scripts/retrain_chronos2_hourly_loras.py) for multivariate LoRA runs after the single-symbol baseline is stable.

### Phase 6: Iron Model Layer

- Treat the "iron model" as a stacker above per-symbol forecasts, not a replacement for the underlying Chronos2 specialists.
- First version should stay simple:
  - Inputs: h1, h6, and h24 quantiles, recent realized vol, spread proxy, correlation-cluster id, and recent sim PnL features.
  - Output: trade edge score and a no-trade gate.
- Use a regularized linear or tree-based model first so attribution stays inspectable and overfitting remains bounded.

### Phase 7: 120-Day Market Simulator Gate

- Keep using the same universe-expansion evaluation path in [`scripts/run_alpaca_stock_expansion.py`](/nvme0n1-disk/code/stock-prediction/scripts/run_alpaca_stock_expansion.py), but extend it from the current `--robust-60d` posture to a 120-day window.
- Reuse the existing early-exit helper in [`src/market_sim_early_exit.py`](/nvme0n1-disk/code/stock-prediction/src/market_sim_early_exit.py) and add comparability gates against the current live baseline.
- Proposed early-out rules for 120-day runs:
  - After 30 percent of bars, stop if candidate max drawdown is already materially worse than baseline and return is negative.
  - After 50 percent of bars, stop if Sortino is far below baseline and drawdown still exceeds profit.
  - After 75 percent of bars, stop if both return and Sortino remain outside promotion bounds.
- Promotion should require:
  - Better total return than baseline or a clearly better Sortino with similar return.
  - No worse than baseline max drawdown by more than a predefined tolerance.
  - Stable trade count, with no pathological overtrading or total inactivity.

## Redeploy Rule

- Do not add a new symbol to the live Supervisor config until it has:
  - Hourly history present.
  - Production-format forecast caches in `unified_hourly_experiment/forecast_cache`.
  - A passing 120-day market-sim comparison versus the current live baseline.
- Once a candidate passes, add it one symbol at a time to [`deployments/unified-stock-trader/supervisor.conf`](/nvme0n1-disk/code/stock-prediction/deployments/unified-stock-trader/supervisor.conf), restart Supervisor, and watch the next recompute cycle before promoting the next symbol.
