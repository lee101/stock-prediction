# Marketsimulator Real-Analytics Smoke Tests (24 Oct 2025)

## Configuration
- **Date:** 24 October 2025 (UTC baseline)
- **Script:** `python marketsimulator/run_trade_loop.py`
- **Symbols:** `COUR GOOG TSLA NVDA AAPL U ADSK ADBE MSFT COIN AMZN AMD INTC QUBT BTCUSD ETHUSD UNIUSD`
- **Initial cash:** \$100,000
- **Iterations:** `--steps 2` (overnight/open-close cadence) with 24-hour strides; `TRADE_STATE_SUFFIX=sim_realfix` to isolate state.
- **Analytics target:** real Kronos/Toto stack with GPU (RTX 3090 Ti), `TOTO_NUM_SAMPLES=16`, `MARKETSIM_KRONOS_SAMPLE_COUNT=8`, `TOTO_DISABLE_COMPILE=1` (torch.compile repeatedly failed with missing artifact directories), `SIM_LOGURU_LEVEL=INFO` to surface summary logs.
- **Outcome:** Both Kronos and Toto forecasting calls still fall back to the simulator analytics because the cached CSV slices emitted by `marketsimulator/data_feed.py` lack the expected `High/Low` columns for several symbols (warning spam between 22:09–22:12 UTC). No live trades were admitted—every strategy line failed the walk-forward Sharpe / dollar-volume gates—so the run closes flat.

## Summary
- **Final equity:** \$100,000.00 (cash \$100,000.00, no open positions).
- **PnL:** +\$0.00 (+0.00%) over the two-step window, annualised ≈ 0.00 % (252d) / 0.00 % (365d).
- **Symbol PnL breakdown:** empty (no realised or open PnL).
- **Runtime observations:** Kronos/Toto continue to fall back immediately. Even with `TOTO_DISABLE_COMPILE=1`, Kronos still aborts when OHLC data are missing. All shortlisted names remain blocked by the risk gates (Sharpe < 0.30 or dollar volume below thresholds), mirroring the behaviour we observed in the longer 30-step sandbox run.

## Key Follow-ups
1. **Restore OHLC coverage for the simulator feeds.** The mock data loader still surfaces price frames that lack `High/Low` for AMZN/AMD/INTC/COIN/BTCUSD/ETHUSD/UNIUSD. Until those pipelines are refreshed, the real backtest path will always drop to the fallback engine.
2. **Re-enable Toto compilation after populating cache directories.** Torch.compile currently fails with `compiled_models/torch_inductor/...main.cpp: No such file or directory`; we disabled compilation to keep the run moving, but the GPU inference path will remain CPU-bound until we fix the cache bootstrap in `backtest_test3_inline._ensure_compilation_artifacts`.
3. **Probe mode clean-up is effective but underutilised.** With the relaxed mock analytics, risk gates prevent all new entries; once the OHLC issue is addressed we should revisit the NVDA/AAPL/MSFT probes with the compiled models to validate the utility of the probe-to-normal transitions.

---

# Marketsimulator Strategy Isolation Runs (23 Oct 2025)

## Simulation Configuration
- **Date:** 23 October 2025 (UTC baseline)
- **Script:** `python marketsimulator/run_trade_loop.py`
- **Step cadence:** `--step-size 12` (advances the hourly simulator data by ~12 hours per iteration to approximate an **open / close** schedule)
- **Iterations:** 30 steps (≈15 simulated trading days)
- **Symbols:** `AAPL MSFT NVDA AMZN META GOOG TSLA BTCUSD ETHUSD SOLUSD`
- **Initial cash:** \$100,000
- **State isolation:** unique `TRADE_STATE_SUFFIX` per strategy to avoid cross-run contamination
- **Analytics mode:** fallback simulator analytics (real Kronos/Toto modules are unavailable in this environment, so deterministic mocks were used)
- **Strategy selection:** forced via the new `TRADE_STRATEGY_WHITELIST` flag (one strategy per run)
- **Risk toggles:** default production gates; `MARKETSIM_DISABLE_SIM_RELAX` left unset so relaxed thresholds were applied even without forcing Kronos-only mode

All runs completed without runtime failures, but every active-strategy simulation finished with sizable drawdowns and uncleared inventory. The lightweight mock analytics trend bearish for the entire window, so strategies continuously accumulated short exposure without receiving profitable exits.

## Summary Table (Equity after 30 steps)

| Strategy whitelist | Final equity | PnL vs \$100k | Annualised (252d / 365d) | Cash balance | Residual positions (qty) | Observations |
|--------------------|-------------:|--------------:|------------------------:|-------------:|--------------------------|--------------|
| `maxdiff` (High/Low breakout) | \$64,342.98 | **−\$35,657.02** (−35.66%) | −25.70% / −26.58% | \$64,752.66 | `AAPL 1.0`, `MSFT 1.0` | Kelly=15 % sizing plus relaxed gates keeps re-probing short setups; equity ends ~36% lower even though cash remains high from short proceeds. |
| `entry_takeprofit` | \$105,313.42 | +\$5,313.42 (+5.31%) | +3.55% / +3.69% | **\$176,920.06** | NVDA 24.60, GOOG 6.0, AAPL 108.45, MSFT 49.05, TSLA 36.15 | Bracket exits never trigger under the mock drift, so inventory keeps scaling. Cash balance balloons because the book is predominately short. |
| `simple` | \$99,090.64 | −\$909.36 (−0.91%) | −0.61% / −0.64% | \$125,836.79 | GOOG 1.0, MSFT 1.0, AAPL 1.0, NVDA 1.0, TSLA 33.9 | Similar behaviour without take-profit overlays—largest losses come from TSLA probes that never flip direction. |
| `ci_guard` | \$99,813.93 | −\$186.07 (−0.19%) | −0.13% / −0.13% | \$136,965.05 | NVDA 1.0, MSFT 1.0, AAPL 1.0, GOOG 6.0, TSLA 33.3 | Consensus gating trims exposure while retaining the same directional bias; residual drawdown is minor but still negative. |
| `all_signals` | \$100,000.00 | \$0.00 (+0.00%) | +0.00% / +0.00% | \$100,000.00 | none | No trades executed—the fallback analytics never produced unanimous direction, so this run serves as a control. |

## Detailed Notes

### High/Low (Maxdiff) Strategy
- Even with mock analytics, the engine continues to chase short setups and trip probe mode. The missing OHLC columns for crypto pairs mean only equities participate, but the repeated probes still realise ~36 % loss over the 30-step window.
- Ending risk is tiny (1 share apiece in AAPL/MSFT) because most of the damage is realised during the run; the annualised figures highlight how punitive the behaviour is without additional gating.

### Entry + Take Profit Strategy
- Take-profit brackets never hit under the persistent bearish drift, so Kelly sizing gradually loads up on short inventory. The strategy ekes out a small gain only because the mocked prices oscillate enough to recycle inventory mid-run.
- Large positive cash balance signals the same exposure pattern—shorts fund cash, while equity (which counts unrealised PnL) captures the risk that still remains.

### Simple Strategy
- Similar to entry/takeprofit but without explicit take-profit legs; losses are modest (−0.9 %) but driven by TSLA probes that never flipped back to long.
- Small unit positions in AAPL, GOOG, MSFT and NVDA remain because the mock analytics occasionally permitted micro trades.

### CI Guard Strategy
- Consensus gating keeps PnL near flat (−0.2 %) while still holding a TSLA short clip. The guardrails appear to help the most when the analytics disagree, but they do not eliminate the need for probe cleanup.

### All-Signals Strategy
- Remains a useful control: without unanimous signals the strategy does nothing, giving us a clean baseline for the new summary instrumentation.

## Pair Allocation Sweep (TRADE_STRATEGY_WHITELIST=`simple`, 30 × 12h steps)

Mock analytics, `SIM_LOGURU_LEVEL=INFO`, unique `TRADE_STATE_SUFFIX`, `PORTFOLIO_DB_PATH` isolated per run.

| top-k | Final equity | PnL | Annualised (252d / 365d) | Cash | Notable holdings | Notes |
|------:|-------------:|----------------------------:|------------------------:|-------------:|-----------------------------|-------|
| 10 | \$105,410.20 | +\$5,410.20 (+5.41%) | +3.61% / +3.76% | \$176,976.40 | NVDA 24.60, GOOG 6.0, MSFT 54.9, AAPL 97.35, TSLA 36.15 | Same outcome as top‑6/top‑3 because only five names ever pass gating. |
| 6 | \$105,410.20 | +\$5,410.20 (+5.41%) | +3.61% / +3.76% | \$176,976.40 | (identical to top‑10) | Increasing `--top-k` above available qualified picks has no effect. |
| 3 | \$105,410.20 | +\$5,410.20 (+5.41%) | +3.61% / +3.76% | \$176,976.40 | (identical to top‑10) | Gatekeepers still return five symbols; surplus slots stay unused. |
| 2 | \$107,095.23 | +\$7,095.23 (+7.10%) | +4.73% / +4.92% | \$186,125.93 | NVDA 24.60, GOOG 6.0, MSFT 54.9, TSLA 30.15, AAPL 178.80 | Tighter cap concentrates sizing; Kelly still allocates across five names but notional per pick rises sharply. |

## Leverage Sweep (GLOBAL_MAX_GROSS_LEVERAGE, `simple`, top-k 6)

- Crypto legs remain capped at 1.0× due to the sizing change in `get_qty`.
- Financing costs apply via `marketsimulator.state._accrue_financing_cost()` using the annual 6.75% default unless otherwise noted.

| Max gross leverage | Final equity | PnL | Annualised (252d / 365d) | Cash | Notable holdings | Notes |
|-------------------:|-------------:|----------------------------:|------------------------:|-------------:|-----------------------------|-------|
| 1.0× | \$103,849.82 | +\$3,849.82 (+3.85%) | +2.58% / +2.68% | \$144,133.55 | NVDA 13.65, GOOG 5.70, MSFT 24.15, AAPL 44.10, TSLA 18.0 | Lower cap shrinks position sizes and keeps cash closer to equity. |
| 1.5× (default) | \$105,410.20 | +\$5,410.20 (+5.41%) | +3.61% / +3.76% | \$176,976.40 | NVDA 24.60, GOOG 6.0, MSFT 54.9, AAPL 97.35, TSLA 36.15 | Baseline reference from earlier table. |
| 2.0× | \$104,878.68 | +\$4,878.68 (+4.88%) | +3.26% / +3.39% | \$194,710.94 | NVDA 32.70, GOOG 7.95, MSFT 72.90, AAPL 143.55, TSLA 37.80 | Higher cap allows larger Kelly targets but financing drag trims edge. |
| 4.0× | \$115,498.35 | +\$15,498.35 (+15.50%) | +10.20% / +10.62% | \$231,985.66 | NVDA 65.55, GOOG 15.90, MSFT 143.40, AAPL 57.20, TSLA 33.90 | Extreme leverage drives big gains under the mock drift but leaves outsized short exposure; financing cost increases materially too. |

## Action Items & Recommendations
1. **Tighten exposure controls for mock runs.** Even without live Kronos/Toto, we should throttle Kelly sizing (e.g., cap at 2–5 %) or force liquidation after repeated probe warnings to keep simulated PnL bounded.
2. **Introduce post-run flattening.** The loop should include an optional liquidation pass so end-of-run equity reflects closed PnL; otherwise comparisons across strategies remain noisy.
3. **Improve fallback analytics realism.** The deterministic mock feed is overly bearish, producing monotonic sell signals. Consider injecting noise or re-enabling the real forecasting stack (once dependencies are available) before making strategy conclusions.
4. **Document required environment toggles.** When analysing individual strategies, call out the need to set `MARKETSIM_DISABLE_SIM_RELAX=1` if we want production-grade gating, or bespoke limits if relaxed thresholds are desired.

Logs for each run are stored under `/tmp/sim_<strategy>30.log` for deeper inspection. Update this report as we iterate on model configuration, Kelly caps, or after re-running with the compiled Toto/Kronos pipelines.

## Full-Universe Mock Sweep (18 symbols, 40 × 12h steps)

Symbols: `COUR GOOG TSLA NVDA AAPL U ADSK ADBE MSFT COIN AMZN AMD INTC QUBT BTCUSD ETHUSD UNIUSD`

| Strategy | Final equity | PnL | Annualised (252d / 365d) | Cash | Notable open positions | Observations |
|----------|-------------:|----------------------------:|------------------------:|-------------:|-------------------------|-------------|
| `simple` | \$99,615.27 | −\$384.73 (−0.38%) | −2.14% / −2.32% | \$101,716.41 | GOOG −1 @ 2,101 (open −\$2.1k) | Only GOOG fired; crypto and several equities skipped because fallback data lacks OHLC columns. |
| `entry_takeprofit` | \$91,455.54 | −\$8,544.46 (−8.54%) | −39.36% / −41.92% | \$147,054.36 | GOOG −13.8, MSFT −110.4 (combined open −\$55.6k) | Bracket exits never triggered; huge short inventory left open with deep unrealised loss. |
| `ci_guard` | \$95,428.94 | −\$4,571.06 (−4.57%) | −23.05% / −24.77% | \$146,048.66 | GOOG −13.8, QUBT −246.45 | Consensus gating reduces participation but still leaves large shorts. |
| `maxdiff` | \$92,227.69 | −\$7,772.31 (−7.77%) | −36.43% / −38.87% | \$92,227.69 | none (all positions closed) | High/low entries rarely hit profit targets; simulator keeps blocking new trades, leaving realised losses. |
| `maxdiff` (`MAXDIFF_DAYONLY=1`) | \$100,000.00 | +\$0.00 (+0.00%) | +0.00% / +0.00% | \$100,000.00 | none | Forcing day-end exits prevented losses but also blocked every trade under the mock analytics. Practical benefit depends on having more realistic forecasts. |

**Notes**
- The mock (`fast`) backtester skipped AMZN/AMD/INTC/crypto pairs because the fallback CSVs are missing `Close/High/Low`. Real Kronos/Toto forecasts (or refreshed data) are required before using this universe to judge strategy quality.
- The new per-symbol breakdown in `run_trade_loop` makes it clear where risk is concentrated (GOOG/QUBT). Any optimisation pass should address the persistent short bias in the fallback analytics.
- A follow-up run with `MAXDIFF_DAYONLY=1` and real analytics enabled (Kronos/Toto with GPU inference) produced no trades; the pipeline fell back to the same stub analytics owing to missing OHLC columns, reinforcing the need to restore the full forecasting stack before further conclusions.
- A trial with `MAX_KELLY_FRACTION=0.05` (simple strategy, 30×12 h, top-k 6) reduced equity to $97.23K (−2.77%). Given the degradation, the Kelly cap experiment was abandoned and the production sizing logic reverted.
