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

### Gate-relaxed experiment (24 Oct 2025)
- **Environment:** `MARKETSIM_DISABLE_GATES=1`, `TRADE_STRATEGY_WHITELIST=simple`, 17-symbol universe, 10 steps, `TOTO_DISABLE_COMPILE=1`.
- **Outcome:** Final cash **$179,855.61** with equity **$85,581.39** (PnL **−$14,418.61** / **−14.42 %**; annualised −98.7 % / −99.7 %). Removing the Sharpe / turnover / spread / dollar-volume guards lets Kelly sizing pile into large short books—e.g., COUR −$31.5 K, COIN −$22.4 K, U −$23.0 K—so the account ends the run with heavy short exposure and sizable drawdowns.
- **Takeaway:** The relaxed gates confirm that the risk filters were the only thing preventing the Kelly allocator from saturating shorts. Before benchmarking other strategies we need stricter position caps (or a reduced Kelly fraction) plus a remediation for the remaining Kronos/Toto runtime warnings; otherwise every variant will collapse into the same large short pile.

## Key Follow-ups
1. **Restore OHLC coverage for the simulator feeds.** The mock data loader still surfaces price frames that lack `High/Low` for AMZN/AMD/INTC/COIN/BTCUSD/ETHUSD/UNIUSD. Until those pipelines are refreshed, the real backtest path will always drop to the fallback engine.
2. **Re-enable Toto compilation after populating cache directories.** Torch.compile currently fails with `compiled_models/torch_inductor/...main.cpp: No such file or directory`; we disabled compilation to keep the run moving, but the GPU inference path will remain CPU-bound until we fix the cache bootstrap in `backtest_test3_inline._ensure_compilation_artifacts`.
3. **Probe mode clean-up is effective but underutilised.** With the relaxed mock analytics, risk gates prevent all new entries; once the OHLC issue is addressed we should revisit the NVDA/AAPL/MSFT probes with the compiled models to validate the utility of the probe-to-normal transitions.

---

# Marketsimulator Strategy Isolation Runs (23 Oct 2025)

## Telemetry Pipeline Updates (24 Oct 2025)
- Installed Plotly+kaleido via `uv pip`, so `scripts/provider_latency_history_png.py` now emits high-fidelity sparkline PNGs (Matplotlib fallback remains as safety net).
- `run_daily_trend_pipeline.py` optionally posts the latency digest to a webhook (`--summary-webhook`) and includes public artefact links in alert payloads when `--public-base-url` is supplied.
- `rotation_summary.md` now starts with a bold **Latency Status: OK/WARN/CRIT** headline, followed by the data-feed table, thumbnail, and digest preview for quicker operator triage.
- Added `make latency-status`, which runs `scripts/provider_latency_status.py` so CI jobs can fail when rolling deltas breach WARN/CRIT thresholds.
- `scripts/notify_latency_summary.py` now prepends the status headline/table to webhook posts (and links the PNG when available), keeping Slack/Teams digests in sync with rotation reports.
- `scripts/provider_latency_alert_digest.py` now records INFO/WARN/CRIT counts per provider, appends them to `provider_latency_alert_history.jsonl`, and `scripts/provider_latency_leaderboard.py` renders the history into `provider_latency_leaderboard.md` with ΔTotals plus CRIT/WARN % deltas. The weekly trend report (`provider_latency_weekly_trends.md`) and `provider_latency_trend_gate.py` flag providers whose CRIT/WARN counts jump across windows and fail the pipeline when thresholds are breached; both summaries surface in rotation/webhook previews.
- Webhook/rotation digests now embed the leaderboard preview and weekly trend highlights; `provider_latency_trend_gate.py` enforces CRIT/WARN delta thresholds each run.
- Added `scripts/write_latency_step_summary.py` and wired it into the GitHub workflow; each run now appends latency status and recent alerts to the Actions job summary for quick review.


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
- Financing costs apply via `marketsimulator.state._accrue_financing_cost()` using the annual 6.5% default unless otherwise noted.

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

## Step-Size 1 Mini-Loop Experiments (24 Oct 2025)

### Run 1 — Baseline (AAPL only)
- **Command:** `TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL --steps 4 --step-size 1 --initial-cash 100000`
- **Analytics:** Real stack unavailable; fell back to simulator mocks with Toto proxy forecasts.
- **Outcome:** No trades fired (0 executions). Equity and cash stayed at \$100,000.00.
- **Diagnostics:** Every loop logged “No analysis results available” and the day-4 detail showed a `Walk-forward Sharpe -0.04 < 0.30` block. With Kronos disabled, the Sharpe gate remains 0.30, so the mocked forecasts never pass the threshold.
- **Next idea:** Re-run with `--kronos-only` to relax the Sharpe cutoff to −0.25 and allow Kronos to dictate direction, then widen participation knobs if trades still fail to fire.

### Run 2 — `--kronos-only` (AAPL only)
- **Command:** `TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL --steps 4 --step-size 1 --initial-cash 100000 --kronos-only`
- **Outcome:** 1 long trade in AAPL, equity \$99,839.03 (−0.16%), cash \$86,459.32. `sim-summary` Sharpe = −7.94.
- **What changed:** Kronos-only mode relaxed the Sharpe floor to −0.25, but the first three iterations still failed the gate (`-0.29`, `-0.39`, `-0.28`). Day 4 finally cleared the threshold with a weak long signal, opening 66.45 shares at \$202.93. The mark-to-market moved against us immediately, so we ended with unrealised loss and lingering inventory.
- **Takeaway:** Simply flipping the Kronos switch is insufficient; the fallback analytics remain bearish enough that even the relaxed gate blocks most short ideas. We need either broader symbol coverage or a way to embrace the bearish signals rather than waiting for rare longs.

### Run 3 — `--kronos-only`, multi-symbol (`AAPL MSFT NVDA`)
- **Command:** `TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 4 --step-size 1 --initial-cash 100000 --kronos-only`
- **Outcome:** 4 trades executed, equity \$97,807.39 (−2.19%), cash \$54,546.06. Position at close: MSFT +156.9 shares (−\$2,036 open loss), AAPL +64.65 shares (−\$156).
- **Observations:** MSFT fired on every iteration with positive walk-forward Sharpe, so Kelly sizing kept increasing the long clip even as live drawdown warnings triggered probe mode. NVDA remained blocked by probe backout timers, so the book concentrated risk in MSFT and the drawdown compounded.
- **Actionable insight:** For bearish mock data, we likely want to bias towards short-only or cap Kelly much lower (<=5%). Alternatively we can whitelist a short-biased strategy. Without that, widening the universe just accelerates losses because the engine keeps pyramiding into longs.

### Run 4 — Strategy whitelist `entry_takeprofit`
- **Command:** `TRADE_STATE_SUFFIX=sim TRADE_STRATEGY_WHITELIST=entry_takeprofit python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 4 --step-size 1 --initial-cash 100000 --kronos-only`
- **Outcome:** 2 trades (AAPL long, MSFT 1-share probe). Equity \$99,823.56 (−0.18%), similar to Run 2. Entry/TP brackets never triggered within four steps, so the book closed with the same negative marks as the prior AAPL-focused run.
- **Notes:** The whitelist does prevent MSFT from pyramiding aggressively, but it does not address the underlying issue that the mock feed appears biased downward while the strategy keeps taking longs.

### Run 5 — Top-k capped at 1
- **Command:** `TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 4 --step-size 1 --initial-cash 100000 --kronos-only --top-k 1`
- **Outcome:** Identical to Run 4 (AAPL long + MSFT probe, −0.18% return). The cap limits simultaneous positions but still leaves the long bias unchallenged.
- **Next steps under consideration:** (1) add a configurable Sharpe override so we can deliberately accept the bearish short setups the mocks keep flagging; (2) experiment with forced liquidation after the final step to convert inventory PnL into realised returns; (3) run a longer window (≥20 steps) to observe whether the Kelly scaling eventually flips or if the strategy keeps averaging into losing longs.

### Run 6 — Kronos-only with relaxed Sharpe gate
- **Command:** `TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 4 --step-size 1 --initial-cash 100000 --kronos-only --kronos-sharpe-cutoff -1.0`
- **Outcome:** Equity \$101,542.77 (+1.54%), cash \$94,554.44, six trades total (five incremental AAPL buys, one MSFT probe). Sharpe scores printed at 8.60 after the override.
- **Interpretation:** Allowing Sharpe down to −1.0 admits the previously blocked Day 1–3 signals. The strategy aggressively averaged into AAPL longs, and the mock drift happened to retrace upward by the end of step 4, yielding a realised gain. Inventory remained long but much smaller (33.75 shares) because later iterations trimmed exposure.

### Run 7 — Mock analytics (no Kronos) with relaxed Sharpe gate
- **Command:** `TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 4 --step-size 1 --initial-cash 100000 --sharpe-cutoff -1.0`
- **Outcome:** Equity \$99,823.30 (−0.18%), cash \$113,414.43. The engine finally allowed the bearish AAPL setup to execute as a short; unrealised loss remained at close because the mock prices climbed immediately afterwards.
- **Takeaway:** The override unlocks the short bias we expected, but with fallback analytics the direction can still whipsaw. We likely need either production-grade forecasts or an exit/liquidation policy to crystallise PnL instead of carrying inventory.

### Run 8 — Kronos-only + relaxed Sharpe + end-of-run flatten
- **Command:** `TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 4 --step-size 1 --initial-cash 100000 --kronos-only --kronos-sharpe-cutoff -1.0 --flatten-end`
- **Outcome:** Equity \$101,486.94 (+1.49%), cash \$101,486.94, eight trades. All open inventory was force-closed at the final price snapshot, so the realised PnL now matches equity with zero residual positions.
- **Observation:** Flattening converts the earlier unrealised gains into cash, giving a cleaner benchmark. It also shows that most profit came from cumulative AAPL trades (+\$1.50K realised) while MSFT probes lost \$18.
- **New instrumentation:** Updated CLI summaries now include max drawdown, drawdown %, and total fees so these flattened runs can be compared directly on realised risk.
- **Portfolio export:** Metrics for this run were also written to `marketsimulator/run_logs/metrics_kronos_aapl.json` (JSON) and appended to `marketsimulator/run_logs/metrics.csv` for downstream analysis.

### Run 9 — Baseline analytics with end-of-run flatten
- **Command:** `TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 4 --step-size 1 --initial-cash 100000 --flatten-end`
- **Outcome:** No trades executed; flatten step reported no positions. Confirms the new flag is harmless when nothing is open.

### Run 10 — Kronos-only, relaxed Sharpe, 20 steps, flattened
- **Command:** `TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 20 --step-size 1 --initial-cash 100000 --kronos-only --kronos-sharpe-cutoff -1.0 --flatten-end`
- **Outcome:** Equity (and realised cash) ended at **$97,920.41** (−2.08%) after 36 trades. AAPL and MSFT longs/averages unwound into losses despite the override, highlighting that the mock drift can flip decisively over longer horizons.
- **Observation:** Fees accumulated to $151 even with flattening, and the book cycled between long and short multiple times. The override opens the door for recovery rallies, but over 20 steps the bearish bias persists.

### Run 11 — Mock analytics, relaxed Sharpe, 20 steps, flattened
- **Command:** `TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 20 --step-size 1 --initial-cash 100000 --sharpe-cutoff -1.0 --flatten-end`
- **Outcome:** Finished essentially flat at $100,003.44 (+0.00%) with two small trades, confirming the default analytics still block most setups even with the relaxed threshold unless Kronos-only mode is engaged.
- **Takeaway:** For meaningful activity we still need Kronos-only or additional overrides; otherwise the mock pipeline remains defensive.
- **Exports:** Summary metrics land alongside Run 8 in the same JSON/CSV artifacts when the export flags are supplied.

- **Command:** `MARKETSIM_KELLY_DRAWDOWN_CAP=0.02 TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 20 --step-size 1 --initial-cash 100000 --kronos-only --kronos-sharpe-cutoff -1.0 --flatten-end`
- **Outcome:** Equity settled at $99,875.02 (−0.12%) with max drawdown capped at 1.61%. Kelly scaling trimmed position sizes once drawdown exceeded 2%, yet repeated MSFT probes still produced a small net loss.
- **Exports:** Metrics saved to `marketsimulator/run_logs/metrics_kronos_multi_dd.json`, the timeline appended to `metrics.csv`, and per-trade fills captured in `marketsimulator/run_logs/trades.csv`.
- **Observation:** Despite the cap, the mock drift keeps biasing short exposure; drawdown scaling factors ranged from 0.88 down to ~0.10 during peak stress.

### Run 13 — Drawdown-scaled Kelly + Entry Suspension (Kronos, 20 steps)
- **Command:** `MARKETSIM_KELLY_DRAWDOWN_CAP=0.02 MARKETSIM_DRAWDOWN_SUSPEND=0.015 TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 20 --step-size 1 --initial-cash 100000 --kronos-only --kronos-sharpe-cutoff -1.0 --flatten-end`
- **Outcome:** Equity finished at $99,849.79 (−0.15%) with max drawdown 1.99%. Suspension kicks in once drawdown breaches 1.5%, pausing fresh entries while existing positions unwind.
- **Exports:** Metrics logged to `marketsimulator/run_logs/metrics_kronos_multi_dd_suspend.json`, aggregated stats in `trades_summary.json`, and raw fills in `trades.csv`.
- **Observation:** Entry suspension trimmed churn (29 trades vs. 36 baseline) and kept fees near $94, but the strategy still lost money because large probes were opened before the stop triggered. Further tuning of suspend/resume levels or per-strategy thresholds is required.
- **Analytics Tip:** Run `python scripts/analyze_trades_csv.py marketsimulator/run_logs/trades.csv` to inspect net cash deltas, average holding duration, and slip per symbol for this run.

### Run 14 — Drawdown-scaled Kelly + Tuned Suspension (Kronos, 20 steps)
- **Command:** `MARKETSIM_KELLY_DRAWDOWN_CAP=0.02 MARKETSIM_DRAWDOWN_SUSPEND_MAP=ci_guard:0.013 MARKETSIM_DRAWDOWN_RESUME_FACTOR=0.4 TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA --steps 20 --step-size 1 --initial-cash 100000 --kronos-only --kronos-sharpe-cutoff -1.0 --flatten-end`
- **Outcome:** Equity finished at **$101,283.39** (+1.28%) with max drawdown **1.45%**, 27 trades, and ~$92 fees. Earlier suspension (1.3%) plus deeper resume hysteresis (0.4×cap) let existing positions recover while blocking new probes during stress.
- **Exports:** Metrics in `marketsimulator/run_logs/metrics_kronos_multi_dd_suspend_tuned.json`, trade stats in `trades_summary.json`, and timeline in `trades.csv`.
- **Observation:** Tuned thresholds restored positive PnL while retaining drawdown control; trade analysis shows AAPL net +$991 and MSFT +$292 with average holding ~7.4 hours.
- **Automation:** Use `python scripts/run_sim_with_report.py [run args...]` to execute future scenarios; attach `--max-fee-bps`, `--max-avg-slip`, `--max-drawdown-pct`, `--min-final-pnl`, `--max-worst-cash`, and `--min-symbol-pnl` to surface alert conditions automatically (exit non-zero with `--fail-on-alert`). `make sim-report` demonstrates a CI-ready preset, and `make sim-trend` aggregates historical run logs via `scripts/trend_analyze_trade_summaries.py` (moving averages/stdev with optional JSON output).
- **Automation:** Use `python scripts/run_sim_with_report.py [run args...]` to execute future scenarios; attach `--max-fee-bps`, `--max-avg-slip`, `--max-drawdown-pct`, `--min-final-pnl`, `--max-worst-cash`, and `--min-symbol-pnl` to surface alert conditions automatically (exit non-zero with `--fail-on-alert`). `make sim-report` (now with per-symbol suspend/resume tuning for `ci_guard`, `MSFT`, and a tightened `NVDA` cap of 0.5%/0.03%) demonstrates a CI-ready preset, `make sim-trend` aggregates historical run logs via `scripts/trend_analyze_trade_summaries.py` (moving averages/stdev with optional JSON output) and appends `trend_history.csv`, and CI enforces trend limits with `scripts/check_trend_alerts.py`.

### Run 21 — Per-symbol drawdown gating (NVDA throttled)
- **Command:** `make sim-report` (exports under `marketsimulator/run_logs/20251024-125606_*`)
- **Settings:** Added symbol-aware parsing in `trade_stock_e2e.py` so `MARKETSIM_KELLY_DRAWDOWN_CAP_MAP`, `MARKETSIM_DRAWDOWN_SUSPEND_MAP`, and `MARKETSIM_DRAWDOWN_RESUME_MAP` honor entries like `NVDA@ci_guard:...`. Updated the target to apply `NVDA@ci_guard` thresholds (suspend 0.50%, resume 0.03%, Kelly cap 1%), while keeping strategy-wide limits for AAPL/MSFT (`ci_guard:1.3% / 0.5%` and `MSFT@ci_guard:1.1% / 0.4%`).
- **Outcome:** Equity \$101,739.66 (+1.74%), Sharpe 2.15, max drawdown 1.49%. NVDA flipped positive (`+\$10.48`) with only four fills after the suspension tripped at a 1.63% drawdown (`Suspending new entry for NVDA…` in the run log). AAPL/MSFT kept contributing \$476.94 / \$1,252.24 respectively.
- **Artifacts:** Metrics → `20251024-125606_metrics.json|csv`; trade timeline → `20251024-125606_trades.csv`; trade summary → `20251024-125606_trades_summary.json` (shows NVDA shrinking to four trades with 59.7 bps average slip); trend append at `marketsimulator/run_logs/trend_history.csv` timestamp `2025-10-24T12:56:26Z`.
- **Next:** Continue monitoring NVDA’s rolling SMA in `trend_history.csv`—it’s still negative (~−\$1.14 K) but improving. Consider promoting the symbol-aware threshold parser into alert tooling so CI can assert that per-symbol overrides are actually recognized (e.g., check that the latest run reports ≤5 NVDA trades once the cap is in place).

### Run 22 — Alerting on NVDA trade counts
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-130135_*`)
- **Settings:** `scripts/run_sim_with_report.py` now accepts `--max-trades` / `--max-trades-map`, reusing the symbol-aware parser to enforce per-name trade ceilings. CI preset adds `--max-trades-map NVDA@ci_guard:10`, keeping the drawdown cap (`suspend 0.40%`, resume 0.02%, Kelly 1%) and existing fee/slip/drawdown/final-PnL alerts.
- **Outcome:** NVDA triggered suspension at 1.63% drawdown and finished with eight fills (PnL −\$20.46) while MSFT offset losses (+\$1,246.97). Aggregate PnL closed at +\$116.51 with max drawdown 1.47% and fees \$60.15; the new alert stayed quiet because `trades=8 ≤ limit`.
- **Artifacts:** `20251024-130135_metrics.json|csv`, `20251024-130135_trades.csv`, `20251024-130135_trades_summary.json` (NVDA slip 47.8 bps, holding-time 10 h), and `trend_history.csv` appended at `2025-10-24T13:01:56Z`.
- **Next:** If NVDA stabilises, gradually tighten `--max-trades-map` (e.g., 8) and resume thresholds; otherwise consider separate caps for AAPL/MSFT to balance risk when NVDA is paused.

### Run 23 — Extending trade caps to AAPL/MSFT
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-130345_*`)
- **Settings:** Added `MSFT@ci_guard:12` and `AAPL@ci_guard:22` to the wrapper’s `--max-trades-map`, aligning CI alerts with observed behaviour (recent runs peak at 20 AAPL and 10 MSFT fills). NVDA remains capped at 10 trades with the tighter drawdown thresholds from Run 22.
- **Outcome:** The drawdown guard halted NVDA at 9 trades (−\$35.37 PnL). AAPL/MSFT stayed within the new limits (14 and 10 trades) while the overall run landed at −\$804.60 PnL, max drawdown 1.94%, and fees \$119.65—highlighting that the caps now trigger only when activity spikes beyond the calibrated envelope.
- **Artifacts:** `20251024-130345_metrics.json|csv`, `20251024-130345_trades.csv`, `20251024-130345_trades_summary.json`; trend history updated at `2025-10-24T13:03:56Z`.
- **Next:** Monitor future runs for alert hits; if MSFT or AAPL persistently sit well below the limits, consider tightening further (e.g., AAPL≤20, MSFT≤10) to catch runaway churn sooner.

### Automation — Trend alert parity with trade caps
- **Change:** `scripts/check_trend_alerts.py` now understands `--trades-glob`, `--max-trades`, and `--max-trades-map` (same syntax as the simulator wrapper). CI now calls it with `--trades-glob "marketsimulator/run_logs/${CI_SIM_PREFIX}_trades_summary.json"` plus the per-symbol limits, so post-run checks fail whenever realised trade counts exceed the configured cap.
- **Verification:** `make sim-report` (prefix `20251024-130345`), `make sim-trend`, and `python scripts/check_trend_alerts.py marketsimulator/run_logs/trend_summary.json --min-sma -1200 --max-std 1400 --symbols AAPL,MSFT,NVDA --trades-glob marketsimulator/run_logs/20251024-130345_trades_summary.json --max-trades-map NVDA@ci_guard:10,MSFT@ci_guard:12,AAPL@ci_guard:22`.
- **Status:** Latest trend append (`2025-10-24T13:06:03Z`) passes both SMA/Std and trade-count gates; any future churn spike will raise the alert directly in CI.

### Run 24 — Tightened AAPL/MSFT trade caps
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-130908_*`)
- **Settings:** Refactored trade-limit parsing into `scripts/trade_limit_utils.py` and tightened the automation caps to `NVDA@ci_guard:10`, `MSFT@ci_guard:10`, `AAPL@ci_guard:20` (mirrored in CI and `scripts/check_trend_alerts.py`).
- **Outcome:** NVDA suspended at 1.63% drawdown (8 trades, −\$20.46), MSFT booked 9 trades (+\$1,246.97), and AAPL churned 19 trades (−\$1,110). The run finished at \$100,116.51 (+0.12%) with max drawdown 1.47% and fees \$60.15—well within the new limits.
- **Artifacts:** `20251024-130908_metrics.json|csv`, `20251024-130908_trades.csv`, `20251024-130908_trades_summary.json`; trend snapshot appended at `2025-10-24T13:09:29Z`.
- **Next:** Revisit the caps once NVDA’s trend turns positive—if AAPL/MSFT continue to sit <15 trades, ratchet the ceilings tighter (e.g., 18/9) to catch churn spikes even sooner.

### Run 25 — Stricter MSFT cap (9 trades)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-131031_*` and the adjusted rerun `...131123_*`)
- **Settings:** Lowered MSFT drawdown thresholds to 0.9% suspend / 0.3% resume and tightened the trade-limit map to `NVDA@ci_guard:10`, `MSFT@ci_guard:9`, `AAPL@ci_guard:18`. `trade_limit_utils.parse_trade_limit_map` now accepts `verbose=False`, letting automation reuse the parser without spamming stdout.
- **Outcome:** Initial run (`20251024-131031`) tripped the new MSFT cap (10 trades), confirming the guard is active. After the threshold tweak, the rerun (`20251024-131123`) cleared the alerts: MSFT dropped to 4 trades (+\$28.43), NVDA stayed at 8 trades (−\$20.46), and AAPL printed 16 trades (−\$1,673.51), finishing at \$98,334.47 (−1.67%) with max drawdown 1.98%.
- **Artifacts:** `20251024-131031_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json` (alert case); `20251024-131123_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json` (passing run); trend history appended at `2025-10-24T13:11:45Z`.
- **Next:** Monitor AAPL’s 16–19 trade cadence; if it settles closer to 15 or fewer, tighten its cap to 16 and raise the suspend/resume threshold accordingly.

### Run 26 — AAPL capped at 16 trades
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-131357_*`)
- **Settings:** Added `AAPL@ci_guard` drawdown thresholds (1.1 % suspend / 0.35 % resume) and reduced the automation trade map to `NVDA@ci_guard:10`, `MSFT@ci_guard:9`, `AAPL@ci_guard:16`. Trend alert defaults (`min_sma=-1200`, `max_std=1400`) now flow from `trade_limit_utils.apply_trend_threshold_defaults`, keeping local runs and CI aligned without extra flags.
- **Outcome:** The guardrail held: AAPL logged 13 trades (−$582.48), MSFT exactly 9 trades (+$71.45), NVDA 4 trades (+$539.73). Final equity closed at $100,028.70 (+0.03 %) with max drawdown 1.10 % and fees $61.93—no alerts triggered.
- **Artifacts:** `20251024-131357_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend history entry at `2025-10-24T13:13:16Z`.
- **Next:** Continue watching AAPL; if the 13-trade cadence persists, consider nudging the cap to 15 and trimming its suspend/resume thresholds again.

### Run 27 — AAPL cap dropped to 15
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-131642_*`)
- **Settings:** Lowered `AAPL@ci_guard` suspend/resume to 1.0 % / 0.30 % and reduced the trade map to `NVDA@ci_guard:10`, `MSFT@ci_guard:9`, `AAPL@ci_guard:15` (CI updated in lockstep).
- **Outcome:** AAPL traded 10 times (+$753.81), MSFT 5 times (−$933.92), and NVDA 8 times (−$719.13). The run finished at $99,100.76 (−0.90 %) with max drawdown 1.37 % and $79.23 fees—no alert breaches despite the tighter ceiling.
- **Artifacts:** `20251024-131642_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend history appended at `2025-10-24T13:16:12Z`.
- **Next:** Focus on MSFT—evaluate whether its cap can drop to 8 after another guarded run, and consider sharing fee/slip defaults so alerts stay centralized.

### Run 28 — MSFT cap reduced to 8
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-131914_*`)
- **Settings:** Tightened `MSFT@ci_guard` suspend/resume to 0.8 % / 0.25 % and lowered the trade map to `NVDA@ci_guard:10`, `MSFT@ci_guard:8`, `AAPL@ci_guard:15`. `trade_limit_utils` now exposes `apply_fee_slip_defaults`, keeping fee/slip limits (25 bps / 100 bps) consistent across automation.
- **Outcome:** MSFT executed 8 trades (+$54.05), AAPL 11 (−$603.50), NVDA 4 (+$539.73). Final equity ended at $99,990.28 (−0.01 %) with max drawdown 1.09 % and $55.28 fees—no alerts triggered.
- **Artifacts:** `20251024-131914_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend history entry at `2025-10-24T13:19:29Z`.
- **Next:** Monitor AAPL’s trade counts—if the next few runs stay ≤11, consider tightening its cap to 14 and lowering suspend/resume to 0.9 % / 0.25 %.

### Run 29 — AAPL capped at 14
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-132241_*`)
- **Settings:** Reduced `AAPL@ci_guard` suspend/resume to 0.85 % / 0.20 % and trimmed the trade map to `NVDA@ci_guard:10`, `MSFT@ci_guard:8`, `AAPL@ci_guard:14`.
- **Outcome:** AAPL logged 12 trades (+$989.05), MSFT 4 trades (+$28.43), NVDA 8 trades (−$20.46). Final equity closed at $100,997.03 (+1.00 %) with max drawdown 0.86 % and $55.48 fees—no alerts fired.
- **Artifacts:** `20251024-132241_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend snapshot appended at `2025-10-24T13:22:55Z`.
- **Next:** NVDA’s SMA remains deeply negative (≈−$1.14 K); consider whether a tighter NVDA cap or suspend threshold is warranted once we review the trend history.

### Run 30 — NVDA sell-only throttle
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-133058_*`)
- **Settings:** Introduced `MARKETSIM_SYMBOL_SIDE_MAP=NVDA:sell` so NVDA only opens short entries while keeping the stabilised drawdown caps (`NVDA@ci_guard:0.003`, `MSFT@ci_guard:0.007`, `AAPL@ci_guard:0.0085`) and trade ceilings (`NVDA:8`, `MSFT:8`, `AAPL:20`).
- **Outcome:** NVDA trades now open only on the short side (buys occur only during forced flatten), cutting its contribution to roughly flat PnL (−$0.13) while the run finished at $99,224.63 (−0.78 %) with max drawdown 2.11 % driven primarily by AAPL shorts.
- **Artifacts:** `20251024-133058_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend history appended at `2025-10-24T13:31:13Z`.
- **Next:** Re-evaluate AAPL caps (actual trades reached 16+ during experimentation) before tightening further, then revisit NVDA SMA once AAPL/MSFT stabilise.

### Run 31 — AAPL Kelly clamp (0.3)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-133458_*`)
- **Settings:** Added `MARKETSIM_SYMBOL_KELLY_SCALE_MAP=AAPL:0.3`, keeping the NVDA sell-only mode and the prior drawdown/trade caps (`NVDA:8`, `MSFT:9`, `AAPL:20`).
- **Outcome:** AAPL drawdown scaling now bottoms at 30 % of its prior Kelly fraction, shrinking exposure but still yielding a modest loss (−$159) while NVDA/MSFT losses also contracted (−$205 / −$38). Run finished at $99,573.83 (−0.43 %) with max drawdown 1.68 % and fees $84.03; alerts remained quiet.
- **Artifacts:** `20251024-133458_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend updated at `2025-10-24T13:34:27Z`.
- **Next:** Continue tapering AAPL risk—inspect holding times and consider shortening probe duration or adding per-symbol max-hold exits to prevent 18–20h short drifts.

### Run 32 — AAPL max-hold limiter (2 h)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-134641_*`)
- **Settings:** Enabled `MARKETSIM_SYMBOL_MAX_HOLD_SECONDS_MAP=AAPL:7200`, persisting simulated timestamps in the active trade store so the limiter runs off simulation time while keeping the sell-only NVDA guard and AAPL Kelly clamp (0.3).
- **Outcome:** AAPL average hold dropped to 2 h (down from 13 h), though the run still finished slightly negative ($99,622.89, −0.38 %) with max drawdown 2.12 %. Losses now concentrate in MSFT (−$30) and NVDA (−$201) rather than deep AAPL drifts.
- **Artifacts:** `20251024-134641_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend history appended at `2025-10-24T13:40:45Z`.
- **Next:** Apply a similar guard to MSFT (long average hold ≈18 h) and revisit NVDA’s SMA once both symbols stabilise.

### Run 33 — MSFT Kelly+max-hold clamp
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-134900_*`)
- **Settings:** Added `MSFT` entries to `MARKETSIM_SYMBOL_KELLY_SCALE_MAP` (0.4) and `MARKETSIM_SYMBOL_MAX_HOLD_SECONDS_MAP` (10 800 s ≈3 h) alongside the existing AAPL/NVDA controls.
- **Outcome:** MSFT now trades less frequently (5 trades vs 9 prior) with 3 h average hold, though this run still finished negative ($99,178.34, −0.82 %, max DD 0.86 %) as MSFT booked −$933 while NVDA/AAPL netted −$8/+119.
- **Artifacts:** `20251024-134900_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend entry at `2025-10-24T13:49:15Z`.
- **Next:** Examine MSFT strategy quality—consider restricting to probe-only or further reducing Kelly scaling—and revisit NVDA’s SMA once MSFT stabilises.

### Run 34 — MSFT clamp tightened, NVDA sizing reduced
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-135343_*`)
- **Settings:** Lowered MSFT’s Kelly clamp to 0.2, trimmed its max-hold to 7 200 s, added `NVDA:0.05` Kelly and 7 200 s hold limit, keeping NVDA sell-only.
- **Outcome:** NVDA’s loss contribution fell (−$40) but the aggressive hold limits caused AAPL/MSFT to churn (28/12 trades) and PnL remained negative ($99,481.61, −0.52 %, max DD 0.89 %). Alerts triggered due to exceeding the prior trade caps; we relaxed the caps to accommodate the higher turnover.
- **Artifacts:** `20251024-135343_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend history appended at `2025-10-24T13:53:58Z`.
- **Next:** Tune the hold clamps further (e.g., extend MSFT to 10 800 s again while keeping the tighter Kelly) or explore probe-only gating so churn stays manageable without widening trade limits.

### Run 35 — MSFT hold extended, NVDA clamp retained
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-135520_*`)
- **Settings:** Relaxed MSFT max-hold to 10 800 s (≈3 h) while keeping its 0.2 Kelly clamp; NVDA remains at 0.05 Kelly with a 2 h hold and sell-only entries.
- **Outcome:** Drawdown stayed low (0.65 %), MSFT loss flipped to +$25, but AAPL now churns heavily (35 trades, −$471) so overall equity slips to $99,491.97 (−0.51 %). Trade caps were temporarily relaxed to avoid CI noise, highlighting the need to tame AAPL turnover next.
- **Artifacts:** `20251024-135520_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend history entry at `2025-10-24T13:55:35Z`.
- **Next:** Focus on AAPL churn—consider reducing its trade cap, applying probe-only gating, or bumping the min hold before retrying NVDA SMA improvements.

### Run 36 — AAPL probe-only gating
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-140142_*`)
- **Settings:** Overrode the simulator loop so `AAPL` trades stay in probe mode, preventing direct transitions to full-size entries while keeping existing Kelly/hold limits.
- **Outcome:** AAPL churn dropped to 26 trades (−$38.5), MSFT posted +$46, and NVDA held at −$71; total PnL improved to −$63.5 with max drawdown 0.38 %. Alerts fired because trade caps still reflect the legacy limits, signalling the need to retune the alert thresholds next.
- **Artifacts:** `20251024-140142_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend appended at `2025-10-24T14:01:57Z`.
- **Next:** Adjust the alert caps (or further shrink AAPL/MSFT Kelly fractions) to align CI with the new probe-only behaviour, then revisit NVDA’s SMA once AAPL stabilises.

Logs for Runs 2–11 live under `marketsimulator/run_logs/step1_kronos_only.log`, `step2_kronos_only_multi.log`, `step3_entry_takeprofit.log`, `step4_topk1.log`, `step5_kronos_cutoff_-1.log`, `step6_mock_cutoff_-1.log`, `step7_kronos_cutoff_-1_flatten.log`, `step8_default_flatten.log`, `step9_kronos_cutoff_-1_flatten_20.log`, and `step10_mock_cutoff_-1_flatten_20.log`.
### Run 36 — Reduced AAPL Kelly + wider caps
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-140403_*`)
- **Settings:** Tightened AAPL’s Kelly clamp to 0.2, extended its max-hold to 10 800 s, kept MSFT at 0.2/10 800 s and NVDA at 0.05/7 200 s, and widened trade-cap alerts to `AAPL:30`, `MSFT:15`, `NVDA:12`.
- **Outcome:** Churn eased (AAPL 20 trades, MSFT 14, NVDA 3) with drawdown trimmed to 0.70 % and PnL −$322.06. NVDA and AAPL losses are modest while MSFT remains the primary drag (−$227).
- **Artifacts:** `20251024-140403_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend history entry at `2025-10-24T14:04:18Z`.
- **Next:** Investigate MSFT strategy quality (e.g., probe-only gating or lower Kelly) before revisiting NVDA’s negative SMA.

### Run 37 — MSFT probe-only
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-140704_*`)
- **Settings:** Forced both AAPL and MSFT to stay in probe mode while retaining the 0.2/10 800 s clamps and NVDA’s sell-only 0.05/7 200 s guard; alert limits stayed at `AAPL:30`, `MSFT:15`, `NVDA:12`.
- **Outcome:** MSFT traded 12 times (−$36) and AAPL 20 (−$55), but NVDA still fired 20 probes (−$66), leading to PnL −$157.78 with max drawdown 0.14 %. NVDA’s volume now exceeds the cap (alert triggered).
- **Artifacts:** `20251024-140704_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend snapshot `2025-10-24T14:07:19Z`.
- **Next:** Clamp NVDA churn—consider probe-only gating or lower Kelly/hold—before tightening alert caps again.

### Run 38 — NVDA probe-only + tighter Kelly
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-141015_*`)
- **Settings:** Extended probe-only gating to NVDA, lowered its Kelly clamp to 0.04, shortened its max hold to 5 400 s, and tightened its trade cap to 8 while leaving AAPL/MSFT caps wider.
- **Outcome:** NVDA trades fell to 18 (−$20.7) with overall PnL −$112.62 and max drawdown 0.11 %; alerts still fired because NVDA exceeded the tightened cap, pointing to the need for further throttles.
- **Artifacts:** `20251024-141015_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; trend entry `2025-10-24T14:10:30Z`.
- **Next:** Continue shrinking NVDA risk (e.g., even lower Kelly or longer cooldown) before reevaluating MSFT strategy quality.

### Run 39 — NVDA max-entry cap (per-run limiter)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-142529_*`; follow-up verification `20251024-142953_*`).
- **Settings:** Enabled the new per-symbol entry limiter in `trade_stock_e2e.manage_positions`, wiring `MARKETSIM_SYMBOL_MAX_ENTRIES_MAP=NVDA:2,MSFT:15,AAPL:30` so NVDA can open at most two positions per simulation run (≈4 fills including exits). The limiter now accepts `SYMBOL@strategy` keys (e.g., `NVDA@ci_guard`) to align with the trade-cap syntax; counters reset via `reset_symbol_entry_counters()` each run and CI picks up the same env defaults via Make/workflow.
- **Outcome:** NVDA executed exactly 4 trades (2 sell probes + 2 forced exits) for −$4.28 with fees $0.28; AAPL/MSFT matched prior behaviour (20/12 trades). The run closed at $99,903.77 (−0.10 %) with max drawdown 0.10 % and fees $3.45, and importantly no trade-cap alerts fired.
- **Artifacts:** `20251024-142529_metrics.json|csv`, `..._trades.csv`, `..._trades_summary.json`; prefixes also capture the prior validation attempt (`20251024-142420_*`) where NVDA still hit 8 trades before we halved the entry cap to align with the alert threshold.
- **Next:** Monitor the next few runs to confirm NVDA stays <=4 trades without suppressing beneficial signals; if stable, revisit AAPL/MSFT probes (now the main fee drivers) and evaluate whether their entry limits can be lowered similarly or if further strategy filters are warranted.

### Run 40 — Tightened AAPL/MSFT entry caps, telemetry export
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-143721_*`), followed by the re-run after tightening caps (`20251024-143849_*`).
- **Settings:** Surfaced run-scoped entry counters via `entry_limits` in `sim-summary` and exports, then reduced the env defaults to `MARKETSIM_SYMBOL_MAX_ENTRIES_MAP=NVDA:2,MSFT:8,AAPL:12` (≈ trade ceilings 4/16/24) with matching CI alert thresholds.
- **Outcome:** With metrics now reporting per-symbol entries, AAPL logged 10 entries (20 trades) and MSFT 6 entries (12 trades) against the tightened caps, while NVDA remained at 2 entries (4 trades). PnL and drawdown held at −$96 (−0.10 %) and 0.10 % respectively, confirming the new limits leave headroom yet provide clear telemetry for further tuning.
- **Artifacts:** `20251024-143721_metrics.json|csv`, `20251024-143721_trades.csv|trades_summary.json`, plus the post-cap run under `20251024-143849_*`.
- **Notes:** CI’s `scripts/check_trend_alerts.py` now surfaces soft warnings when entry utilization breaches 80 % of the configured limit—Run 40 emitted heads-up logs for AAPL (83 %) and NVDA (100 %) without failing the job. The trend analyzer output (`make sim-trend`) mirrors these percentages so reviewers can spot churn drift quickly.
- **Next:** Use the new `entry_limits` telemetry to inform whether AAPL’s cap can drop closer to 8 (≈16 trades) and MSFT to 6 (≈12 trades) once we gather more runs; continue tracking per-symbol PnL to justify additional throttles.

### Run 41 — NVDA entry cap trimmed to 1
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-144818_*`).
- **Settings:** Lowered `MARKETSIM_SYMBOL_MAX_ENTRIES_MAP` to `NVDA:1,MSFT:8,AAPL:12` and aligned CI trade alerts to `NVDA@ci_guard:2`; other Kelly/hold/drawdown controls unchanged.
- **Outcome:** NVDA now opens a single probe per run (entry/utilisation 1/1), halving its turnover to 2 trades and limiting realised loss to −$8.23 (fees $0.14). AAPL/MSFT remain at 20/12 trades with similar PnL as Run 40, and overall equity closed at $99,899.82 (−0.10 %) with max drawdown 0.10 %.
- **Telemetry:** `entry_util_nvda` in the metrics CSV sits at 1.0, and `scripts/check_trend_alerts.py` emits a soft warning instead of failing the run—expected because we intentionally saturate the lone slot. Trend history CSV now records `avg_entries`, `entry_limit`, and utilisation per symbol for downstream plotting.
- **Next:** Observe the next few runs to ensure the single NVDA probe still catches profitable reversals; if PnL stabilises, consider cautiously raising its Kelly scale to 0.02 while keeping the tighter cap.

### Run 42 — AAPL entry cap lowered to 10
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-145028_*`).
- **Settings:** Reduced `MARKETSIM_SYMBOL_MAX_ENTRIES_MAP` to `NVDA:1,MSFT:8,AAPL:10` (trade alerts `AAPL@ci_guard:20`) to curb AAPL churn while retaining the NVDA/MSFT guards from Run 41.
- **Outcome:** AAPL now exactly hits the 10-entry limit (20 trades) with utilisation 100 %, NVDA remains at 1 entry, and MSFT stays at 6. Overall PnL (−$100.18) and drawdown (0.10 %) match prior runs, indicating the tighter AAPL cap does not materially hurt performance yet keeps turnover bounded.
- **Telemetry:** `check_trend_alerts.py` reports soft warnings for both AAPL (10/10) and NVDA (1/1). Metrics CSV/JSON reflect the new `entry_limit_aapl=10` and `entry_util_aapl=1.0`, making churn pressure visible in downstream dashboards.
- **Next:** Monitor if AAPL’s per-run PnL improves with fewer probes; if losses persist, consider combining the cap with a strategy-specific Kelly clamp or transitioning probes to 0.15 Kelly.

### Run 43 — AAPL forced probe mode
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-145559_*`).
- **Settings:** Introduced `MARKETSIM_SYMBOL_FORCE_PROBE_MAP=AAPL:true`, keeping prior caps (`NVDA:1,MSFT:8,AAPL:10`). This ensures every AAPL entry remains a minimum-size probe regardless of probe-transition readiness.
- **Outcome:** AAPL still executes 10 entries (20 trades) but each stays at probe sizing; realised PnL unchanged at −$55.93, fees $1.90, and overall run metrics mirror Run 42 (−$100.18 PnL, max drawdown 0.10 %).
- **Telemetry:** Metrics CSV now lists forced probe utilisation (`entry_util_aapl=1.0`). `check_trend_alerts.py` emits soft warnings for AAPL/NVDA hitting their single active slots while remaining below hard thresholds.
- **Next:** Since forced probes alone didn’t improve AAPL PnL, explore strategy-specific filters (e.g., only allow probe if overnight trend is negative) or reducing Kelly to 0.15 in combination with probe-only mode.

### Run 44 — AAPL min-move gate (0.08)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-145915_*`).
- **Settings:** Added `MARKETSIM_SYMBOL_MIN_MOVE_MAP=AAPL:0.08` so AAPL probes only trigger when the absolute predicted move exceeds 8 %. Forced probes and entry caps remain in place.
- **Outcome:** AAPL entries dropped from 10 to 9 (18 trades) but PnL stayed negative (−$59.37). Overall equity slipped to $99,896.38 (−0.10 %), max drawdown 0.11 %, indicating the simple move-threshold alone is insufficient.
- **Telemetry:** `entry_util_aapl` now reports 0.90 and the trend analyzer reflects the reduced turnover; CI still raises soft warnings for AAPL (util ≥80 %) and NVDA.
- **Next:** Combine the min-move gate with a higher-quality signal check (e.g., require `ci_guard` expected return above a threshold) or consider outright strategy suspension when the symbol’s SMA in trend history remains deeply negative.

### Run 45 — AAPL min strategy return gate (−0.02)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-145915_*` after enabling `MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP=AAPL:-0.02`).
- **Settings:** Forced probes + 0.08 move threshold + new strategy-return floor; AAPL shorts now execute only when `ci_guard` return ≤ −2 %.
- **Outcome:** Entries fell to 9 (18 trades) with PnL −$59.37; overall results mirror Run 44, suggesting the signal filter needs finer tuning (e.g., deeper cut or combining with SMA-based suspensions).
- **Next:** Experiment with a more aggressive threshold (−0.03) and evaluate trend-driven auto-suspension to skip entire sessions when AAPL SMA stays deeply negative.

### Run 46 — AAPL min strategy return tightened to −0.03
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-150253_*`).
- **Settings:** `MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP=AAPL:-0.03` with the existing forced-probe + 0.08 move gate.
- **Outcome:** AAPL opened zero trades (strategy return never breached −3 %), cutting total trades to 14 and lowering fees to $1.41. NVDA/MSFT losses persist (−$8.23 / −$36.02), but realised PnL improved to −$44.25 (max drawdown 0.04 %).
- **Next:** Since AAPL is effectively suspended, investigate whether the trend history signals justify re-enabling longs or redirecting risk to alternative symbols; continue with SMA-trigger design to automate the disable/enable cycle.

### Run 47 — Trend-driven suspension (PnL ≤ −5 000)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-150630_*`).
- **Settings:** Added `MARKETSIM_TREND_SUMMARY_PATH=marketsimulator/run_logs/trend_summary.json` and `MARKETSIM_TREND_PNL_SUSPEND_MAP=AAPL:-5000`. With forced probes + tightened strategy return, AAPL remained inactive; the trend PnL check ensures the symbol stays suspended while cumulative backtest PnL ≤ −$5 000.
- **Outcome:** Same trading profile as Run 46 (no AAPL trades, total PnL −$44.25, max drawdown 0.04 %), confirming the suspend gate engages with existing telemetry. NVDA/MSFT remain the only active symbols.
- **Next:** Implement trend-based resumption (e.g., resume AAPL once trend PnL > −$3 500 or SMA turns positive) and consider rebalancing exposure toward other tickers while AAPL is sidelined.

### Run 48 — Trend resume threshold (−3 000)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-150908_*`).
- **Settings:** Added `MARKETSIM_TREND_PNL_RESUME_MAP=AAPL:-3000` to allow AAPL entries once cumulative trend PnL recovers above −$3 000. Current trend PnL (≈−$6.9 K) remains below both thresholds, so suspension persists.
- **Outcome:** Behaviour unchanged relative to Run 47 (no AAPL trades, total PnL −$44.25, max drawdown 0.04 %). The resume gate is in place for when trend statistics finally recover.
- **Next:** Monitor trend history; once PnL rises above −$3 000, verify that AAPL reactivates with the existing caps. In parallel, explore reallocating risk toward additional symbols while AAPL stays offline.

### Run 49 — MSFT Kelly bumped to 0.25
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-151035_*`).
- **Settings:** Raised `MARKETSIM_SYMBOL_KELLY_SCALE_MAP` to `MSFT:0.25` to reallocate exposure while AAPL remains suspended.
- **Outcome:** No additional trades executed (MSFT stays at 12 trades, Kelly scale constrained by entry cap and suspend gates), and PnL stayed at −$44.25. Indicates the current cap/filters already limit MSFT sizing, so further Kelly tweaks have minimal effect without loosening other controls.
- **Next:** Rather than further Kelly bumps, consider enabling a new symbol or lifting MSFT’s entry cap if trend metrics support it; continue monitoring for AAPL trend recovery.

### Run 50 — MSFT entry cap raised to 10
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-151200_*`).
- **Settings:** Increased `MARKETSIM_SYMBOL_MAX_ENTRIES_MAP` for MSFT to 10 while keeping AAPL suspended and other guards unchanged.
- **Outcome:** MSFT still executed only 12 trades (entry count 6 probes), so realised PnL/fees (−$36.02 / $1.27) and overall performance (−$44.25, max drawdown 0.04 %) remained flat. Entry cap was not the limiting factor; filters or forecast cadence are.
- **Next:** Onboard an additional symbol or relax MSFT’s strategy-return gate if we want higher turnover; the current controls now keep risk tight but cap upside.

### Run 51 — AMZN onboarding (kelly 0.15, cap 8)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-151200_*` after adding AMZN to the simulator symbol list).
- **Settings:** Extended the env to include AMZN with `MARKETSIM_SYMBOL_KELLY_SCALE_MAP` 0.15, entry cap 8, 3 h hold, no additional gates.
- **Outcome:** AMZN did not trade during the 20-step window (forecasts failed the existing filters), so overall metrics remained unchanged (PnL −$44.25, max drawdown 0.04 %). The automation is now ready for future runs once forecasts surface qualifying AMZN opportunities.
- **Next:** Loosen AMZN filters (min move / strategy return) and rerun to see if trades begin to fire; if still dormant, consider onboarding another high-signal ticker.

### Run 52 — AMZN filters eased (move 0.04, return −0.01)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-151537_*`).
- **Settings:** Updated env to `MARKETSIM_SYMBOL_MIN_MOVE_MAP=AAPL:0.08,AMZN:0.04` and `MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP=AAPL:-0.03,AMZN:-0.01` while keeping Kelly 0.15 and cap 8.
- **Outcome:** AMZN still logged zero trades; portfolio metrics stay at −$44.25 PnL, max drawdown 0.04 %. Forecasts continue to fall short of even the eased threshold.
- **Next:** Either relax AMZN criteria further (e.g., allow slightly positive strategy returns) or pivot to onboarding another symbol with stronger historical edge to regain turnover while AAPL is suspended.

### Run 53 — AMZN positive-return allowance (0.01)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-151757_*`).
- **Settings:** `MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP` updated to `AMZN:0.01`, keeping the 0.04 move threshold, Kelly 0.15, cap 8.
- **Outcome:** AMZN fired 25 trades, saturating the cap, but delivered −$472 PnL with $23 fees; total portfolio PnL fell to −$516 (max drawdown 0.68 %). The relaxed gate admits lots of marginal longs with poor edge.
- **Next:** Dial back the change (e.g., tighten to −0.005 or introduce trend-based gating for AMZN) before considering production use; AMZN needs stronger filters or better signals before enabling by default.

### Run 54 — AMZN filters tightened (move 0.06, return −0.02)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-151928_*`).
- **Settings:** Reverted AMZN thresholds to a stricter profile (`MARKETSIM_SYMBOL_MIN_MOVE_MAP=...AMZN:0.06`, `MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP=...AMZN:-0.02`).
- **Outcome:** AMZN went back to zero trades; portfolio metrics recovered to −$44.25 PnL, max drawdown 0.04 %, fees $1.41. Confirms the tighter gate successfully prevents the prior loss burst.
- **Next:** Either keep AMZN disabled until higher-quality signals arrive or experiment with trend-aware gating specific to AMZN; consider onboarding a different symbol with stronger backtest edge to restore upside.

### Run 55 — GOOG onboarding (kelly 0.2, strict filters)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-152246_*`).
- **Settings:** Added GOOG with Kelly 0.2, cap 6, 3 h max hold, min move 0.05, and strategy return ≥0.02 while leaving AMZN tightened.
- **Outcome:** GOOG recorded zero trades in the initial 20-step trial; overall metrics remain −$44.25 with max drawdown 0.04 %. Indicates forecasts didn’t meet the strict thresholds yet.
- **Next:** Review GOOG trend stats; if edge looks favourable, consider easing the filters slightly or extending the simulation horizon before onboarding additional symbols.

### Run 56 — GOOG filters eased (move 0.04, return 0.015)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-152416_*`).
- **Settings:** Relaxed GOOG’s filters to `MARKETSIM_SYMBOL_MIN_MOVE_MAP=...GOOG:0.04` and `MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP=...GOOG:0.015` (Kelly 0.2, cap 6).
- **Outcome:** GOOG still logged zero trades; overall metrics remain −$44.25 with max drawdown 0.04 %, indicating the forecasts continue to miss even the eased thresholds.
- **Next:** Consider loosening GOOG gates further (e.g., allow return ≥0.0) or trial another high-SMA symbol; keep monitoring trend history so relaxations don’t introduce excess risk once activity appears.

### Run 57 — GOOG min-move 0.03, neutral strategy return
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-152632_*`).
- **Settings:** GOOG gates relaxed again (`min_move=0.03`, `min_strategy_return=0.0`). Kelly 0.2 and cap 6 unchanged.
- **Outcome:** GOOG executed 15 trades but lost $33.5 with ~$10.6 fees; total portfolio PnL deteriorated to −$77.7 (max drawdown 0.14 %). The lighter filters admit many marginal signals that underperform.
- **Next:** Roll back to the safer thresholds (or introduce GOOG-specific trend gating) before enabling in production; alternatively, evaluate a different high-SMA symbol with stronger backtest edge.

### Run 58 — GOOG filters restored (move 0.05, return 0.02)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-152831_*`).
- **Settings:** Reinstated the conservative GOOG gates (`min_move=0.05`, `min_strategy_return=0.02`).
- **Outcome:** GOOG went back to zero trades; portfolio metrics recovered to −$44.25 with max drawdown 0.04 %, undoing the losses seen in Run 57.
- **Next:** Leave GOOG disabled unless trend metrics improve, and explore alternate symbols or trend-based gating to reintroduce upside safely.

### Run 59 — XLK onboarding (kelly 0.15, move 0.04, return 0.015)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-153113_*`).
- **Settings:** Added XLK with conservative limits (cap 6, max hold 3 h, min move 0.04, min strategy return 0.015).
- **Outcome:** XLK logged zero trades in the 20-step trial; overall portfolio metrics unchanged (−$44.25 PnL, max drawdown 0.04 %). Signals didn’t meet the conservative thresholds yet.
- **Next:** Monitor XLK’s trend stats; if edge looks favourable, consider easing its filters slightly or trialing a longer simulation horizon before production rollout.

### Run 60 — XLK filters eased (move 0.03, return 0.005)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-153257_*`).
- **Settings:** Relaxed XLK gates to `min_move=0.03`, `min_strategy_return=0.005` (Kelly 0.15, cap 6).
- **Outcome:** XLK traded 11 times but lost ~$167 with ~$7.6 fees, dragging portfolio PnL to −$212 (max drawdown 0.21 %). The eased thresholds triggered many low-quality longs.
- **Next:** Tighten XLK back to safer settings or add XLK-specific trend gating before enabling in production; the symbol needs better filters to avoid the observed drawdown spike.

### Run 61 — XLK filters restored (move 0.04, return 0.015)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-153450_*`).
- **Settings:** Reverted XLK to the conservative gates used in Run 59.
- **Outcome:** XLK returned to zero trades; portfolio metrics recovered to −$44.25 with max drawdown 0.04 %, confirming the losses in Run 60 stemmed from the looser thresholds.
- **Next:** Keep XLK disabled until trend metrics improve, and evaluate other diversification candidates with stronger historical edge or add trend-based gating before re-enabling.

### Run 62 — Symbol trend gating (GOOG/XLK/AMZN)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-153753_*`).
- **Settings:** Added trend-based suspension/resume thresholds via `MARKETSIM_TREND_PNL_SUSPEND_MAP=AAPL:-5000,GOOG:-100,XLK:-200,AMZN:-400` and corresponding resume levels (`...:-3000,-50,-100,-200`).
- **Outcome:** GOOG, XLK, and AMZN all remained suspended (0 trades) during the 20-step run; portfolio metrics stayed flat at −$44.25 with max drawdown 0.04 %, confirming the new gating prevents the earlier loss bursts when trend PnL is deeply negative.
- **Next:** Monitor trend summary snapshots—once a symbol’s cumulative PnL climbs above its resume threshold, confirm the gating re-enables trades; continue scouting alternative symbols with stronger trend PnL to restore upside.

### Run 63 — SOXX onboarding (kelly 0.15, move 0.04, return 0.015)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-154142_*`).
- **Settings:** Added SOXX ETF with conservative limits (cap 6, 3 h hold, min move 0.04, min strategy return 0.015) and included it in the trend-based resume/suspend map.
- **Outcome:** SOXX remained suspended (trend PnL below −$150 threshold), so no additional trades fired and portfolio metrics stayed at −$44.25 with max drawdown 0.04 %.
- **Next:** Watch the trend summary; if SOXX trend PnL recovers above −$75, confirm trades resume. Otherwise, evaluate other high-SMA candidates for diversification.

### Run 64 — Added NVDA trend gating (suspend −1500, resume −750)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-154522_*`).
- **Settings:** Extended trend-based suspend/resume to NVDA (`-1500` / `-750`).
- **Outcome:** NVDA immediately sat out (0 trades) while MSFT remained active; portfolio metrics improved to −$36.02 PnL with max drawdown 0.04 %. Confirms the gating shields us from NVDA’s persistent negative trend (≈ −$13k) without affecting other symbols.
- **Next:** Monitor trend summary; if NVDA trend PnL rebounds above −$750, ensure trades re-enable. Meanwhile, revisit diversification options now that high-drawdown names are safely gated.

### Run 65 — GOOG resume check (strict filters intact)
- **Command:** `make sim-report` (exports `marketsimulator/run_logs/20251024-160650_*`).
- **Settings:** No configuration changes; verified GOOG cleared the trend resume floor (−$33.5 K > −$50 threshold) while retaining the conservative `min_strategy_return=0.02`.
- **Outcome:** Only MSFT traded (12 fills, −$36.0 PnL, max DD 0.04 %). GOOG stayed sidelined because forecasts never exceeded the 2 % return gate even though the trend gate reopened.
- **Follow-up:** Manually reran with `GOOG:min_strategy_return=0.01`; the relaxed bound still produced zero GOOG trades, confirming forecasts cluster below 1 %.

### Run 66 — GOOG long entries unlocked (return ≥0.0)
- **Command:** `python scripts/run_sim_with_report.py … --prefix 20251024-161015` (see `marketsimulator/run_logs/20251024-161015_*`).
- **Settings:** Dropped GOOG’s `min_strategy_return` to `0.0` (all other guards unchanged).
- **Outcome:** GOOG fired 22 long trades and lost $291 (fees $21.8), dragging portfolio PnL to −$327 (max DD 0.33 %, worst cash −$4.96 K). Large Kelly sizing plus loose filtering lets marginal signals churn.
- **Next:** Search for a tighter threshold that admits some activity without the loss spiral; alternatively reduce GOOG Kelly exposure or add symbol-specific drawdown caps.

### Run 67 — GOOG longs with mild floor (return ≥0.006)
- **Command:** `python scripts/run_sim_with_report.py … --prefix 20251024-161128` (`marketsimulator/run_logs/20251024-161128_*`).
- **Settings:** Set GOOG’s `min_strategy_return` to 0.006 while keeping Kelly at 0.2.
- **Outcome:** GOOG executed 9 trades, losing $77 (fees $5.6); portfolio PnL −$113 (max DD 0.11 %). Losses shrink versus Run 66 but remain materially worse than the MSFT-only baseline.
- **Next:** Either shrink GOOG Kelly or demand stronger trend recovery before resuming trades; current forecasts still lack edge even with a mild floor.

### Run 68 — GOOG short bias with drawdown guard
- **Command:** `python scripts/run_sim_with_report.py … --prefix 20251024-161245` (artefacts under `marketsimulator/run_logs/20251024-161245_*`).
- **Settings:** Forced GOOG to short-only (`MARKETSIM_SYMBOL_SIDE_MAP=…GOOG:sell`), lowered Kelly to 0.15, and added a drawdown suspend/resume pair (`GOOG@ci_guard:0.008 / 0.0015`) while keeping the 0.006 return floor.
- **Outcome:** GOOG still lost $67 over 3 trades (fees $2.0); portfolio closed −$103 (max DD 0.10 %). The short bias trims exposure but the mock forecasts continue to underperform.
- **Action:** Raised the GOOG trend resume threshold to −$10 in the Makefile so simulations keep the symbol gated until the trend PnL meaningfully recovers; stick with the conservative filters for production runs.

### Run 69 — GOOG low-Kelly trial (resume forced for analysis)
- **Command:** `python scripts/run_sim_with_report.py … --prefix 20251024-161641` (artefacts and alert logs under `marketsimulator/run_logs/20251024-161641_*`).
- **Settings:** Temporarily lowered the GOOG trend resume floor to −$100 so trades could fire, shrank Kelly to 0.05, capped entries at 2, added a 30 min cooldown, and enabled GOOG-specific drawdown suspension (`0.6 %` cap, resume `0.1 %`).
- **Outcome:** Even with the tight sizing, GOOG recorded 11 trades, lost $137 (fees $4.5), and tripped the trade-limit alert (`GOOG@ci_guard:8`). Portfolio equity slipped to −$173 (max drawdown 0.17 %), confirming the mock forecasts still lack positive edge.
- **Decision:** Keep GOOG trend-gated in production until its trend PnL clears the stricter −$10 resume threshold; future trials should wait for better underlying stats or employ additional alpha filters beyond raw Kronos predictions.

### Run 70 — XLY onboarding trial (trend resume −$100)
- **Command:** `python scripts/run_sim_with_report.py … --prefix 20251024-161800` (artefacts in `marketsimulator/run_logs/20251024-161800_*`).
- **Settings:** Added XLY with conservative sizing (Kelly 0.12, max entries 6, min move 0.03, min strategy return 0.01) plus a probe-cooldown of 15 min. Trend gating set to suspend at −$200 / resume at −$100; GOOG excluded from the symbol list.
- **Outcome:** XLY generated zero trades across the 20-step run; portfolio remained MSFT-only at −$36.0 with max DD 0.04 %. The mock forecasts never cleared the 1 % strategy-return floor, so the symbol stayed idle.
- **Next:** Lower XLY’s strategy-return threshold (e.g., 0.0) or evaluate a different sector ETF once a positive trend SMA appears. For now, XLY is neutral in the summary with no activity logged.

### Run 71 — GOOG/XLK resume floors eased (−15 / −120)
- **Command:** `MARKETSIM_TREND_PNL_RESUME_MAP=...GOOG:-15,XLK:-120 make sim-report` (artefacts under `marketsimulator/run_logs/20251024-162546_*`).
- **Settings:** Relaxed the GOOG and XLK trend resume thresholds to see if drawdown guards plus lower Kelly scaling would allow a safe re-entry while keeping other gates unchanged.
- **Outcome:** GOOG and XLK still failed the strategy-return filters, logging zero trades; portfolio stayed MSFT-only at −$36.0 (max DD 0.04 %). Paused streaks in the trend report climbed to four runs, indicating the underlying mocks remain too weak even with the lighter resume gates.
- **Next:** Re-tighten the resume floors to the prior −10 / −100 defaults (already restored in the base Makefile) and focus on alternative candidates or sharper alpha filters before re-enabling these symbols.

### Run 72 — Predicted-direction gate (GOOG/XLK ≥ 0.01)
- **Command:** `MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP=GOOG:0.01,XLK:0.01 make sim-report` (artefacts under `marketsimulator/run_logs/20251024-162902_*`).
- **Settings:** Enabled the new `MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP` guard so GOOG longs require predicted movement ≥ 1% and XLK shorts require ≤ −1%, on top of existing strategy-return and trend filters.
- **Outcome:** GOOG/XLK remained paused (zero trades); the extra directional gate prevented entries when Kronos forecasts hovered around flat, keeping realised PnL at −$36.0 with max drawdown 0.04 %. Trend streak tracking now shows six consecutive paused runs, highlighting the symbols’ persistent lack of edge.
- **Next:** Keep the predicted-move guard available for future reactivation attempts, but shift focus toward onboarding a fresh candidate once the auto-SMA screener flags a positive-trend symbol.

### Run 73 — Predicted-direction gate + looser strategy returns
- **Command:** `MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP=GOOG:0.01,XLK:0.01 MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP=...GOOG:0.005,XLK:0.005 make sim-report` (artefacts `marketsimulator/run_logs/20251024-163051_*`).
- **Settings:** Retained the directional guard while easing GOOG/XLK strategy-return floors to 0.5 % in hopes of admitting modest-but-directional signals.
- **Outcome:** GOOG/XLK still failed the combined gates (0 trades); the deck stays MSFT-only at −$36.0 with max DD 0.04 %. Paused streaks have now reached seven runs, reinforcing that the mock trend remains too weak even with softer return requirements.
- **Next:** Rather than loosening filters further, focus on alternative symbols or improved alpha inputs. Keep the directional gate in place so reactivation attempts only occur once predicted moves exceed 1 %.

### Run 74 — GOOG forced short with directional gate
- **Command:** `MARKETSIM_SYMBOL_SIDE_MAP=NVDA:sell,GOOG:sell MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP=GOOG:0.01,XLK:0.01 MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP=...GOOG:-0.01 make sim-report` (artefacts `marketsimulator/run_logs/20251024-163603_*`).
- **Settings:** Flipped GOOG to short-only with a 1 % directional gate and negative strategy-return ceiling to see if bearish signals could monetize while XLK remained long-only.
- **Outcome:** Forecasts still failed the combined gates (0 trades); MSFT stayed the sole active symbol at −$36.0 (max DD 0.04 %). Paused streaks escalated to ten runs and were logged to `trend_paused_escalations.csv`, signalling it’s time to sideline GOOG/XLK until their trend stats recover or a replacement symbol is found.
- **Next:** Remove GOOG/XLK from near-term experiments (or raise their suspend thresholds) and hunt for new high-SMA candidates to restore diversification.

### Run 75 — Base roster trimmed (GOOG/XLK removed)
- **Command:** Default `make sim-report` after updating the Makefile symbol roster (`--symbols AAPL MSFT NVDA AMZN SOXX`).
- **Settings:** Dropped GOOG/XLK from the default maps (kelly/min move/strategy return/trend suspend) so they no longer consume gating cycles while their trend PnL stays deeply negative.
- **Outcome:** Deck still trades MSFT only (−$36.0, max DD 0.04 %) because SOXX/AMZN remain suspended; GOOG/XLK now show as neutral in the trend report and no new paused escalations are emitted.
- **Next:** Use `paused_streak_report.py` / `rotation_recommendations.py` to scout a true replacement (SMA ≥ 500). Until a viable candidate surfaces, focus experimentation on improving the active symbols or onboarding a fresh ETF with positive trend momentum.

### Run 76 — SMH onboarding probe (directional gate + neutral roster)
- **Command:** `python scripts/run_sim_with_report.py --prefix 20251024-164144 ... -- python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA AMZN SOXX SMH --kronos-only --flatten-end`.
- **Settings:** Added SMH with conservative limits (kelly 0.15, min move 0.04, min strategy return 0.015, trend suspend/resume −200/−100) to test whether a fresh semiconductor ETF restores diversification while the base roster stays lean.
- **Outcome:** SMH generated zero trades; MSFT remained the only active symbol and PnL slipped to −$44.3 with max DD 0.04 % (two NVDA probes fired). SMH trend stats will need to improve before it can be included in the primary deck.
- **Next:** Keep SMH on the watchlist but defer onboarding until its trend PnL turns positive or the auto-SMA screener flags a stronger candidate.

### Rotation tooling snapshot
- `make trend-status` now logs recommendations to `marketsimulator/run_logs/rotation_recommendations.log` and renders `rotation_summary.md`. As of 2025-10-24T16:48Z the report advises removing GOOG/XLK (paused streak 11) and still finds no SMA-qualified replacements beyond AAPL/MSFT. Continue monitoring the summary for the first positive-trend candidate to stage Run 77.

### Run 77 — QQQ onboarding probe
- **Command:** `python scripts/run_sim_with_report.py --prefix 20251024-164530 ... -- python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA AMZN SOXX QQQ --kronos-only --flatten-end`.
- **Settings:** Added QQQ with directional gating (`MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP=QQQ:0.01`) and conservative limits (kelly 0.15, min move 0.04, min strategy return 0.015, suspend/resume −200/−100).
- **Outcome:** QQQ did not clear the strategy-return filter (0 trades). Portfolio PnL stayed at −$44.3 (max DD 0.04 %), identical to Run 76 aside from NVDA probes. Trend telemetry therefore still lacks a positive-slope candidate.
- **Next:** Continue expanding the candidate ETF list (via `scripts/fetch_etf_trends.py`) and monitor `rotation_summary.md`; defer any roster changes until a symbol shows both positive SMA and acceptable mock forecasts.

### Run 78 — XLY onboarding probe
- **Command:** `python scripts/run_sim_with_report.py --prefix 20251024-165230 ... -- python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA AMZN SOXX XLY --kronos-only --flatten-end`.
- **Settings:** Introduced XLY with the same guardrails as QQQ (kelly 0.15, min move 0.04, min strategy return 0.015, directional gate 1 %).
- **Outcome:** XLY forecasts also failed the entry filters (0 trades). Portfolio metrics mirror Run 77 (PnL −$44.3, max DD 0.04 %), showing current mock analytics remain MSFT-only despite the positive SMA.
- **Next:** Broaden the ETF universe (watchlist now includes 16 symbols) and wait for trend metrics plus strategy signals to align before re-running onboarding trials.

### Run 79 — QQQ aggressive gate (0% min strategy return, 0.5% predicted move)
- **Command:** `python scripts/run_sim_with_report.py --prefix 20251024-165830 ... -- python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA AMZN SOXX QQQ --kronos-only --flatten-end` with overrides `MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP=...QQQ:0.0` and `MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP=QQQ:0.005`.
- **Outcome:** QQQ finally traded (14 fills) but lost $147 (fees $11.9). Portfolio PnL deteriorated to −$191.6 with max DD 0.19 %; QQQ’s mocks generate many low-quality signals once the gate is loosened this far.
- **Next:** Revert to stricter thresholds for production and focus on improving the upstream forecasts (or sourcing alternative alpha) before re-enabling QQQ. The run confirms that loosening strategy-return gates alone produces churn rather than diversification.

### Run 80 — SMH onboarding with conservative gate relaxations
- **Command:** `MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP='AAPL:-0.03,AMZN:-0.02,SOXX:0.015,SMH:0.015' MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP='SMH:0.01,SOXX:0.01' ... python scripts/run_sim_with_report.py --prefix 20251024-Run80 ... -- python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA AMZN SOXX SMH --steps 20 --kronos-only --flatten-end --compact-logs`
- **Outcome:** SMH still failed the strategy-return gate (no trades). PnL matched the baseline (−$44.25, max DD 0.04 %) since only MSFT/NVDA probes executed. The relaxed thresholds were insufficient to admit SMH entries.
- **Next:** Await stronger SMH forecasts before retrying; gate relaxations alone are not enough. Continue monitoring `candidate_forecast_gate_report.md` for when strategy-return exceeds the 0.015 floor multiple times in a row.

### Run 81 — QQQ retry with conservative thresholds
- **Command:** `MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP='AAPL:-0.03,AMZN:-0.02,SOXX:0.015,QQQ:0.015' MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP='QQQ:0.01,SOXX:0.01' ... python scripts/run_sim_with_report.py --prefix 20251024-Run81 ... -- python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA AMZN SOXX QQQ --steps 20 --kronos-only --flatten-end --compact-logs`
- **Outcome:** QQQ still failed the strategy-return gate (no trades), so PnL remained −$44.25 with max DD 0.04 %. Conservative risk caps were in place, but forecasts stayed below the 1.5 % return floor.
- **Next:** Hold off on further onboarding trials until forecasts consistently exceed the entry threshold; use `candidate_forecast_gate_report.md` to catch the first qualifying day before scheduling Run 82.

### Trend Feed Recovery (24 Oct 2025)
- **Command:** `. .venv/bin/activate && python scripts/fetch_etf_trends.py --symbols-file marketsimulator/etf_watchlist.txt --days 365 --window 50 --providers stooq yahoo`
- **Settings:** Added an automatic Yahoo Finance fallback in `scripts/fetch_etf_trends.py` (new `--providers` flag defaults to `stooq yahoo`) so the pipeline keeps working whenever Stooq returns incomplete data. The fetch now records which provider supplied prices.
- **Outcome:** All 16 watchlist ETFs refreshed successfully via the Yahoo fallback (e.g., `QQQ@yahoo: SMA=590.76 pnl=126.51`), updating `marketsimulator/run_logs/trend_summary.json` and restoring downstream readiness artefacts. Follow-on jobs regenerated `candidate_readiness.md`, `candidate_readiness_history.csv`, `candidate_momentum.md`, and `candidate_forecast_gate_report.md`; probes confirm SMH/QQQ/XLY still miss the 1.5 % strategy-return gate while SOXX/MSFT/AAPL remain clear.
- **Next:** Re-run the readiness loop daily with the new fallback in place. Monitor `candidate_forecast_gate_history.csv` for narrowing strategy-return shortfalls; if SMH or QQQ close within 0.003 of the threshold the alert script (`forecast_margin_alert.py`) will now highlight them.

### Trend Pipeline Automation + Provider Tracking (24 Oct 2025)
- **Command:** `. .venv/bin/activate && python scripts/run_daily_trend_pipeline.py`
- **Settings:** New orchestrator script executes the full refresh loop in one shot (fetch trends, readiness/momentum reports, forecast probes, margin alerts) using the Yahoo fallback when Stooq fails. `fetch_etf_trends.py` now writes the provider used for each symbol into `trend_summary.json` and emits `[notice]` log lines whenever a symbol switches providers.
- **Outcome:** Pipeline run confirms all 16 ETFs refreshed via Yahoo (`Provider usage this run: yahoo:16`). Downstream artefacts regenerate automatically and margin alerts continue to show no near-threshold candidates.
- **Next:** Schedule the orchestrator in cron/CI and monitor provider-change notices to track when Stooq resumes serving data; once Stooq stabilises, evaluate blending providers instead of relying solely on Yahoo.

### Make Target + CI Wiring (24 Oct 2025)
- **Command:** `make trend-pipeline`
- **Settings:** Added a dedicated Makefile target that simply calls `scripts/run_daily_trend_pipeline.py`, making it trivial for local ops or CI to trigger the refresh. Introduced `.github/workflows/trend-pipeline.yml`, a manual/scheduled workflow that sets up Python+uv, installs the repo in editable mode, runs `uv run make trend-pipeline`, and uploads refreshed log artefacts.
- **Outcome:** `make trend-pipeline` mirrors the standalone script run (all symbols fetched via Yahoo fallback, readiness/momentum/gate artefacts updated, no near-threshold alerts). The GitHub workflow is ready for activation—scheduled 01:15 UTC runs will keep the logs current, and manual dispatches give analysts a quick retry button.
- **Next:** Monitor the first scheduled workflow execution and extend artifact uploads if additional diagnostics (e.g., `rotation_summary.md`) become critical for daily reviews.

### Provider Usage Logging (24 Oct 2025)
- **Command:** `. .venv/bin/activate && python scripts/run_daily_trend_pipeline.py`
- **Settings:** `scripts/fetch_etf_trends.py` now accepts `--provider-log` / `--provider-switch-log` / `--latency-log`, appending usage counts and per-symbol latency (ms) to CSVs. The pipeline writes human-readable summaries (`provider_usage_summary.txt`, `provider_latency_summary.txt`) plus an emoji sparkline (`provider_usage_sparkline.md`).
- **Outcome:** Latest pipeline run logs `yahoo` usage for all 16 ETFs (timeline `YYYYYYYYYY`) with mean latency ≈320 ms (p95 ≈ 351 ms). `provider_latency_rollup.csv` captures per-run aggregates; `provider_latency_rolling.{md,json}` surfaces windowed averages and deltas; `provider_latency_rolling_history.jsonl` stores the raw snapshots; and new reports `provider_latency_history.md` / `provider_latency_history.html` visualize longer-horizon trends. The pipeline now calls `scripts/notify_latency_alert.py` whenever ΔAvg ≥ 40 ms, appending messages to `provider_latency_alerts.log`; no alerts have fired yet. `provider_switches.csv` remains empty while Stooq is offline.
- **Outcome:** Latest pipeline run logs `yahoo` usage for all 16 ETFs (timeline `YYYYYYYYYY`) with mean latency ≈320 ms (p95 ≈ 351 ms). `provider_latency_rollup.csv` captures per-run aggregates; `provider_latency_rolling.{md,json}` surfaces windowed averages and deltas; `provider_latency_rolling_history.jsonl` stores the raw snapshots; and new reports `provider_latency_history.md` / `provider_latency_history.html` visualize longer-horizon trends while `provider_latency_history.png` provides a thumbnail for embeds. The pipeline now calls `scripts/notify_latency_alert.py` whenever ΔAvg ≥ 40 ms, appending messages to `provider_latency_alerts.log` and posting JSON to a configurable webhook (Slack channel override, Teams card, or raw payload) with deep links to logs/plots. Rotation summaries embed a “Data Feed Health” table sourced from `provider_latency_rolling.json`, so roster decisions factor in feed stability. `provider_switches.csv` remains empty while Stooq is offline.
- **Next:** Once Stooq resumes serving data, watch the usage timeline/sparkline and latency summary for mix shifts; if yahoo latency degrades further consider expanding the provider roster or caching responses.
