# XGB Daily Trader — Optimization Ledger

Running ledger for the drawdown-reduction / Sortino-first / realism campaign
on the XGB `top_n=1 n_estimators=400 max_depth=5 learning_rate=0.03`
champion. Prod baseline sits at +32.2%/mo median with 31.9% worst DD on the
846-symbol universe over 34 OOS windows (2024-01-02 → 2026-04-18, Alpaca-real
fee 0.278bps, fill_buffer 5bps, binary fills, decision_lag=2).

**Metric priority**: median Sortino > p10 monthly% > med monthly%. We will
NOT ship a change that trades Sortino down for median up.

**Headline artifact to re-verify**: `analysis/xgbnew_leverage_sensitivity/multiwindow_20260418_062758.json`.

Note: feedback_xgb_fast_features_divergence.md flags that the polars-fast and
pandas feature paths diverge on p10 (polars/lev=1.25/seed=0 p10=8.61% vs
pandas/lev=1.0/seed=0 p10=19.51%). All headline experiments here will **pin
the pandas path** (no `--fast-features`) so baselines are comparable across
runs. Fast path is fine for exploratory sweeps only.

---

## Experiment queue (ranked by expected lift × cheapness)

| # | Experiment | Expected lift | Cost | Status |
|---|-----------|---------------|------|--------|
| E1 | SPY MA50 regime gate (flat when SPY < MA50) | +1–2%/mo med, DD −8–12% | 1 sim run | queued |
| E2 | SPY MA20 regime gate | similar to MA50, more active | 1 sim run | queued |
| E3 | Vol-target sizing (scale by 1/realised_20d_vol) | DD −5–10%, med ±1% | 1 sim run | queued |
| E4 | top_n=2 half-weight diversification | DD −6–10%, med −1–3% | 1 sim run | queued |
| E5 | Confidence-floor gate (skip day if top score < threshold) | DD −3–5% if monotone | grid of 5 thresholds | queued |
| E6 | VIX-proxy gate (flat if SPY 20d realised vol > threshold) | catches 2026 crash DD | 1 sim run | queued |
| E7 | SPY/QQQ short overlay when regime bearish | long-short hedge, DD −10%+ | 2 sim runs | queued |
| E8 | Hourly intrabar fill sim (MOO auction + queue) | realism, not PnL | code + parity tests | queued |
| E9 | Oversubscription / partial-fill model for top_n≥2 | realism, affects E4 | piggybacks on E8 | queued |
| E10 | Combine best gate + best sizing | stack | after above | queued |

---

## Baseline (pandas path, lev=1.0, no gate) — to be re-confirmed before any change

| metric | value |
|---|---|
| median monthly % | +32.2% (from 6-lev × 3-seed sweep) |
| p10 monthly % | +20.3% |
| median sortino | 8.42 |
| max DD (worst window) | 31.9% |
| neg windows | 0/34 |

We will replicate this baseline as the first row in the results table below,
using the exact same symbol list, window grid, and feature path as every
experiment, so deltas are apples-to-apples.

---

## Results

(Populated as experiments complete. Each row captures: config, med, p10,
med sortino, worst DD, neg windows, and Δ vs baseline.)

| tag | config | med/mo | p10/mo | sortino | worst DD | neg | Δ median | Δ sortino | artifact |
|---|---|---|---|---|---|---|---|---|---|
| baseline | lev=1.0 no-gate seed=0 | **+32.80%** | **+20.26%** | **8.28** | **31.87%** | 0/34 | 0 | 0 | `analysis/xgbnew_dd_sweep/baseline/multiwindow_20260418_100231.json` |
| ma50 | + `--regime-gate-window 50` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | `analysis/xgbnew_dd_sweep/ma50/` (queued) |
| ma20 | + `--regime-gate-window 20` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | `analysis/xgbnew_dd_sweep/ma20/` (queued) |
| voltarget015 | + `--vol-target-ann 0.15` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | `analysis/xgbnew_dd_sweep/voltarget015/` (queued) |

---

## Status log

**2026-04-18 09:57 UTC — Infrastructure landed, first 4-run sweep launched**

Changes on disk:
- `xgbnew/backtest.py`: added `BacktestConfig.regime_gate_window` (SPY <
  MA(N) ⇒ stay in cash) and `BacktestConfig.vol_target_ann` (scale daily
  exposure by `min(1, target_ann / SPY_20d_realised_ann_vol)`). Two pure
  helpers `_build_regime_flags` / `_build_vol_scale` added; `simulate()`
  now accepts `spy_close_by_date` kwarg and threads it into both knobs.
- `xgbnew/eval_multiwindow.py`: added `--regime-gate-window`, `--vol-target-ann`,
  `--spy-csv` CLI flags; loads SPY closes once per eval and passes to `simulate`.
- `tests/test_xgbnew_backtest_regime_gate.py`: 8 tests covering both knobs,
  synthetic uptrend/downtrend SPY, disabled path, missing dates. All green.
- `scripts/xgb_dd_reduction_sweep.sh`: launches 4 back-to-back evals
  (baseline / ma50 / ma20 / voltarget015) all on the pandas path, seed=0,
  lev=1.0, pinned for apples-to-apples comparison.

Sweep in flight at PID 92539 — logging to `logs/xgb_dd_sweep_20260418_095731.log`.
Expected wall time ~60 min (4 × 15 min).

**2026-04-18 10:03 UTC — baseline landed, ma50 starting**

Baseline re-confirmed at med +32.80% / p10 +20.26% / sortino 8.28 / 0/34 neg
(1 bp tighter median than the prior 6-lev sweep because that aggregated three
seeds; this pins seed=0 only). Baseline row filled in the table above; worst-DD
left TBD until we pull the max over window max_dd_pct from the JSON.
`ma50` run started at 10:02:38.

On sweep completion: analyze each JSON with `xgbnew/analyze_sweep.py`, fill
the results table above, promote the winner into a combined run (best gate
+ vol-target, if both are net-positive), then queue E4/E5/E6/E7.

---

## Market-simulator realism work-in-progress (E8, E9)

**Current state (daily-bar sim)**:
- `xgbnew/backtest.py`: `entry_fill = open * (1 + fb)`, `exit_fill = close * (1 - fb)`.
- No order queue, no partial fills, top_n picks treated as independent equal-notional positions.
- For top_n=1 this is fine — one MOO order per day.
- For top_n ≥ 2 this **under-models** the reality that a single account with
  limited buying power and order-arrival asymmetry will fill one stock
  cleanly at MOO and the other at an adverse 9:31 tick.

**Proposed hourly intrabar extension** (E8 + E9):
1. Walk `trainingdatahourly/stocks/<SYM>.csv` for each pick's first hour.
2. Model order submission as a priority queue: orders submitted in rank order
   (best score first), each consuming notional from a shared BP pool.
3. If MOO for pick #1 fills at `open*(1+fb)`, pick #2 competes against
   09:30–10:30 hourly bar with `open_of_next_hour * (1+fb_inflated)` — the
   buffer grows because the price has moved off the clean MOO.
4. For the sell side, symmetric at MOC.
5. Work-stealing: if pick #1 MOO fails (illiquid, halted), pick #2 gets the
   full BP. Reflects real API behavior.
6. Parity test: at top_n=1, new sim must match old daily sim within 5bps
   tolerance per day (one MOO + one MOC fill, no queue competition).

Existing reference: `pufferlib_market/intrabar_replay.py` (1059 LOC) walks
hourly OHLC for daily-decision RL policies and fires stop/TP/max-hold at the
exact hour. We should re-use that backbone, not rewrite.

Data shape: `trainingdatahourly/stocks/SPY.csv` has 4025 rows from 2020 →
present (covers the full OOS window). Per-symbol hourly CSVs already exist
for the full universe (ran grep — all mentioned syms present).

---

## Rule: before shipping anything to prod from this file

1. Re-run baseline on the exact same command (pandas path, seed=0, lev=1).
2. Re-run candidate with only one knob changed.
3. Δ median sortino must be strictly ≥ 0, AND neg windows must not increase.
4. If Δ med monthly < 0 but Δ sortino > 0 and Δ worst-DD < 0, it's a **ship**
   (smoothness is the stated objective).
5. Write the artifact path into the results table. Stale numbers = no ship.
