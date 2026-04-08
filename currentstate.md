# Current State — 2026-04-08

Snapshot of what is really running on the Alpaca live account, why it has stopped
trading, what infrastructure we have to iterate with, and the shortlist of
experiments most likely to lift monthly PnL toward the 27%/month HARD RULE.

This is the launch pad for the current autonomous push. Keep it updated as the
picture changes; promote deployable findings into `alpacaprod.md` and push
session narratives into `alpacaprogress8.md`.

---

## 1. What is actually running in production (verified on host)

| Component | State | Notes |
|-----------|-------|-------|
| `daily-rl-trader.service` | **ACTIVE** since 2026-04-08 10:23:53 UTC, PID 2599365 | `.venv313/bin/python -u trade_daily_stock_prod.py --daemon --live --allocation-pct 12.5` |
| Checkpoint | `pufferlib_market/prod_ensemble/tp10.pt` + 31 sibling ckpts (32-model softmax ensemble) | Pre-RSI daily feature schema (legacy). Schema guard now enforced at load. |
| Symbols | AAPL MSFT NVDA GOOG META TSLA SPY QQQ PLTR JPM V AMZN | 12 tickers, 25-action head |
| Allocation | `--allocation-pct 12.5` | Single-name concentration, **no** leverage |
| `unified-orchestrator.service` | `inactive` | Hourly meta trader offline since 2026-03-29 (Gemini exhausted) |
| `alpaca-monitor.service/timer` | loaded (timer active, unit dead between ticks) | 6h health monitor |
| LLM stock trader (glm-4-plus) | process up, Alpaca writes gated by singleton | Not producing real trades |
| Alpaca LIVE keys | **Were flagged expired on 2026-04-06 audit** | Need to re-verify — daemon logs do not show 401s post-restart, so keys may already be refreshed. TODO: hit the account endpoint directly. |

The singleton writer lock + death-spiral guard from the 2026-04-08 session are
live; only one process can write to the live Alpaca account, and any sell >50bps
below the most recent recorded buy for the same symbol will `RuntimeError`.

## 2. Why nothing has traded since ~2026-04-01 (ROOT CAUSE)

The daemon is healthy and emitting signals once per day. It is **not** being
blocked by API keys or the singleton lock. It is being blocked by the execution
safety gate:

```
src/daily_stock_defaults.py:81: DEFAULT_MIN_OPEN_CONFIDENCE = 0.20
trade_daily_stock_prod.py:2068: if confidence < float(min_open_confidence): BLOCK
```

Observed ensemble confidences from journald:

- 2026-04-01 13:35 — confidence **11.7%** → blocked
- 2026-04-07 13:35 long_META — confidence **9.3%**, value_est +0.485 → blocked (would have opened under any sane threshold)
- 2026-04-08 13:35 long_AAPL — confidence **10.6%**, value_est **−0.0215** → blocked (value_estimate gate would also block this one, correctly)

`confidence` is `softmax(logits).max()` over a **25-action** head. With 12
symbols × (long/short/flat) + cash the ceiling is structurally ~0.15–0.20 even
for a high-conviction policy. The 0.20 floor is effectively a kill switch. It
was almost certainly intended as a 20% edge threshold on a **different**
normalization (e.g., margin over uniform, or per-symbol long/flat softmax), not
raw 25-way softmax.

**Impact:** zero open submissions since the gate was introduced. The only
execution we see in logs is the GOOG managed-close on 2026-04-07, which
correctly fired the exit path. `execution_status="blocked_open_gate"` is the
universal terminal state for every run.

### Two gate fields, both relevant
- `min_open_confidence = 0.20` — naive softmax floor (broken by construction)
- `min_open_value_estimate = 0.0` — value-head floor (reasonable, kept)

On 2026-04-07 the META signal had positive value_est AND was still blocked by
confidence alone. That's a strictly losing behaviour — a rule that never fires
cannot help us, and leaving it on means the ensemble's validated PnL is
unreachable.

### Fix strategy (before code change)
1. Recompute the correct confidence semantics used by the ensemble (softmax_avg
   over which axis?) and decide whether to (a) lower the floor to a
   statistically meaningful percentile of historical confidence or (b) replace
   with a margin metric `top1 − top2`.
2. Backtest the proposed threshold on the last 120 trading days using the
   offline replay harness already used in `alpacaprod.md` (legacy feature
   mapping). Confirm it recovers the META-style signals without opening on
   the AAPL-style −value signals.
3. Only then change `DEFAULT_MIN_OPEN_CONFIDENCE` and restart. `value_estimate`
   floor stays as the real edge gate.

Do NOT push a blind fix to prod without replay; the 2026-04-08 alpacaprod entry
already shows that "current RSI mapping + current ensemble" replay is
**−0.92% total / −0.16% monthly** on 120 days, so we cannot simply uncork the
gate and expect 27%/month to materialize.

## 3. Validation vs live: the other mismatch (still unresolved)

From 2026-04-08 audit in `alpacaprod.md`:

- Live ensemble was trained on legacy daily features (no RSI).
- Checked-in feature builder now emits RSI(14) in place of duplicate `trend_20d`.
- Schema guard in `src/daily_stock_feature_schema.py` prevents silent mixing,
  but the live ensemble runs on the **legacy** code path, so holdout numbers
  from `stocks12_v5_rsi/*` (s42 med=+36%, 0/58 neg) are **not** the numbers
  production is realizing.

120-day calibrated replay of the running 32-model ensemble:

| Config | Total | Monthly | Sortino | MaxDD | Trades |
|--------|-------|---------|---------|-------|--------|
| Legacy feature map (matches live) | +0.21% | +0.04% | 0.14 | −2.99% | 24 |
| Current RSI feature map | −0.92% | −0.16% | −0.53 | −3.04% | 26 |
| RSI + 95% concentration | −5.17% | −0.93% | — | −21.58% | — |
| RSI + 190% @ 2x BP | −22.86% | −4.44% | — | −43.15% | — |

**Conclusion:** the current live ensemble is ~break-even at best under honest
replay, miles below the 27%/month HARD RULE, and concentration / leverage make
it catastrophically worse. We cannot deploy v5_rsi s42 onto the legacy ensemble
path either (feature mismatch). Two paths forward:

- **Path A — Replace ensemble with v5_rsi family.** Train 5–10 robust v5_rsi
  seeds (beyond just s37/s38/s42), ensemble them, validate on a longer window
  than Jul–Nov 2025, deploy cleanly under the RSI schema. Removes the mismatch
  and plausibly reaches double-digit monthly.
- **Path B — Retrain the 32-model prod family on the RSI features.** Slower and
  less exciting; keeps the ensemble architecture we already validated end to end.

Path A is cheaper. The bottleneck is honest validation data length.

## 4. Inventory of training/eval tooling (what we can iterate with today)

### RL training backends
- **PufferLib / C env** (`pufferlib_market/`) — pure C market sim + PPO. All
  current prod checkpoints live here (`prod_ensemble/*.pt`, `stocks12_v5_rsi/*`,
  `stocks12_v5_rsi_cryptorcp/*`, `ablate_{tp02,obsonly}/*`). Build with
  `cd pufferlib_market && python setup.py build_ext --inplace` in `.venv313`.
  This is the main workhorse and we should keep using it.
- **Crypto champion** (`crypto12_ppo_v8_h1024_300M_annealLR`) — 30d 2658.9× best,
  pure PPO h1024 + RunningObsNorm + BF16 autocast + CUDA-graph. The
  crypto-recipe port to daily stocks has already been tried and **failed**
  (see `alpacaprod.md` 2026-04-07 entries: 8/8 seeds under baseline). Root cause
  diagnosed: obs-norm hurts daily stocks because returns are near-Gaussian. Do
  not revisit this recipe wholesale.
- **HF Trainer / TRL stack** — present in repo (`fp4/`, `src/autoresearch_stock/`,
  chronos2 wrappers, `tests/test_hftrainer_step_timing.py`). Used for Chronos2
  LoRA and autoresearch experiments, **not** for prod trading policies. Useful
  for forecasting side — Gemini 3.1 reforecaster sits in the hourly unified
  path, which is currently inactive.
- **Chronos2** (`src/models/chronos2_wrapper.py`, `multiscale_chronos.py`) —
  forecasting model with trainable LoRAs. Currently dormant in the daily stock
  loop; powers the hourly unified trader when that comes back online.
- **nanochat / modded-nanogpt** — reference ML code, not wired into trading.

### Evaluation
- `scripts/eval_100d.py` — HARD RULE gate, binary fills, `decision_lag=2`,
  fail-fast on max-dd. This is the authority on whether a model is deployable.
- `pufferlib_market/evaluate_holdout.py` — window-level holdout with per-seed
  `eval_holdoutN.json`.
- `src/marketsim_video.py` + `scripts/render_prod_stocks_video.py` — MP4/HTML
  intrabar replay, serves artifacts via `src/artifacts_server.py`.
- 120-day calibrated replay harness referenced from `alpacaprod.md` — the most
  recent "is the running service actually printing money?" check. This is the
  correct tool for reproducing live PnL offline.
- C parity harness — `ctrader/binance_bot/` now runs the s42 MLP in pure C
  end-to-end with byte-perfect obs parity and 1.4e-6 MLP parity. Useful as a
  future live bot path and as a fast, GPU-free backtest sandbox. The v0 trade
  sim inside it is still too simplified to replace the Python replay.

### Acceleration wishlist (the user specifically asked for this)
- **Fail-fast marketsim** — the existing `--fail-fast-max-dd 0.20` flag already
  exists on `eval_100d.py`; extend the same pattern to the 120-day replay path
  and the per-seed holdout runner. If `total < X` or `drawdown > Y` at day N,
  abort the rest of the window.
- **Skip video on loser runs** — `render_prod_stocks_video.py` currently
  renders unconditionally. Gate the MP4 writer on `total_return > threshold`
  so dud runs cost only the sim time, not the ffmpeg time.
- **Symbol early-out** — the 12-symbol sweep can short-circuit the first N
  symbols' trades if cumulative return is already below kill threshold.

These are the three concrete sim-speed wins to land before the next big sweep.

## 5. What I'd try next (ranked by expected PnL lift × chance of success)

Each item ends with the single experiment that would prove or disprove it.

### High priority
1. **Diagnose & repair the execution safety gate.** Top priority. Without it,
   nothing we do upstream matters. Replay the last 120 days with
   `min_open_confidence ∈ {0.0, 0.05, 0.10, 0.15, 0.20}` and chart recovered
   PnL vs false-positive rate. Pick the one that maximises Sortino on the
   legacy ensemble replay. One afternoon of work, probably the single largest
   immediate PnL delta available.

2. **Swap ensemble to v5_rsi family (Path A).** Train v5_rsi seeds 50–100
   with the exact same recipe as s37/s38/s42 (which are the only three known
   robust seeds), rank by holdout p10, build a 10–20 model ensemble, run
   120-day replay, compare to the legacy 32-model replay. If p10 > legacy
   and monthly median > legacy, deploy. Rough cost: a few hours GPU time per
   seed on the local rig, cheaper than gpu_pool.

3. **Extend v5_rsi validation period.** Current s42 holdout is Jul–Nov 2025
   only. Before any redeploy, rebuild `stocks12_daily_v5_rsi_val.bin` with at
   least 2024-01 → 2025-12 and rerun exhaustive eval. This is the
   "don't trust short-window champions" guardrail the Crypto70 debacle
   taught us.

### Medium priority
4. **Per-symbol monolithic training.** `ctrader/market_sim.h MAX_SYMBOLS=64` is
   the cap on the monolithic scale path; extending it to 100+ symbols is the
   "scale the universe" lever flagged in alpacaprod.md. Expected direct
   expected-value lift if we can find even marginally profitable new symbols,
   because the ensemble picks its highest-conviction symbol per day.

5. **Confidence-calibrated sizing.** Replace the binary `allocation_pct` with a
   sizing function over the ensemble's confidence margin (top1 − top2) or
   value_estimate. Bigger bets on high-conviction days, smaller on marginal.
   Only valuable once the gate is fixed — otherwise it's a sizing rule on an
   empty set.

6. **Gemini 3.1 reforecaster test.** The user flagged it as "working nicely" on
   the hourly pipeline but `unified-orchestrator.service` is dead. Bring it
   back up in paper-only mode, compare hourly PPO vs hourly PPO + Gemini
   reforecast on the 60-day binance/stocks sim, decide whether to restart it
   live. Independent of the daily stock fix.

### Low / speculative
7. **Hourly stocks via newnanoalpacahourlyexp.** Per-symbol architecture is
   blocked on the MAX_SYMBOLS=64 cap; decide if we fix the cap or stay on the
   daily cadence. Daily has higher Sharpe in the repo's historical record
   (see `project_daily_vs_hourly.md` — daily wins 3.3×), so this is
   speculative.
8. **Chronos2 LoRA fine-tune on daily stock returns as aux head.** Cheap to
   try, has never moved the needle in prior sessions.

## 6. Open threads to keep in mind
- The currently running `daily-rl-trader.service` PID 2599365 predates the
  state-file hardening / feature-schema guard repo fixes from today's session.
  Restart was already done; if something else lands, restart again.
- `unified-orchestrator.service` + Gemini key renewal is a **separate** track.
- Crypto side: s71 champion + s118/s123 are still deployment-blocked
  (bull-market overfitting). Not touching for this push.
- Alpaca LIVE key validity needs a direct GET /v2/account probe. Logs suggest
  auth is fine post-2026-04-08 restart but no live trades have executed, so
  we haven't proven writes work.

## 6b. 2026-04-08 session findings (the plan has to change)

Running the 120-day calibrated replay (`--backtest-entry-offset-bps 5 --backtest-exit-offset-bps 25`, allocation 12.5%, local data, `trainingdata/` latest 2026-04-01) on each candidate produced:

| Config | Total | Annualized | Sortino | MaxDD | Trades |
|---|---|---|---|---|---|
| 32-model legacy prod ensemble | +0.21% | +0.44% | 0.14 | −2.99% | 24 |
| v5_rsi s37 solo | −2.38% | −4.94% | −1.19 | −3.27% | 4 |
| v5_rsi s38 solo | +0.32% | +0.68% | 0.22 | −3.23% | **0 (degenerate)** |
| v5_rsi s42 solo | −3.12% | −6.44% | −1.32 | −3.77% | 26 |
| v5_rsi s37+s38+s42 ensemble | −3.61% | −7.44% | −1.74 | −4.33% | 39 |
| ens3 @ 95% concentration | −8.12% | −16.29% | −0.49 | −23.74% | 31 |
| ens3 @ 95% + `--backtest-buying-power-multiplier 2.0` | −8.12% | −16.29% | −0.49 | −23.74% | 31 |

Three hard facts come out of this:

1. **The "confidence gate" is not the bottleneck.** Sweep of `--min-open-confidence ∈ {0.00 … 0.20}` and `--min-open-value-estimate ∈ {0, 10}` produced **byte-identical** backtest results. The live execution gate exists only in the live run path (`trade_daily_stock_prod.py:2057–2070`); the backtest harness skips it entirely. Consequences:
   - The "+0.04% monthly" number in `alpacaprod.md` was already an **ungated** number. Removing the live gate cannot uncover more PnL than what the replay already shows, because that's the ungated ceiling.
   - Gate tuning is **unvalidatable** today. Before tuning anything, the backtest must actually call the gate function (fix the plumbing — one new code path). Otherwise every threshold sweep is noise.

2. **The v5_rsi champions are overfit to their Jul–Nov 2025 holdout.** s42's memory-famous "med +36% / 0/58 neg" evaporates into −3.12% / Sortino −1.32 on the last 120 live trading days. s38's "0.22 Sortino" is a numerical artefact of zero trades. The ensemble of the three robust seeds is worse than any single seed. This is a textbook overfit-to-validation-window failure — same pattern that killed the Crypto70 s670/s275/s292 short list on exhaustive eval.

3. **Nothing currently checked in can reach 27%/month under honest replay.** Even the legacy 32-model ensemble is ~break-even. Concentration and leverage just linearly scale a near-zero signal into a −8%/120d drag. `--backtest-buying-power-multiplier` appears to be a **no-op in this code path** (identical output with 1× vs 2×). That's a second backtest-plumbing bug that has to be fixed before any leverage decision is meaningful.

### Updated diagnosis
- We do not have a deployable positive-edge algorithm on the last 120 live trading days.
- The holdout validation regime the ensemble uses is not honest: short window, non-contiguous with live, no walk-forward.
- The backtest tooling has two silent gaps (gate skipped, leverage multiplier ignored) that made those issues invisible until this replay.

### What has to happen before training more seeds
1. **Patch backtest gate plumbing.** Make `run_backtest_variant_matrix_*` actually enforce `min_open_confidence` + `min_open_value_estimate` so we can sweep them honestly. Add a regression test in `tests/test_trade_daily_stock_prod.py`.
2. **Patch backtest leverage.** Make `--backtest-buying-power-multiplier 2.0` actually scale order sizing. Regression test.
3. **Rebuild honest validation dataset.** `stocks12_daily_v5_rsi_val.bin` needs to include at least 2024-01 → 2026-04 with walk-forward windows, not only Jul–Nov 2025. Add the 120-day live replay as an additional gate so future champions have to pass both exhaustive holdout AND the exact harness that generates the production number.
4. **Only then** run a fresh seed sweep. Expand beyond 12 symbols (alpacaprod.md flagged MAX_SYMBOLS=64 bump as a lever; the 12-ticker universe is too thin to support 27%/mo without overfitting).
5. **Measure Sortino-per-allocation.** The 27%/mo goal is reachable two ways: (a) find a policy that hits +27%/mo at 1× notional (hard on 12 mega-caps), or (b) find a policy with monotonic smoothness where 2× leverage recovers target. Path (b) needs the leverage plumbing fixed first.

### Recommendation to user
- The "confidence-gate replay sweep" in the original plan is **not worth running** as a PnL lever — the gate has no backtest plumbing and the ungated ensemble is already break-even. It still has to be patched, but for correctness, not for PnL discovery.
- Proposed revised short-list in priority order:
  1. Fix backtest gate + leverage plumbing (both bugs found this session). Small code + tests.
  2. Rebuild validation dataset with 2024-01 → 2026-04 walk-forward windows; re-rank all existing checkpoints under the new regime. This may already surface a winner we've been ignoring.
  3. Launch a proper v5_rsi seed sweep (50–100 seeds) under the new regime. Use fail-fast eval + skip-video-on-losers so duds cost seconds.
  4. Expand symbol universe beyond the 12 current tickers as a parallel lever.
  5. Only then touch live allocation / leverage / gate thresholds.

Estimated cost: (1)–(2) are a few hours of engineering on the local box with no GPU. (3) is GPU-hours. I can proceed with (1) and (2) immediately.

## 7. Immediate next actions (this session)
1. Verify Alpaca LIVE keys with a direct read-only account call. Update this
   file with balance + last trade timestamp.
2. Implement the gate diagnostic: replay 120 days with varying
   `min_open_confidence`, plot, choose new default, land as a config change.
3. Add fail-fast + skip-video switches to the sim tooling as described in §4.
4. Kick off v5_rsi seed sweep (seeds 50–100, same recipe as s42) in
   background; log to `alpacaprogress8.md` as each result lands.
5. Only after (1)–(3) land: decide whether to promote v5_rsi ensemble over
   the 32-model ensemble.

Touch `alpacaprod.md` on each actual prod change. Everything investigative goes
in `alpacaprogress8.md`. This file is the living overview.
