# Current Algorithms — Monitor Ledger

**Authoritative source**: `alpacaprod.md` top section is always the canonical
live ledger. This file is a short pointer for the hourly monitor so it knows
which services should exist, where the best config lives, and what the bar
is to beat. Keep this file in sync with `alpacaprod.md` whenever we deploy.

Last synced: **2026-04-21 13:10 UTC — deploy gate below corrected: measured on pre-stale-fix `oos2024_ensemble_gpu`. Fresh-ensemble re-measurement (2026-04-21) shows 0/108 positive-median cells at ms=0.85. LIVE is correctly in hold-cash mode — the gate is WAI. See §Deploy gate below + `project_xgb_true_oos_no_edge_2026_04_21.md`.**

---

## 1. XGB Daily Trader — LIVE (primary, as of 2026-04-19 full stack)

- **Process**: `xgbnew.live_trader` (supervisor unit `xgb-daily-trader-live`,
  launcher `deployments/xgb-daily-trader-live/launch.sh`). NOT systemd.
- **Broker boundary**: **direct Alpaca SDK** — no `trading_server` on
  `127.0.0.1:8050`. Writes go straight to `api.alpaca.markets` using
  `ALP_KEY_ID_PROD` / `ALP_SECRET_KEY_PROD`.
- **Singleton**: `src/alpaca_singleton.py::enforce_live_singleton` fires at
  startup; the live-writer lock at `strategy_state/account_locks/alpaca_live_writer.lock`
  must show `service_name: xgb_live_trader` and a `pid` that matches
  `supervisorctl status xgb-daily-trader-live`.
- **HARD RULE #3 (death-spiral guard, TIME-AWARE as of commit 84ae21ef)**:
  `xgbnew/live_trader.py` calls `record_buy_price(sym, fill_px)` after each
  BUY and `guard_sell_against_death_spiral(sym, "sell", current_price)`
  before each SELL. Tolerance auto-selects by the age of the recorded buy:
  - Buy ≤ 8h old (intraday regime): sells > **50 bps** below buy are refused.
  - Buy > 8h old (hold-through overnight regime): sells > **500 bps** below
    buy are refused. This widens the tolerance to tolerate normal overnight
    gaps without tripping the guard.
  RuntimeError from the guard crashes the loop by design; supervisor
  autorestart handles reconnection. DO NOT "catch and skip" this exception.
- **Ensemble**: 5-seed alltrain GPU —
  `analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed{0,7,42,73,197}.pkl`,
  blend mode `mean` (predict_proba averaged). Models trained with
  `n_estimators=400 max_depth=5 learning_rate=0.03` on 2020-01-01 → 2026-04-19
  (846-symbol universe). Retrained weekly by `scripts/xgb_weekly_retrain.sh`
  (systemd timer `xgb-auto-retrain.timer`, Sun 23:00 UTC).
- **Deployed flags (live launch.sh)**:
  - `--top-n 1` — one symbol per session.
  - `--allocation 2.0` — 2× leverage (buy_notional = 200% equity).
  - `--min-score 0.85` — **ensemble-calibrated conviction gate**. Holds
    cash if top-1 blended `predict_proba < 0.85`. In OOS validation on
    `oos2024_ensemble_gpu` models across 60 windows, **~44% of trading
    days cleared the gate** — the other ~56% held cash. Daily top-score
    distribution: min 0.35, median 0.84, max 0.94. See §EXPECTED
    BEHAVIOR below for what "healthy no-pick" looks like.
  - `--hold-through` — if tomorrow's top-1 pick == today's held symbol,
    skip the SELL-then-BUY round trip; hold overnight. On no-pick days
    the current position is also held (does not flat at close).
  - `--min-dollar-vol 50000000` — inference liquidity floor. Drops
    ~240 symbols with $ADV < $50M. Strict-dominance win in OOS (+5.23 p10).
  - `--min-vol-20d 0.12` — inference volatility floor (tightened 0.10→0.12
    on 2026-04-20 13:17 UTC, strict-dom TRUE-OOS on `oos2024_ensemble_gpu`).
    Drops dead-zone low-vol names; stacks cleanly with 50M dolvol floor.
- **Baseline expected monthly return (TRUE OOS on `oos2024_ensemble_gpu`,
  2025-01-02 → 2026-04-19, 60 windows, fee=0.278bps fb=5bps, vol_20d=0.12)**:
  - deploy fees: **median +153%/mo, p10 +97%, 0/60 neg, worst DD 7.18%**
  - 36× fee stress: **median +102%/mo, p10 +58%, 0/60 neg, worst DD 8.34%**
  - intraday worst-DD (unrealised): ~13% both regimes
- **Health check threshold for realised PnL**: target 60-70% of in-sample
  headline = **≥ +40%/mo** in first month. **< +10%/mo over 2 trading
  weeks = broken** (signal drift, fill issues, or stale data pipeline).
- **Character (CHANGED 2026-04-19 with hold-through)**:
  - Signal: one session per trading day (at ~09:20 ET pre-open).
  - On pick-day: BUY at 09:30 ET, **usually hold into the next trading
    session** (flat overnight is the exception, not the rule).
  - On no-pick day: hold whatever's already in the position — **not
    force-flat**. This is by design (`--hold-through` semantics).
  - Top-1 pick is ~$57K notional on $28K equity at `--allocation 2.0`.
- **Logs**: `sudo tail -f /var/log/supervisor/xgb-daily-trader-live.log` and
  `…-error.log` (singleton lock + stderr). Trade log (structured JSONL):
  `analysis/xgb_live_trade_log/YYYY-MM-DD.jsonl` — one file per session
  with `session_start`, `scored` (top20 + per_seed_scores), `filtered`,
  `pick`, `buy_submitted`, `buy_filled`, `sell_submitted`, `session_end`.

### EXPECTED BEHAVIOR — healthy "no pick" vs silent bug

Because `--min-score 0.85` is designed to gate low-conviction days, the
expected baseline is **~44% of trading days result in a BUY, ~56% in no-op**.
The monitor must distinguish:

**HEALTHY no-pick signature (leave alone)**:
```
[xgb-live] Scoring 846 symbols...
[xgb-live] Conviction filter min_score=0.85: 0/602 candidates pass (top score=0.7036)
[xgb-live] No picks today — holding current positions (if any).
```
- `top score` is in range 0.55–0.84 (below gate but non-trivial).
- `n_candidates` > 400 (universe filters are firing normally).
- Per-seed scores in trade_log vary meaningfully across seeds (σ > 0.01).
- A few of these in a row is fine; base rate is 2–3 no-pick days in a row.

**BROKEN signatures (escalate)**:
- `top score` always exactly the same value across sessions → stale
  features (CSV pipeline stuck on one day, e.g. the known
  `trainingdata/train/` frozen-at-2026-04-10 bug — only matters if
  live_trader falls back to CSVs; normally it fetches fresh bars).
- `top score` collapses to near 0.5 (model random/broken) OR NaN.
- `n_candidates` = 0 (universe pull failed).
- `n_candidates` static for > 2 sessions (no-trading-day gate not firing on holiday).
- 10+ no-pick days in a row — possible signal drift, not just low conviction.
- Per-seed scores identical across ALL seeds → a single model being used,
  ensemble blend broken.

### Deploy gate (what must pass before swapping models or knobs)

**⚠ 2026-04-21 CORRECTION**: the numbers below were measured on the
**pre-stale-fix** `oos2024_ensemble_gpu` ensemble. A fresh retrain
(`oos2024_ensemble_gpu_fresh/`, trained 2026-04-21 07:49 UTC with the
loader-priority fix in place) produces a scoring distribution whose max
blended probability does not clear **0.75** on the true-OOS window
2025-01 → 2026-04-20 — every ms ≥ 0.75 cell in
`analysis/xgbnew_daily/sweep_20260421_fresh_vs/sweep_20260421_115402.json`
fires **zero** trades over 59 windows. Low-ms cells (ms 0.55–0.70) do fire
but all have negative median + p10 with neg_frac 17–75%. **0/108 cells
with positive median.** Same null result on `oos2025h1_ensemble_gpu_fresh`
(192-cell sweep per `project_xgb_true_oos_no_edge_2026_04_21.md`). Same
result on the `retrain_through_2026_03_20_ensemble` heldout sweep (0/42
positive at ms ≥ 0.65 over 8 windows 2026-03-21 → 2026-04-20). Every
prior "+141%/mo" claim was either in-sample (on `alltrain_ensemble_gpu`)
or pre-stale-fix (on pre-debe551d `oos2024_ensemble_gpu`).

**Interpretation**: the tariff-crash regime (2026-03 onwards) is not
predictable from 2020–2025H1 features with the current feature set. LIVE
being in hold-cash mode is the correct behavior. Don't lower the gate to
force trades — that would be a lossy bet.

**New deploy gate rule (2026-04-21)**: the bar is still "strict-dominance
on all four PnL metrics + intraday DD not worse by > 1pp", but it is now
measured on:

1. `oos2024_ensemble_gpu_fresh` (2024-12-31 cutoff, fresh-features retrain)
   across 59 windows 2025-01-02 → 2026-04-20.
2. `oos2025h1_ensemble_gpu_fresh` (2025-06-30 cutoff) across ~45 windows
   2025-07-01 → 2026-04-20.
3. Any new retrain-through ensemble's held-out window (currently
   `retrain_through_2026_03_20_ensemble` / `retrain_through_2026_02_28_ensemble`).

**A candidate becomes deploy-worthy only when at least ONE of these three
fresh true-OOS evaluations shows all of: median ≥ +27%/mo, p10 ≥ 0,
neg_frac ≤ 20%, worst-DD ≤ current LIVE worst-DD.** The old pre-fix
"+141%/mo" bar is retired until a positive-edge config is found on the
fresh ensembles.

Deployed LIVE config (archived pre-fix bar for reference, 60 windows
2025-01-02 → 2026-04-19 on pre-stale-fix `oos2024_ensemble_gpu`):

| metric | deploy fees | 36× fee stress |
|---|---:|---:|
| median %/mo (PRE-FIX, REFERENCE ONLY) | +141 | +108 |
| p10 (PRE-FIX) | +96 | +68 |
| n_neg / n_windows (PRE-FIX) | 0/60 | 0/60 |
| worst realized DD (PRE-FIX) | 7.18% | 8.34% |
| worst intraday DD (PRE-FIX) | 12.93% | 13.12% |

Keep using `xgbnew/sweep_ensemble_grid.py` with `--ensemble-sort-key
goodness` or `robG` as secondary, but always against a `_fresh` or
`retrain_through_*` ensemble dir.

⚠ **`alltrain_ensemble_gpu` is NOT OOS for filter validation** — it's
trained through today. Its "60-window" numbers are fully in-sample on the
validation window. For any filter-threshold tuning or feature-set change,
always re-eval on a `_fresh` or `retrain_through_*` ensemble. See
`feedback_alltrain_is_not_oos.md` and
`project_oos_ensembles_trained_pre_stale_fix.md`.

### Known data-pipeline hazard (2026-04-20)

`xgbnew/dataset.py::_load_symbol_csv` prefers `trainingdata/train/` (frozen
at 2026-04-10) over `trainingdata/` (fresh through 2026-04-15). The weekly
retrain and OOS sweeps are running on ~10-day-stale data. **Live inference
is UNAFFECTED** because `live_trader` fetches fresh bars from Alpaca,
not from the CSV loader. But the weekly retrain will produce a model
trained on stale ground truth until we flip the loader priority (or delete
the frozen `train/` subdir). See `project_xgb_stale_training_csvs.md`.

## 2. Daily RL Trader — STOPPED (intentional, kept for rollback)

- **Process**: `trade_daily_stock_prod.py --daemon --live` under supervisor
  unit `daily-rl-trader`. **STOPPED 2026-04-19 10:40 UTC** to free the
  singleton lock for XGB.
- **Do NOT start it** while `xgb-daily-trader-live` holds the lock — the two
  would race and one would crash on import.
- **Rollback procedure** (only on user's explicit direction):
  ```bash
  sudo supervisorctl stop xgb-daily-trader-live
  sudo supervisorctl start trading-server daily-rl-trader
  ```

## 3. Trading-server (broker boundary) — STOPPED

- Supervisor unit `trading-server` **STOPPED 2026-04-19 10:40 UTC**. It was
  the broker boundary for `daily-rl-trader`, now unused. Port 8050 is
  CLOSED — do NOT expect `curl http://127.0.0.1:8050/health` to work.
  The former expected `writer_lock_held_by_me=true` check is retired.
- Restart only as part of a daily-rl-trader rollback (see §2).

## 4. Other services (present but not primary signal)

- `llm-stock-trader` — Gemini-driven picks YELP/NET/DBX/OPTX. Owns its own
  singleton slot (account `llm_stock_writer`). Check supervisor; not urgent.
- `unified-orchestrator.service` — **DEAD** since 2026-03-29 (Gemini API
  exhausted). Do not attempt to revive without budget.
- Crypto RL (crypto12_ppo_v8) — training-snapshot only, **not OOS-validated**,
  do NOT deploy.

## 5. Realism invariants (marketsim truth vs prod truth)

- Ground truth for XGB swap decisions = `xgbnew/sweep_ensemble_grid.py` on
  the `oos2024_ensemble_gpu` 60-window grid with `fee_rate=0.0000278`
  (Alpaca real) and `fill_buffer_bps=5`.
- XGB uses `fee=0.278bps` to match Alpaca real. RL used `fee=10bps` and is
  out of prod; keep the two paths clearly separate in any re-eval.
- CPU/GPU hist drift: `tree_method=hist` on GPU drifts from CPU. Never
  promote a GPU-trained config without re-validating on CPU.
- Weekend/holiday gate: `xgbnew/live_trader._is_today_trading_day()` queries
  `/v2/clock` at top of both `run_session()` and `run_session_hold_through()`.
  No BUY will be placed on non-trading days even if a top score clears.

## 6. Where the hourly monitor's deploy authority starts and stops

**Auto-deploy authorized** (no human confirmation needed):
- Hot-swap the 5-seed ensemble for a 5-seed / 10-seed variant IFF the new
  variant beats the deploy gate in §1 on every metric (all four PnL +
  intraday DD).
- Restart a crashed `xgb-daily-trader-live` (its singleton + guard state
  are stable; supervisor autorestart is the happy path).
- Tweak `--min-dollar-vol` and `--min-vol-20d` inference floors if the
  adjusted value strict-dominates on the `oos2024_ensemble_gpu` sweep.

**Explicitly NOT auto-deploy** (human-only):
- Bump `--leverage` / `--allocation` (currently 2.0). Raising it further
  is a concentration-risk escalation.
- Lower `--min-score` below 0.85. (Would trade more days but widen DD.)
- Change `--top-n` above 1.
- Disable `--hold-through`.
- Rotate API keys.
- Any change to `alpaca_singleton.py` or the death-spiral guard (HARD RULE).
- Bring `daily-rl-trader` back online (that's a rollback, user only).

See `monitoring/hourly_prod_check_prompt.md` for the full per-phase procedure.
