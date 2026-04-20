# Current Algorithms — Monitor Ledger

**Authoritative source**: `alpacaprod.md` top section is always the canonical
live ledger. This file is a short pointer for the hourly monitor so it knows
which services should exist, where the best config lives, and what the bar
is to beat. Keep this file in sync with `alpacaprod.md` whenever we deploy.

Last synced: **2026-04-20 13:17 UTC — vol-floor tightened 0.10→0.12 (strict-dom OOS); lev=2.0, ms=0.85, hold-through, 50M/0.12 inference floors**

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

On 846-symbol OOS grid using `oos2024_ensemble_gpu` (2024-12-31 cutoff; this
is the TRUE-OOS ensemble, not `alltrain_ensemble_gpu` which is retrained
through today) across 60 windows 2025-01-02 → 2026-04-19, fb=5bps fee=0.278bps
binary fills `decision_lag=2`, at `lev=2.0 ms=0.85 ht=1 top_n=1`:

| metric | deploy fees | 36× fee stress |
|---|---:|---:|
| median %/mo | **+141** | **+108** |
| p10 | **+96** | **+68** |
| n_neg / n_windows | **0/60** | **0/60** |
| worst realized DD | **7.18%** | **8.34%** |
| worst intraday DD | **12.93%** | **13.12%** |

A candidate replaces the current ensemble **only if it meets all five
simultaneously** (all four PnL metrics + intraday DD not worse by > 1pp).
Use `xgbnew/sweep_ensemble_grid.py` on the `oos2024_ensemble_gpu` artifact
directory with `--ensemble-sort-key goodness` or `robG` as secondary.

⚠ **`alltrain_ensemble_gpu` is NOT OOS for filter validation** — it's
trained through today. Its "60-window" numbers are fully in-sample on the
validation window. For any filter-threshold tuning or feature-set change,
always re-eval on `oos2024_ensemble_gpu`. See
`feedback_alltrain_is_not_oos.md`.

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
