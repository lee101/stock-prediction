# Current Algorithms — Monitor Ledger

**Authoritative source**: `alpacaprod.md` top section is always the canonical
live ledger. This file is a short pointer for the hourly monitor so it knows
which services should exist, where the best config lives, and what the bar
is to beat. Keep this file in sync with `alpacaprod.md` whenever we deploy.

Last synced: 2026-04-19 10:40 UTC — **XGB deploy swap**

---

## 1. XGB Daily Trader — LIVE (primary, as of 2026-04-19)

- **Process**: `xgbnew.live_trader` (supervisor unit `xgb-daily-trader-live`,
  launcher `deployments/xgb-daily-trader-live/launch.sh`). NOT systemd.
- **Broker boundary**: **direct Alpaca SDK** — no `trading_server` on
  `127.0.0.1:8050` anymore. Writes go straight to `api.alpaca.markets` using
  `ALP_KEY_ID_PROD` / `ALP_SECRET_KEY_PROD`.
- **Singleton**: `src/alpaca_singleton.py::enforce_live_singleton` fires at
  startup; the live-writer lock at `strategy_state/account_locks/alpaca_live_writer.lock`
  must show `service_name: xgb_live_trader` and a `pid` that matches
  `supervisorctl status xgb-daily-trader-live`.
- **HARD RULE #3 (death-spiral guard)**: `xgbnew/live_trader.py` calls
  `record_buy_price(sym, fill_px)` after each BUY and
  `guard_sell_against_death_spiral(sym, "sell", current_price)` before each
  SELL. RuntimeError from the guard crashes the loop by design; supervisor
  autorestart handles reconnection. DO NOT "catch and skip" this exception.
- **Ensemble**: 5-seed alltrain GPU —
  `analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed{0,7,42,73,197}.pkl`,
  blend mode `mean` (predict_proba averaged). Models trained with
  `n_estimators=400 max_depth=5 learning_rate=0.03` on 2020-01-01 → 2026-04-19
  (846-symbol universe).
- **Allocation**: `--allocation 0.25` (25% of portfolio) per pick, `--top-n 1`
  (1 symbol per day), `--leverage 1.0` bare. Queued upgrade: `lev=1.25`
  (+10.9pp median in-sample but +6.9pp worst DD; **human-approval only**).
- **Baseline expected monthly return (IN-SAMPLE 2025-01-02 → 2026-04-10)**:
  median +38.85%, p10 +4.82%, sortino 18.86, neg 2/30, worst DD 31.44%.
  Real PnL should track **60–70% of in-sample** → **~+25%/mo healthy,
  <+10%/mo for 2 weeks = broken**.
- **Character**: buy-open (9:30 ET) / sell-close (15:50 ET) same session,
  flat overnight. Top-1 pick is ~$7K notional on $28K equity.
- **Signal cadence**: one session per trading day. `--loop` sleeps to
  09:20 ET next business day between sessions.
- **Logs**: `sudo tail -f /var/log/supervisor/xgb-daily-trader-live.log` and
  `…-error.log` (singleton lock + stderr).

### Deploy gate (what must pass before swapping models or knobs)

On same 846-symbol OOS grid (30 windows 2025-01-02 → 2026-04-10, fb=5bps,
fee=0.278bps, binary fills at `decision_lag=2`):
- **5-seed top_n=1 lev=1.0 (deployed)**: med +38.85 / p10 +4.82 / sortino 18.86 / neg 2/30 / dd 31.44
- Never downgrade on median ≥ +35 AND p10 ≥ +4 AND neg ≤ 2/30 AND worst_dd ≤ 32 simultaneously.

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

- Ground truth for XGB swap decisions = `xgbnew/eval_pretrained.py` on 30
  windows 2025-01-02 → 2026-04-10 with `fee_rate=0.0000278` (Alpaca real) and
  `fill_buffer_bps=5`. These match the live fill model.
- XGB uses `fee=0.278bps` to match Alpaca real. RL used `fee=10bps` and is
  out of prod; keep the two paths clearly separate in any re-eval.
- CPU/GPU hist drift: `tree_method=hist` on GPU drifts from CPU. Never
  promote a GPU-trained config without re-validating on CPU.

## 6. Where the hourly monitor's deploy authority starts and stops

**Auto-deploy authorized** (no human confirmation needed):
- Hot-swap the 5-seed ensemble for a 5-seed / 10-seed variant IFF the new
  variant beats the deploy gate in §1 on every metric.
- Restart a crashed `xgb-daily-trader-live` (its singleton + guard state
  are stable; supervisor autorestart is the happy path).

**Explicitly NOT auto-deploy** (human-only):
- Bump `--leverage` above 1.0 (even to 1.25).
- Increase `--allocation` above 0.25.
- Change `--top-n` above 1.
- Rotate API keys.
- Any change to `alpaca_singleton.py` or the death-spiral guard (HARD RULE).
- Bring `daily-rl-trader` back online (that's a rollback, user only).

See `monitoring/hourly_prod_check_prompt.md` for the full per-phase procedure.
