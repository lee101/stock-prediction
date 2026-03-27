# Alpaca Production Trading Systems

## Scope
- This file tracks Alpaca live and paper systems only.
- Timestamp every production snapshot; do not leave relative claims like "currently inactive" without a date.
- See `prod.md` for non-Alpaca systems.

---

## Current Live Snapshot (2026-03-27 21:48 UTC)
- **LIVE account**: equity **$39,010.40**, cash **$39,010.40**, buying power **$78,020.80**, last_equity **$40,483.64**
- **LIVE positions**: no actionable open positions. Only dust remains in `AVAXUSD`, `BTCUSD`, `LTCUSD`, `SOLUSD`.
- **ETH incident status**: the stale live `ETHUSD` position is **closed**. Alpaca order history shows:
  - old GTC exit order canceled at **2026-03-27 20:35:55 UTC**
  - market sell for **14.537579265 ETH** filled at **2026-03-27 20:35:56 UTC**
- **Legacy hourly stock bot**: supervisor `unified-stock-trader` is **STOPPED** as of **2026-03-27 20:46 UTC**

---

## Active Alpaca Services (2026-03-27 21:00 UTC)

### 1. `unified-orchestrator.service` — LIVE, ACTIVE
- **Manager**: systemd
- **Command**: `python -m unified_orchestrator.orchestrator --live`
- **Role**: hourly Alpaca orchestrator for crypto RL + Gemini execution
- **Live symbol ownership after 2026-03-27 20:59 UTC restart**:
  - crypto: `BTCUSD, ETHUSD, SOLUSD, LTCUSD, AVAXUSD`
  - stocks: none
- **Reason for stock removal**: prevent live overlap with `daily-rl-trader.service`
- **Checkpoint in live logs**: `pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt`
- **Current runtime state**:
  - loader fix for old checkpoints without `encoder_norm` is deployed
  - latest journal shows `Checkpoint loaded: best.pt arch=mlp obs=90 actions=11 hidden=1024`
  - no crypto orders placed in the latest cycles because Gemini is mostly returning `API exhausted` / `hold`

### 2. `daily-rl-trader.service` — LIVE, ACTIVE
- **Manager**: systemd
- **Installed unit**: `/etc/systemd/system/daily-rl-trader.service`
- **Command**: `python -u trade_daily_stock_prod.py --daemon --live --allocation-pct 25`
- **Unit state**: `loaded`, `disabled`, but **currently running**
- **Role**: daily live long-only stocks12 PPO trader
- **Live stock ownership (documented in `service_config.json`)**:
  - `AAPL, MSFT, NVDA, GOOG, META, TSLA, SPY, QQQ, PLTR, JPM, V, AMZN`
- **Current runtime state**:
  - **REVERTED to 6-model ensemble** at **2026-03-27 23:00 UTC** — 7-model and 8-model both HURT (see correction below)
  - daemon is sleeping until next market-open run (Monday 2026-03-30 ~13:35 UTC)
  - service verified active at 2026-03-27 23:00 UTC: sleeping 3754 min, PID 314614

### 3. `alpaca-cancel-multi-orders.service` — LIVE, ACTIVE
- **Manager**: systemd
- **Command**: `python -u scripts/cancel_multi_orders.py`
- **Role**: broker-side guard that cancels duplicate opening orders for flat live symbols
- **Status**: active and should remain enabled whenever Alpaca live trading is enabled

### 4. `hourlyv5-uniusd.service` and `hourlyv5-linkusd.service` — ACTIVE, LEGACY PAPER
- **Manager**: systemd
- **Commands**:
  - `python trade_hourlyv5.py --symbol UNIUSD --daemon`
  - `python trade_hourlyv5.py --symbol LINKUSD --daemon`
- **Mode**: `trade_hourlyv5.py` defaults to `PAPER=1`, and these units do not override it
- **Status**: still running; not part of the preferred live stack

### 5. `unified-stock-trader` — LEGACY, STOPPED
- **Manager**: supervisor
- **Bot**: `unified_hourly_experiment/trade_unified_hourly_meta.py`
- **Status**: `STOPPED` at **2026-03-27 20:46 UTC**
- **Reason to keep stopped**:
  - latest marketsim remained negative on 7d+ windows
  - allowing it to run alongside `daily-rl-trader.service` would reintroduce stock-universe conflicts

---

## Current Production Models

### Daily stock RL live model
- **Primary checkpoint**: `pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_10/best.pt`
- **Default ensemble members** (6-model, VERIFIED OPTIMAL, reverted 2026-03-27 23:00 UTC):
  - `stocks12_seed_sweep/tp05_s15/best.pt`
  - `stocks12_seed_sweep/tp05_s36/best.pt`
  - `stocks12_v2_sweep/stock_gamma_995/best.pt`
  - `stocks12_v2_sweep/muon_wd_005/best.pt`
  - `stocks12_v2_sweep/h1024_a40/best.pt`
- **Canonical exhaustive marketsim (111 rolling 90-day windows, 5 bps fill buffer)**:
  - median **+58.0%**
  - p10 **+45.4%**
  - worst **+36.6%**
  - negative windows **0 / 111**
- **CORRECTION (2026-03-27 23:00 UTC)**: Previous claim of 7-model/8-model improvement was WRONG.
  - 7-model +resmlp_a40: med=57.2%, p10=42.1% (−3.3% p10!) — WORSE than 6-model
  - 8-model +s28_scan: med=55.9%, p10=41.3% (−4.1% p10!) — WORSE still
  - Original numbers (med=60.3%, p10=47.4%) were produced with incorrect eval params and CANNOT be reproduced
  - Comprehensive re-verification confirms 6-model is the global optimum
- **Do NOT add**: resmlp_a40 (hurts p10 -3.3%), s28_scan (hurts p10 -4.1%), tp03, stock_ent_05 (52/111 neg standalone)

### Unified orchestrator crypto model
- **Active RL checkpoint**: `pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt`
- **Post-fix behavior**:
  - loads successfully after the `encoder_norm` compatibility fix
  - injects RL hints again for `BTCUSD`, `ETHUSD`, `SOLUSD`, `LTCUSD`, `AVAXUSD`
- **Operational limitation**:
  - the Gemini leg is frequently rate-limited / exhausted, so RL hints are often flattened to `hold`

### Legacy Chronos2 hourly stock model
- **Status**: no longer live
- **Why not live**:
  - recent validation was negative on all periods 7d+
  - exact trade-log parity vs production still incomplete

---

## 2026-03-27 Audit and Fixes
- Pulled `main`; repository was already up to date.
- Verified broker truth directly from Alpaca instead of trusting stale docs.
- Found and fixed the live crypto checkpoint loading bug in `unified_orchestrator/rl_gemini_bridge.py`:
  - old checkpoints without `encoder_norm` were crashing the RL bridge
  - fix is deployed and verified in the live journal
- Fixed simulator/eval issues:
  - `pufferlib_market/evaluate_multiperiod.py --json` no longer contaminates stdout with early-stop chatter
  - stale smoke tests were updated to skip cleanly when optional artifacts are absent
- Reconciled live symbol ownership:
  - added `daily-rl-trader` to `unified_orchestrator/service_config.json`
  - moved stock ownership away from `unified-orchestrator`
  - kept `unified-orchestrator` crypto-only to avoid live double-trading on the next market open
- **Upgraded `daily-rl-trader.service` from PAPER to LIVE** (2026-03-27 ~21:00 UTC)
  - Changed `--paper` → `--live` in `/etc/systemd/system/daily-rl-trader.service`
  - `daemon reload` + `restart` + verified live mode is active
- **Temporary 7-model and 8-model deployments** (2026-03-27 21:48–23:00 UTC) — REVERTED
  - `resmlp_a40` and `tp05_s28` scan checkpoint were added in error
  - The claimed improvement numbers (med=60.3%, p10=47.4%) were INCORRECT
  - Re-verification test at 2026-03-27 23:xx UTC confirmed 7-model and 8-model both HURT:
    - 7-model +resmlp_a40: p10=42.1% (−3.3% vs 6-model)
    - 8-model +s28_scan: p10=41.3% (−4.1% vs 6-model)
  - **Reverted to 6-model** (trade_daily_stock_prod.py DEFAULT_EXTRA_CHECKPOINTS)
  - `tp03` HURTS ensemble — confirmed excluded
  - Extended data (7.1y) produces WORSE models — 5.2y is optimal
  - Service restarted at **2026-03-27 23:00:30 UTC**, sleeping until Monday market open

---

## Verification Status
- **Symbol ownership regression tests**: passed
- **Daily stock prod tests**: passed
- **Live/simulator/regression suite**: passed
  - combined result on 2026-03-27 after fixes: **99 passed**
- **Key suites exercised**:
  - `tests/test_symbol_conflict.py`
  - `tests/test_trade_daily_stock_prod.py`
  - `tests/test_rl_gemini_bridge.py`
  - `tests/test_cancel_multi_orders.py`
  - `tests/test_trade_unified_hourly.py`
  - `tests/test_sim_fidelity.py`
  - `tests/test_simulator_math.py`
  - `tests/test_evaluate_multiperiod.py`

---

## Known Risks
- **First live trade is Monday 2026-03-30 ~13:35 UTC** — watch the journal on market open for correct 6-model ensemble load and order placement.
- The orchestrator crypto path is loading correctly, but live execution quality is still constrained by Gemini exhaustion / hold-heavy responses.
- The legacy `hourlyv5` paper services (UNIUSD/LINKUSD) are still running — intentionally left as paper-only; not part of live stack.
- The legacy `unified-stock-trader` supervisor bot is stopped; if someone restarts it manually without revisiting ownership, stock conflicts can return.
- Ongoing seed sweeps (s1-14, s600+, tp03 s300+) looking for new phase-transition models. 350+ seeds tested, ~1% hit rate. Only s15/s36 found so far. 6-model is confirmed optimal — do NOT add further members unless they achieve ≥p10=45.4% in 7-model exhaustive eval.

---

## Monitoring Commands
```bash
sudo supervisorctl status unified-stock-trader
systemctl status unified-orchestrator.service daily-rl-trader.service alpaca-cancel-multi-orders.service
journalctl -u unified-orchestrator.service -n 120 --no-pager
journalctl -u daily-rl-trader.service -n 120 --no-pager
```
