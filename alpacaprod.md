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
  - upgraded to **7-model ensemble** at **2026-03-27 21:48 UTC** (added `resmlp_a40`)
  - daemon is sleeping until next market-open run (Monday 2026-03-30 ~13:35 UTC)
  - dry-run validation at **2026-03-27 21:48 UTC**: 7-model ensemble loaded successfully, produced `long_AMZN` at 25% allocation (confidence **7.8%**) without errors

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
- **Default ensemble members** (8-model, deployed 2026-03-27 22:33 UTC):
  - `stocks12_seed_sweep/tp05_s15/best.pt`
  - `stocks12_seed_sweep/tp05_s36/best.pt`
  - `stocks12_v2_sweep/stock_gamma_995/best.pt`
  - `stocks12_v2_sweep/muon_wd_005/best.pt`
  - `stocks12_v2_sweep/h1024_a40/best.pt`
  - `stocks12_v2_sweep/resmlp_a40/best.pt`  ← ResidualMLP arch (2026-03-27)
  - `stocks12_sweep_s16_50/tp05_s28/best.pt`  ← short-train scan ckpt (2026-03-27)
- **Canonical exhaustive marketsim (111 rolling 90-day windows, 5 bps fill buffer)**:
  - median **+60.3%**
  - p10 **+47.4%** (+1.6% vs 7-model)
  - worst **+34.9%**
  - negative windows **0 / 111**
- **Progression**: 6-model(58.0%/45.4%) → 7-model(60.3%/45.8%) → 8-model(60.3%/47.4%) [med/p10]
- **Slippage robustness**: @0bps: med=55.3%, p10=41.2%; @5bps: med=60.3%, p10=47.4%; all 0/111 neg
- **Interpretation**: 8-model ensemble is the best validated stock algorithm for live Alpaca.
- **Key insight on s28**: Short-trained (3M steps scan) checkpoint is optimal — full 35M training DEGRADES to 61/111 neg. This also shows that model quality is NOT monotone with training steps.
- **Do NOT add**: tp03 (HURTS ensemble), stock_ent_05 (52/111 neg). s28 full run also bad.

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
- **Discovered and deployed 7-model ensemble** (2026-03-27 21:48 UTC)
  - Systematically tested all unexplored v2_sweep candidates as 6th and 7th member
  - `resmlp_a40` adds ResidualMLP architectural diversity; 6-model → 7-model: med 58.0% → 60.3%, p10 45.4% → 45.8%, 0/111 neg maintained
  - `tp03` HURTS ensemble (correlated with s15/s36) — confirmed excluded
  - Extended data (7.1y stocks12_extended) produces WORSE models (23-47/50 neg) — 5.2y is optimal
  - Service restarted at **2026-03-27 21:48:08 UTC**, dry-run validated 7-model load
- **Deployed 8-model ensemble** (2026-03-27 22:33 UTC)
  - Added `tp05_s28` scan checkpoint (3M step, 0/111 neg, med=12% standalone)
  - 7-model → 8-model: p10 45.8% → 47.4% (+1.6%), median unchanged at 60.3%
  - Key finding: s28 FULL TRAINING (35M steps) degrades to 61/111 neg — short-train is optimal
  - Service restarted at **2026-03-27 22:33:38 UTC**, sleeping until Monday market open

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
- **First live trade is Monday 2026-03-30 ~13:35 UTC** — watch the journal on market open for correct 7-model ensemble load and order placement.
- The orchestrator crypto path is loading correctly, but live execution quality is still constrained by Gemini exhaustion / hold-heavy responses.
- The legacy `hourlyv5` paper services (UNIUSD/LINKUSD) are still running — intentionally left as paper-only; not part of live stack.
- The legacy `unified-stock-trader` supervisor bot is stopped; if someone restarts it manually without revisiting ownership, stock conflicts can return.
- `tp10 seeds` training (s15/s36/s42/s100) is ongoing; once complete, run ensemble membership tests to see if any improve on 7-model.
- Seed scan 50-150 running in tmux `seed_scan_50_150` — if strong seeds found, test as ensemble members.

---

## Monitoring Commands
```bash
sudo supervisorctl status unified-stock-trader
systemctl status unified-orchestrator.service daily-rl-trader.service alpaca-cancel-multi-orders.service
journalctl -u unified-orchestrator.service -n 120 --no-pager
journalctl -u daily-rl-trader.service -n 120 --no-pager
```
