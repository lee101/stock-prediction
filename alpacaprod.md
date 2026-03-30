# Alpaca Production Trading Systems

## Scope
- This file tracks Alpaca live and paper systems only.
- Timestamp every production snapshot; do not leave relative claims like "currently inactive" without a date.
- See `prod.md` for non-Alpaca systems.

---

## Current Live Snapshot (2026-03-29 05:22 UTC)
- **LIVE account**: equity **$38,719.38**, cash **$38,719.38**, buying power **$77,438.76**, last_equity **$39,090.40**
- **LIVE positions**: no actionable open positions. Only dust remains in `AVAXUSD`, `BTCUSD`, `LTCUSD`, `SOLUSD`.
- **LIVE open orders**: none
- **Crypto status**: no stranded `ETHUSD` position remains; latest orchestrator journal at **2026-03-29 05:01 UTC** shows all crypto symbols on `hold`.
- **Daily stock live path**: exact prod dry-run `trade_daily_stock_prod.py --once --dry-run --live` completed successfully at **2026-03-29 04:50 UTC** and produced `long_GOOG` at **14.4%** confidence.
- **Daily stock restart status**: the long-only action-decode fix is on disk, but `daily-rl-trader.service` has **not** been restarted since the fix; `systemctl restart daily-rl-trader.service` still requires interactive authentication from this shell.
- **Legacy hourly stock bot**: supervisor `unified-stock-trader` is still **ACTIVE** as PID `153050`, and its runtime command still includes overlapping stock symbols that conflict with `daily-rl-trader.service`

---

## Active Alpaca Services (2026-03-28 09:29 UTC)

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
  - a local patch now exists to fall back to RL-only plans when Gemini is exhausted, but that patch is **not live** until `unified-orchestrator.service` is restarted

### 2. `daily-rl-trader.service` — LIVE, ACTIVE
- **Manager**: systemd
- **Installed unit**: `/etc/systemd/system/daily-rl-trader.service`
- **Command**: `python -u trade_daily_stock_prod.py --daemon --live --allocation-pct 25`
- **Unit state**: `loaded`, `disabled`, but **currently running**
- **Role**: daily live long-only stocks12 PPO trader
- **Live stock ownership (documented in `service_config.json`)**:
  - `AAPL, MSFT, NVDA, GOOG, META, TSLA, SPY, QQQ, PLTR, JPM, V, AMZN`
- **Current runtime state**:
  - daemon is sleeping until next market-open run (Monday 2026-03-30 ~13:35 UTC)
  - current process has been active since **2026-03-29 02:14:48 UTC**
  - on-disk live path now verifies cleanly on the full 17-model `prod_ensemble` stack; exact dry-run passed at **2026-03-29 04:50 UTC**
  - current in-memory daemon still needs a restart before Monday open to pick up the 2026-03-29 long-only decode fix

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

### 5. `unified-stock-trader` — LIVE, ACTIVE, CONFIG-DRIFTED
- **Manager**: supervisor
- **Bot**: `unified_hourly_experiment/trade_unified_hourly_meta.py`
- **Status**: running as PID `153050`
- **Current runtime symbol list**: `NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV`
- **Owned symbol list (`service_config.json`)**: `DBX,TRIP,MTCH,NYT,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV`
- **Installed supervisor config (2026-03-28 09:29 UTC)**: updated to the owned 11-symbol list above, but supervisor has **not** been reloaded/restarted yet, so the live process still carries the old overlapping 18-symbol command
- **Why this is still a problem**:
  - latest marketsim remains negative on 7d+ windows
  - replay parity on the recent live `TSLA` + `ABEV` entries is now exact on count and quantity after fixing the replay harness to infer live `execute_trades_start` sizing context; remaining mismatch is price MAE (`0.3265`), not missing or mis-sized entries
  - allowing the current process to persist into Monday open would overlap `daily-rl-trader.service` on `AAPL/MSFT/NVDA/GOOG/META/TSLA/PLTR`

---

## Current Production Models

### Daily stock RL live model
- **Primary checkpoint**: `pufferlib_market/prod_ensemble/tp10.pt`
- **Default ensemble members**: `tp10 + s15 + s36 + gamma_995 + muon_wd_005 + h1024_a40 + s1731 + gamma995_s2006 + s1401 + s1726 + s1523 + s2617 + s2033 + s2495 + s1835 + s2827 + s2722`
- **Documented prod selection note in `trade_daily_stock_prod.py` (current on-disk truth)**:
  - 17-model screen-phase ensemble
  - **0 / 111** negative windows
  - median **+55.6%**
  - p10 **+41.2%**
  - worst **+22.5%**
- **Live-path fix applied 2026-03-29**:
  - stock checkpoints are still 25-action short-capable heads
  - long-only inference now masks the short action block before argmax / softmax
  - the exact prod dry-run no longer crashes on out-of-range action decode
- **Operational note**: the fix is verified on disk and in dry-run, but not yet loaded by the sleeping `daily-rl-trader.service` process because restart requires interactive auth

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
- **Daily stock live-path dry-run**: passed
  - `trade_daily_stock_prod.py --once --dry-run --live` completed successfully at **2026-03-29 04:50 UTC**
- **Inference + daily stock targeted tests**: passed
  - `tests/test_pufferlib_market_inference.py`
  - `tests/test_trade_daily_stock_prod.py`
  - result: **20 passed**
- **Broader training / dispatch / WandB regression slice**: passed
  - `tests/test_pufferlib_market_inference.py`
  - `tests/test_trade_daily_stock_prod.py`
  - `tests/test_pufferlib_market_train_logging.py`
  - `tests/test_dispatch_rl.py`
  - `tests/test_dispatch_multiseed.py`
  - `tests/test_pufferlib_market_autoresearch_rl.py`
  - `tests/test_autoresearch_stock.py`
  - result: **144 passed**

---

## Known Risks
- **First live stock trade is Monday 2026-03-30 ~13:35 UTC** — restart `daily-rl-trader.service` before then so the daemon loads the 2026-03-29 stock decode fix.
- The orchestrator crypto path is loading correctly, but live execution quality is still constrained by Gemini exhaustion / hold-heavy responses.
- The local fix for Gemini exhaustion -> RL-only fallback has passed targeted tests, but it is not live until the orchestrator is restarted.
- The legacy `hourlyv5` paper services (UNIUSD/LINKUSD) are still running — intentionally left as paper-only; not part of live stack.
- The live `unified-stock-trader` supervisor bot is running on an old overlapping symbol list. The on-disk supervisor config is corrected, but a controlled reload/restart is still required before Monday open.
- Current quick probes still show seed sensitivity. The latest 3-seed screen on 2026-03-29 found no new config with `0%` negative-seed rate; best partial candidates were `stock_obs_norm_tp05` and `stock_wd_05`, both still at **33%** negative-seed rate.

---

## Monitoring Commands
```bash
sudo supervisorctl status unified-stock-trader
systemctl status unified-orchestrator.service daily-rl-trader.service alpaca-cancel-multi-orders.service
journalctl -u unified-orchestrator.service -n 120 --no-pager
journalctl -u daily-rl-trader.service -n 120 --no-pager
```
