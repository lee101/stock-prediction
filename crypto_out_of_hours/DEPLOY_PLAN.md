# Out-of-Hours Crypto Trader — Deploy Plan

## Summary
Extension of `crypto_weekend/` that trades crypto during ALL hours when NYSE
is closed, not just the weekend. The strategy:
  - Weekend: buy top-1 crypto with trend filter (sma_mult=1.0, vol≤3%).
  - Weekday overnight: buy top-2 crypto with tight trend filter
    (sma_mult=1.05, mom_vs_sma≥0%, vol≤2%).
  - Exits at every NYSE market_open so never overlaps `xgb-daily-trader-live`.

## 120d Marketsim Result (2025-12-17 → 2026-04-15, 81 sessions, fee=10bps)

| Strategy | monthly% | p10% | neg% | max_dd% | cum% | trades |
|---|---|---|---|---|---|---|
| MVP weekend-only (BTC+ETH+SOL, same 120d) | **−0.61** | −1.12 | 11.8 | −5.64 | −2.46 | 4/17 |
| OOH weekend-only (10-crypto, tight trend) | +2.09 | 0.00 | 1.2 | −1.66 | +8.47 | 6/81 |
| **OOH combined (weekend + weekday overlay)** | **+2.71** | **0.00** | **3.7** | **−1.66** | **+11.11** | 12/81 |

**Strict-beat MVP on both `monthly PnL` and `max_dd`**:  PASS.

Fee robustness: combined is positive at 10bps (+2.71%/mo) and 20bps (+2.11%/mo),
weakens at 50bps (+0.29%/mo). Strict-beats MVP at 10, 20, 50bps.

Cross-regime (6 non-target 120d windows): combined > weekend-only monthly PnL
in 6/9 windows; strict-beats in 2/9. The weekday overlay is positive in recent
regimes and fee-efficient at 10bps but widens DD modestly in older volatile
periods. Ship both modes and default to `--weekend-only` if operator prefers
the safest profile.

## Files
- `sessions.py` — NYSE-calendar-based session enumeration (DST + holiday aware).
- `ooh_backtest.py` — per-session panel + signal apply + PnL + summarize.
- `backtest_grid.py` — wider grid sweep (trend + mean-reversion variants).
- `find_combined.py` — combined-strategy search.
- `validate_combined.py` — multi-window robustness check.
- `final_report.py` — side-by-side table at 10/20/50 bps.
- `live_trader.py` — skeleton with `--dry-run` preview of next 14 days.
- `test_strategy.py` — 8 regression tests (all green).

## Deploy Checklist (if operator approves)
1. Create separate Alpaca subaccount (`ALPACA_ACCOUNT_NAME=crypto-out-of-hours`).
2. Register new supervisor unit in `scripts/deploy_live_trader.sh` `LIVE_WRITER_UNITS`
   (HARD RULE #2 — new entry points must be in same commit).
3. Wire `alpaca_wrapper` crypto buy/sell in `live_trader.py` (currently skeleton).
4. Add session-state persistence at `<state>/crypto_out_of_hours/` for:
   - last session we traded (idempotent on restart)
   - buy_prices (so existing death-spiral guard applies; HARD RULE #3).
5. Create `deployments/crypto-out-of-hours-live/launch.sh`:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   cd /nvme0n1-disk/code/stock-prediction
   source .venv/bin/activate
   export ALPACA_ACCOUNT_NAME=crypto-out-of-hours
   export GOOGLE_APPLICATION_CREDENTIALS=secrets/google-credentials.json
   # default: weekend-only mode (safest; matches 120d strict-beat bar)
   exec python -u crypto_out_of_hours/live_trader.py \
       --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT AVAXUSDT LINKUSDT LTCUSDT DOTUSDT \
       --max-gross 1.0 \
       --weekend-only \
       --poll-seconds 60 \
       --log-file /nvme0n1-disk/code/stock-prediction/deployments/crypto-out-of-hours-live/trader.log
   ```
6. Toggle to combined mode (drop `--weekend-only`) only after:
   - 30 days of weekend-only LIVE with realised PnL matching sim.
   - Operator review of weekday-overnight fire-rate on LIVE data.
7. Update `alpacaprod.md` with status + validated marketsim score.

## Safety Invariants
- NEVER hold crypto position during NYSE open. Enforced by:
  - Session-bounds check in `compute_session_picks`.
  - `--weekend-only` degrades gracefully (never fires weekday).
  - `live_trader.py` must check NYSE state every poll; if NYSE transitions
    to open while we hold, emit closing market order.
- Additive with `xgb-daily-trader-live` (separate Alpaca account).
- Death-spiral guard (HARD RULE #3) applies if wired through `alpaca_wrapper`.

## Not In Scope
- Hourly XGB classifier / PufferLib RL — the simple trend+vol filter already
  strict-beats MVP at 10bps; no need for ML complexity in this 120d window.
  If/when the filter stops firing in a new regime, reconsider.
