# Archive: XGB alltrain 5-seed ensemble LIVE @ 10:40 UTC (lev=1.0 bare)

Superseded 2026-04-19 by full-stack activation (hold_through + min_score 0.85 + lev 2.0).
Original state preserved below.

---

### 2026-04-19 10:40 UTC — XGB alltrain 5-seed ensemble LIVE (supervisor: `xgb-daily-trader-live`)

**First live XGB deploy.** Supervisor pid 2343601 holds the
`alpaca_live_writer` singleton lock. Session loops indefinitely; first
trade session Monday 2026-04-20 09:30 ET.

| field | value |
|---|---|
| supervisor unit | `xgb-daily-trader-live` (NOT systemd) |
| launcher | `deployments/xgb-daily-trader-live/launch.sh` |
| models | `analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed{0,7,42,73,197}.pkl` |
| blend | mean over 5 seeds (predict_proba averaged) |
| universe | `symbol_lists/stocks_wide_1000_v1.txt` (846 tradable) |
| top_n | 1 |
| allocation | 25% of portfolio |
| min dollar vol | $5M |
| leverage | 1.0 (bare; knee is 1.25 per `project_xgb_leverage_sweet_spot.md`) |
| paper | **False — LIVE** (`ALP_PAPER=0 ALLOW_ALPACA_LIVE_TRADING=1`) |
| account equity | $28,679 |
| Alpaca path | direct SDK (singleton lock + per-call death-spiral guard) |

**Replaces**: `daily-rl-trader` + `trading-server` (both stopped
10:40 UTC). The RL ensemble delivered +7.47%/mo median; XGB OOS is
+32.48%/mo median 0/34 neg on the same period.

**Safety (HARD RULE #3)**: `xgbnew/live_trader.py` now calls
`record_buy_price(sym, fill_px)` after each BUY and
`guard_sell_against_death_spiral(sym, "sell", current_price)` before each
SELL. Guard RuntimeError propagates (crashes the loop; supervisor
autorestart). Tests at `tests/test_xgbnew_live_trader_guard.py` (10/10 green).

**Trading-day gate added 2026-04-19 14:45 UTC** (commit 001686f8). The
Saturday 10:40 UTC deploy submitted a MSFT DAY BUY at 13:30 UTC — Alpaca
accepted and queued the order for Monday's open (`status: accepted,
expires_at: 2026-04-20T20:00:00Z`). Would have double-bought on Monday
alongside the fresh Monday pick. Cancelled manually (204, filled_qty=0,
no fill). `_is_today_trading_day()` now queries `/v2/clock` at top of
`run_session()` and skips the session when the market won't open today.
See `project_alpaca_weekend_day_orders.md` in memory.

**Monitor**: `sudo tail -f /var/log/supervisor/xgb-daily-trader-live.log`
or the singleton check `cat strategy_state/account_locks/alpaca_live_writer.lock`.

**Rollback**: `sudo supervisorctl stop xgb-daily-trader-live &&
sudo supervisorctl start trading-server daily-rl-trader`.

**Deploy-config baseline (in-sample, 2025-01-02 → 2026-04-10, 30d × 14d stride windows, 30 windows)**:

| config | median %/mo | p10 | sortino | worst DD | neg |
|---|---:|---:|---:|---:|---:|
| **deployed: 5-seed top_n=1 lev=1.0** | **+38.85** | +4.82 | 18.86 | 31.44 | 2/30 |
| 10-seed top_n=1 lev=1.0 | +38.05 | +4.74 | 19.63 | 31.62 | 2/30 |
| 5-seed top_n=1 lev=1.25 | +49.73 | +5.44 | 18.79 | 38.35 | 2/30 |
| 5-seed top_n=2 lev=1.0 | +34.24 | +5.12 | 18.06 | 31.58 | 3/30 |

Notes:
- **In-sample** (alltrain train through 2026-04-19 includes full OOS grid). Use as upper-bound, not OOS.
- 10-seed is NOT strictly better — sortino +0.77 but median −0.80; don't hot-swap.
- lev=1.25 is the leverage knee: +10.88 med but +6.91 worst DD. Flip via user approval.
- top_n=1 still dominates 4/6 metrics vs top_n=2. Don't change.
- Artifacts: `analysis/xgbnew_deploy_baseline/deploy_{5seed,10seed}_lev{1,125}_top{1,2}.json`.
