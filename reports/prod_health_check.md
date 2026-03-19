# Production Health Check -- 2026-03-19

## Overall Assessment: DEGRADED

The Gemini API quota is exhausted for binance-hybrid-spot, blocking all trading decisions since ~2026-03-19 00:19 UTC. Cache services and worksteal-daily are healthy. No positions are at risk (dust only), but the bot cannot enter new trades until API quota resets.

---

## Trading Services

### binance-hybrid-spot
- **Status:** RUNNING (pid 2609036, uptime 1d 3h)
- **Health:** DEGRADED -- Gemini API exhausted
- **Last trade:** 2026-03-19 00:18 UTC (SELL orders for SOLUSDT and DOGEUSDT)
- **Last non-API-exhausted reasoning:** 2026-03-18 17:44 UTC
- **API exhausted since:** All 6 symbols returning "API exhausted" since 2026-03-19 ~01:19 UTC (220 occurrences total in log)
- **Portfolio:** USDT=3311.39, total=$3316.23
- **Positions:** Dust only -- BTC 3.8e-06, ETH 1.059e-05, SOL 0.00081, DOGE 0.718, SUI 0.0029, AAVE 0.04343
- **Note:** AAVE position (0.04343, ~$5 notional) has pending Sell signal of ~$116 but "API exhausted" prevents execution. SOL and DOGE positions were being actively sold on 2026-03-18, orders filled or cancelled by 2026-03-19 00:18.
- **Errors:** APIError(-2011) "Unknown order sent" on cancel attempts (orders already filled), otherwise clean.

### binance-worksteal-daily
- **Status:** RUNNING (pid 2608998, uptime 1d 3h)
- **Health:** HEALTHY
- **Last cycle:** 2026-03-19 13:05 UTC -- 0 positions, equity=$3319
- **Next cycle:** 2026-03-20 00:05 UTC
- **Positions:** 0 open
- **Errors:** None in error log
- **Note:** Running daily at UTC midnight, fetching bars for 30/30 symbols consistently. No dip entries triggered recently (market not meeting dip thresholds). Multiple restarts on 2026-03-17/18 but no impact on trading.

### binance-rl-cache-refresh
- **Status:** RUNNING (pid 2608826, uptime 1d 3h)
- **Health:** HEALTHY (with minor warnings)
- **Symbols:** BTC, ETH, SOL, DOGE, AAVE, LINK (h1 + h24 forecasts)
- **Last refresh:** 2026-03-19 09:13 UTC (up-to-date, 5-min cycle)
- **Warnings:** Repeated heuristic fallback for BTC/ETH/LINK 24h Chronos forecasts (invalid model outputs). One ConnectionResetError for BTCFDUSD on 2026-03-19 04:06 UTC.
- **1h forecasts:** Working normally for all symbols.

### binance-doge-cache-refresh
- **Status:** RUNNING (pid 413670, uptime 6d 22h)
- **Health:** HEALTHY (with minor warnings)
- **Symbols:** DOGE, AAVE (h1 forecasts only)
- **Last refresh:** 2026-03-19 09:14 UTC
- **Warnings:** Intermittent ConnectionResetError for DOGEUSDT (transient, self-recovering). One DNS resolution failure on 2026-03-13 (network outage, resolved).

### binance-5min-collector
- **Status:** RUNNING (pid 888673, uptime 22d)
- **Health:** HEALTHY
- **Last entry:** 2026-03-19 22:18 UTC (DOGEUSDT: +2 bars, every 5 min)

### stock-deleverage-bot
- **Status:** RUNNING (pid 387610, uptime 3d 14h)
- **Health:** HEALTHY
- **Note:** Market closed (after hours), sleeping 60s between checks. Normal behavior.

---

## Stopped Services (Expected)

| Service | Stopped Since | Note |
|---------|---------------|------|
| unified-stock-trader | Not started | Replaced by hybrid-spot |
| stock-cache-refresh | Not started | Replaced by rl-cache-refresh |
| binance-meta-margin | Mar 12 | Superseded |
| binance-sui / sui-margin / sui-cache | Feb 17-22 | SUI strategy deprecated |
| alpaca-exit-nflx | Feb 14 | Completed |
| alpaca-stocks13-ppo | Feb 17 | Deprecated |
| binanceexp1-* | Feb 7-21 | Experiment concluded |

---

## Key Issues

1. **CRITICAL: Gemini API quota exhausted** for binance-hybrid-spot. All 6 symbols return "API exhausted" with conf=0.00, zero orders placed since 2026-03-19 00:19 UTC (~21 hours). The bot is functionally idle. Action: check Gemini API billing/quota, or switch to a backup model.

2. **Minor: Chronos 24h forecast heuristic fallbacks** for BTC, ETH, LINK in rl-cache-refresh. The model is producing invalid outputs for 24h horizon. 1h forecasts work fine. Not blocking trading but degrades forecast quality.

3. **Minor: Intermittent Binance API connection resets** for DOGEUSDT klines. Self-recovering, ~1 per day. No data gaps observed.

---

## Live State Files

### binance_worksteal/live_state.json
```json
{
    "positions": {},
    "last_exit": {},
    "peak_equity": 3318.9990246372
}
```
No open positions. Peak equity $3319.

---

## Equity Summary

| Service | Equity | Positions |
|---------|--------|-----------|
| binance-hybrid-spot | $3,316 | Dust (AAVE ~$5) |
| binance-worksteal-daily | $3,319 | 0 |
| stock-deleverage-bot | N/A (market closed) | N/A |

---

Generated: 2026-03-19T22:19Z
