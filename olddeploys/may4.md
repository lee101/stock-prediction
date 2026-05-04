# May 4 Production Snapshot

This is the snapshot requested before any replacement of the current production
path. No XGB replacement was deployed on 2026-05-04 because the aggressive
top-1 stock candidate failed the heldout/stress risk gate.

Current state at 2026-05-04 11:15 UTC:

- Alpaca live writer: `trading-server` on `127.0.0.1:8050`, pid `4130990`.
- Live strategy client: `daily-rl-trader`, pid `4131404`, account
  `live_prod`, bot id `daily_stock_sortino_v1`, allocation pct `12.5`,
  `min_agree_count=2`.
- XGB supervisor sidecar: inert `sleep infinity`; not holding the Alpaca
  writer lock and not allowed to trade unless explicitly re-enabled.
- Stock universe for the live server: 32 Alpaca stock symbols listed under
  `live_prod` in `src.trading_server.server`.
- Order safety: live stock execution goes through the trading server/alpaca
  wrapper path; XGB stock promotion remains blocked until a candidate clears
  heldout/stress validation.
