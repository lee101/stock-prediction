# Best Binance Hourly XGB Portfolio Pack

Last updated: 2026-05-04

## Current Frontier

The current deployed frontier is the aggressive correlation-packed variant:

- Label: `h12_short_corr168_08_aggr2777_dd1998_20260504`
- Eval window: 2026-01-04 to 2026-05-04, about 120 days
- Market: Binance spot hourly data, top 36 symbols by recent dollar volume
- Side: short-only
- Horizon: 12 hours
- Positions: 2 active positions, 4 pending entry watchers
- Selection: `first_trigger`
- Correlation packing: 168 hourly bars, min 48 return observations, max signed correlation 0.8
- Entry allocator: concentrated, max single position fraction 0.8, min second position fraction 0.2
- Leverage: 2.55x gross
- Drawdown entry scaler: start 4.5%, full 15%, floor 32%
- Fee/slippage realism: 10 bps fee, 5 bps fill buffer, 10 bps force-close slippage, 6.25% margin APR, decision lag 2

## Current Result

- Monthly return: 27.77%
- Total return: 165.39%
- Max drawdown: 19.98%
- Sortino: 4.85
- Exits: 260
- Selection score: 870.80

Artifacts:

- CSV: `analysis/binance_hourly_aggressive_sweep_20260504.csv`
- HTML review: `analysis/binance_hourly_aggressive_sweep_20260504.html`
- JSON summary: `analysis/binance_hourly_aggressive_sweep_20260504.json`

Deployed live runner:

- Supervisor program: `binance-hourly-xgb-margin-pack`
- Launch: `deployments/binance-hourly-xgb-margin-pack/launch.sh`
- Runner: `scripts/binance_hourly_xgb_margin_trader.py`
- Current plan artifact: `analysis/binance_hourly_xgb_margin_plan_latest.json`

Live verification on 2026-05-04:

- Binance writer audit passes with one `xgb_hourly_pack` writer and the old Binance trading daemons stopped.
- Existing margin coverage is `6/6` covered.
- First live cycle selected `FILUSDT` and `TONUSDT`; both entries filled `0`, were canceled, and post-cycle coverage stayed `6/6` covered.
- The runner now has a post-cycle settle wait before coverage re-audit to handle transient Binance margin asset state after canceled borrow-style orders.
- Entry orders are supervised for the full simulated 1-hour TTL before cancel;
  any filled short quantity gets a matching `BUY`/`AUTO_REPAY` exit while the
  process is still watching the order.

## Previous Frontier

This is the best corrected Binance hourly XGB result found so far after adding
direct trailing pairwise correlation gating on top of the short-mode candidate
ranking and concentrated allocator leverage fixes.

- Eval window: 2026-01-02 to 2026-05-02, about 120 days
- Market: Binance spot hourly data, top 36 symbols by recent dollar volume
- Side: short-only
- Horizon: 12 hours
- Positions: 2 active positions, 2 pending entry watchers
- Selection: `first_trigger`
- Correlation packing: 168 hourly bars, min 48 return observations, max signed correlation 0.8
- Entry allocator: concentrated, max single position fraction 0.8, min second position fraction 0.2
- Leverage: 2.55x gross
- Drawdown entry scaler: start 4.5%, full 15%, floor 32%
- Fee/slippage realism: 10 bps fee, 5 bps fill buffer, 10 bps force-close slippage, 6.25% margin APR, decision lag 2

## Result

- Monthly return: 27.95%
- Total return: 168.01%
- Max drawdown: 29.66%
- Sortino: 4.09
- Exits: 246
- Selection score: 852.54

Artifacts:

- CSV: `analysis/binance_hourly_twopos_h12_short_corr168_08_2795_20260504.csv`
- HTML review: `analysis/binance_hourly_twopos_h12_short_corr168_08_2795_20260504.html`
- Trace JSON: `analysis/binance_hourly_twopos_h12_short_corr168_08_2795_20260504.json`

Previous frontier:

- `analysis/binance_hourly_twopos_h12_short_diversified_27pct_20260503.csv`
- Monthly return 27.18%, total return 161.22%, max drawdown 31.46%, Sortino 3.89, 248 exits.

## Caveats

The current aggressive variant is deployed, but it is still a one-sided short
strategy. Keep the rollout small and watch live fill quality, borrow behavior,
exit coverage, and regime drift closely. The previous `27.95%/mo` row crossed
the monthly PnL target but was not deployed because its max drawdown was still
near 30% and writer isolation had not yet been cleaned up.

## 2026-05-04 Deploy Path

- Added `scripts/binance_hourly_xgb_margin_trader.py` as the dedicated
  one-cycle inference/deploy runner for this candidate.
- Dry-run on refreshed data through `2026-05-04T05:00:00Z` selected `0`
  candidates at `2026-05-04T03:00:00Z`; the same was true with live account
  state disabled, so no live order was sent.
- Live execution is intentionally blocked unless writer isolation is clean,
  data is fresh, exit coverage passes, and `ALLOW_BINANCE_XGB_LIVE_TRADING=1`
  is set.
- For short entries, the live path places `SELL`/`AUTO_BORROW_REPAY`, watches
  fills, places matching `BUY`/`AUTO_REPAY` exits for filled quantity, and
  cancels unfilled entries by default.
- Current machine is now deploy-clean for Binance: the old trading daemons are
  stopped and the process audit allows exactly one `xgb_hourly_pack` writer.

## 2026-05-04 Notes

- Pulled upstream XGB research work through `496b47ac`, including `xgbbest.md`,
  `xgbnew/` correlation-packing work, BitBankGo-style worksteal allocation
  probes, and a torch hourly level optimizer.
- Direct correlation gating worked better than the earlier PC2 bucket heuristic
  when used alone. The best 36-symbol row at `corr_window=168`,
  `corr_max_signed=0.8`, and `2.55x` reached 27.95% monthly with 29.66% DD.
- Combining direct correlation with the old PC buckets over-pruned the edge.
  A hard `corr_max_signed=0.4` row went negative; `0.8` was the useful setting.
- Expanding the exact same shape to 60 symbols failed badly
  (`-20%/mo`, DD above 75%). More pairs need learned/rolling symbol filters
  before they are useful.
- The pulled long-only 48h hourly level optimizer was refuted as-is on Binance:
  BTCUSDT, ETHUSDT, and SOLUSDT all compounded near -100% in the smoke run.
  This direction still needs side-aware long/short choice and portfolio-level
  risk controls before it is worth comparing to the XGB pack.

## Next Ideas

- Reduce one-sided short risk with long/short hedging that does not destroy the
  short edge.
- Keep improving correlation-aware packing: try soft penalties instead of hard
  skips, and train a rolling symbol filter for the 60-symbol universe.
- Expand beyond 36 symbols and let the selector opportunistically trade more
  pairs while constraining liquidity.
- Combine hourly and daily context: train hourly execution levels against
  24-48 hour forecast windows, then fit buy/sell levels that maximize realized
  simulator PnL.
- Explore fast retrain/test-time adaptation over 1-2 day windows as part of the
  simulated algorithm, not as offline leakage.
