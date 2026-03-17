# Production Pipeline Notes

Updated: 2026-03-17

## Scope

This repo currently has more than one Binance-facing trading path. The main ones that matter for daily crypto are:

1. `trade_mixed_daily.py`
2. `rl-trading-agent-binance/trade_binance_live.py`
3. `binanceneural/trade_binance_daily_levels.py`

They are not the same strategy family, and they do not all have the same idempotency and quote-routing behavior.

## Verified Account Facts

Checked against the live Binance API on 2026-03-17:

- Open orders on `BTCFDUSD`, `BTCUSDT`, `ETHFDUSD`, and `ETHUSDT`: `0`
- Orders on those four pairs during the UTC window `2026-03-17 00:00:00` to `2026-03-18 00:00:00`: `0`
- Recent major-pair spot activity visible through the API is on `BTCFDUSD` and `ETHFDUSD`
- I could not reproduce the quoted `BTC/USDT` and `ETH/USDT` sell rows for 2026-03-17 from the current API history on those pairs

That means the current account state does not show any active BTC/ETH major-spot orders on USDT right now, and the most recent visible major-pair spot history in this account is predominantly FDUSD.

## Current Pipeline Families

### 1. `trade_mixed_daily.py`

Purpose:
- Daily mixed stock+crypto RL inference utility
- The checked-in systemd unit `systemd/daily-rl-trader.service` is paper-only

Training path:
- Trained from daily binaries exported by `pufferlib_market.export_data_daily`
- Searched via `pufferlib_market.autoresearch_rl`
- The older documented champion was `ent_anneal`
- The more recent local fresh-universe retests point to `reg_combo_2` as the stronger robustness anchor

Inference semantics:
- Uses `align_daily_price_frames()` plus `latest_snapshot()`
- `latest_snapshot()` uses the last row currently present in the daily CSVs

Important idempotency note:
- This path is only fully idempotent for same-day reruns if the underlying daily CSVs are unchanged
- If a cache refresh process updates today's in-progress daily row intraday, rerunning can change the inferred signal because the latest row becomes the new "close"

So for `trade_mixed_daily.py`, your intuition is broadly right:
- the main same-day rerun drift comes from the current daily row changing
- that changes the effective current close and therefore the feature snapshot

### 2. `rl-trading-agent-binance/trade_binance_live.py`

Purpose:
- Binance RL+LLM hybrid live trader
- Spot BTC/ETH already route to FDUSD in this path

Training path:
- PPO-based RL model from the `rl-trainingbinance` family
- Repo docs currently describe `binance6_ppo_v1_h1024_100M` as the main live hybrid anchor
- LLM layer then refines entry/exit and allocation

Execution semantics:
- Spot mode:
  - `BTCUSD -> BTCFDUSD`
  - `ETHUSD -> ETHFDUSD`
  - alts mostly use USDT pairs
- Margin mode:
  - BTC/ETH switch to USDT pairs by design because the margin execution path is USDT-based
- Stablecoin conversion is already built in via `ensure_quote_balance()`

Idempotency note:
- This path is stateful at the portfolio/order level
- Same signal rerun can still produce a different action if balances, open orders, borrow headroom, or open positions have changed

### 3. `binanceneural/trade_binance_daily_levels.py`

Purpose:
- Chronos2 daily buy/sell levels trader

Training / model path:
- Uses daily CSV history
- Runs Chronos2 daily OHLC forecasting
- Derives buy/sell levels from quantiles
- Optional direction filter uses predicted close return

Execution semantics after the current patch:
- `BTCUSD -> BTCFDUSD`
- `ETHUSD -> ETHFDUSD`
- The script now auto-converts between `USDT` and `FDUSD` when funding the preferred quote side
- Existing base inventory exits also route through the explicit execution pair, so existing BTC/ETH inventory sells go to FDUSD now

Idempotency note:
- This path is forecast-idempotent across same-day reruns if the closed daily history is unchanged
- `_forecast_today_levels()` explicitly drops today's still-open daily bar before generating the forecast
- So rerunning later on the same UTC day should keep the same forecasted buy/sell levels

What can still change on same-day rerun here:
- account balances
- whether a prior watcher already filled or partially filled
- watcher expiry timing
- live stop-loss behavior, because that uses the current market price
- exchange filters / min notional edge cases

## Pair Routing Status

As of this update:

- `rl-trading-agent-binance/trade_binance_live.py` already had FDUSD routing for BTC/ETH spot
- `binanceneural/trade_binance_daily_levels.py` now also has explicit FDUSD routing for BTC/ETH spot
- `binanceneural` watcher spawning now carries an explicit `exchange_symbol`, so reruns dedupe against the real pair, not just the internal symbol

This matters because the internal symbol can stay `BTCUSD` / `ETHUSD` while the actual exchange order is placed on `BTCFDUSD` / `ETHFDUSD`.

## Idempotency Risks

### Fixed in this pass

- `binanceneural.binance_watchers._matches_plan()` had a real runtime bug: it called `_float_rel_eq` and `_float_qty_eq` even though those helpers did not exist in the file
- That could break watcher dedupe at runtime
- The helpers are now present, and tests cover the exchange-symbol-aware matching path

### Still important

- If a rerun changes the limit price, qty, side, or exchange pair, watcher dedupe should not treat it as identical
- If a rerun happens after fills or partial fills, the next decision can differ even when the forecast did not
- If the daily CSV refresh writes an in-progress current-day row, `trade_mixed_daily.py` can legitimately drift on rerun

## Main Failure Modes

### 1. Pipeline confusion

Different scripts in this repo have different assumptions:

- some are paper-only
- some are Binance spot
- some are Binance margin
- some are Chronos2-only
- some are RL+LLM hybrid

It is easy to think a pair-routing fix landed "everywhere" when it only landed in one live family.

### 2. Training/live mismatch

The strongest daily RL training results in the repo are not always the same models that survive realistic intraday replay or production order semantics.

Known mismatch classes:

- daily C-env training versus hourly replay execution
- mixed stock+crypto training versus Binance-only live deployment
- fee/slippage assumptions versus actual pair-specific execution

### 3. Current-day bar drift

`trade_mixed_daily.py` can move intraday if today's row is refreshed.

`binanceneural/trade_binance_daily_levels.py` should not move intraday from the forecast itself because it drops today's bar before forecasting.

### 4. Stablecoin funding drift

Even when the target execution pair is correct, a trade can still fail if the account is sitting in the wrong quote asset and conversion is not handled before order submission.

That is why the daily-levels path now checks spendable quote across both `USDT` and `FDUSD` and converts the shortfall before entry.

### 5. Margin pair expectations

For the hybrid Binance live path, BTC/ETH on margin still use USDT pairs by design.
That is not a bug in that script; it is part of how the margin execution layer is written.

## Practical Read

- If the active production path is `trade_binance_live.py`, BTC/ETH spot should already be FDUSD
- If the active production path is `binanceneural/trade_binance_daily_levels.py`, BTC/ETH spot is now FDUSD after this patch
- If the active production path is `trade_mixed_daily.py`, that is a separate daily RL utility and not the spot order-entry implementation itself

## Recommended Operational Checks

After any deployment touching Binance pair routing:

1. Query open orders on `BTCFDUSD`, `BTCUSDT`, `ETHFDUSD`, `ETHUSDT`
2. Query full order history for the last 24h on those same pairs
3. Confirm the watcher metadata carries the intended `exchange_symbol`
4. Confirm stablecoin conversions only occur when quote funding is actually short
5. Confirm the intended production runner, not just a checked-in paper service, is the process being restarted
