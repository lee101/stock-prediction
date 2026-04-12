# trade_stock_wide

Wide daily stock planner with a more realistic intraday replay path.

## What it does

- builds a pre-open daily plan from `backtest_forecasts(...)`
- selects the best strategy per symbol using forecasted PnL plus average return
- ranks a wide universe and keeps the top `N` planned trades
- can re-rank the daily universe using `pnl`, `sortino`, `hybrid`, or a tiny augmented MLP selector
- reserves notional per planned pair while enforcing a gross leverage cap
- supports optional Gemini 3.1 sizing overrides per selected name, capped by a configured max pair size
- can optionally ingest bounded `autoresearch_stock` sweep results as a symbol-level RL prior
- replays the fixed daily plan through real hourly bars for that session
- simulates work stealing between active watches before a trade is actually filled
- writes one centralized run log and one per-symbol log file
- can optionally run a Gemini 3.1 overlay on the selected top picks and compare it against the base replay before using it

## Live execution

`trade_stock_wide.live` is the only intended live entry surface for this package.
It does not talk to Alpaca directly. It submits through `src.trading_server.client`
so the trading server remains the single account writer above Alpaca.

Safety rules in the live adapter:

- claim the trading-server writer lease before any order submission
- heartbeat the writer lease while staging orders
- revalidate every long plan before submission
- hard-fail if `take_profit_price <= entry_price`
- attach the planned entry and take-profit prices in order metadata for auditability
- rely on the existing Alpaca singleton and death-spiral guard behind the server

## Replay model

The hourly replay is intentionally simple and deterministic:

- the plan is frozen before the market opens
- pending orders wake up once price gets within the watch activation window of the planned entry
- active watches reserve capacity even before a fill, which lets work stealing happen under the leverage cap
- if leverage is saturated, a better forecast can steal an active watch slot from a worse forecast that is not already too close to execution
- a bar must push through the planned entry by the fill buffer before the trade counts as filled
- take-profit checks begin on the hour after entry
- if take-profit never hits, the position exits at the final hourly close of that session

This is more realistic than the original daily-bar shortcut because it captures:

- whether the plan would have filled at all
- which symbol would have consumed leverage first
- the interaction between multiple planned trades inside the same day

## Logging

Each run creates:

- `trade_stock_wide/logs/run_YYYYMMDD_HHMMSS/central.log`
- `trade_stock_wide/logs/run_YYYYMMDD_HHMMSS/symbols/<SYMBOL>.log`

The central log contains the plan and replay summary. Per-symbol logs record intraday fill and exit events for that symbol.

## Gemini Overlay

The optional Gemini overlay is a post-selection refinement step:

- the base planner still chooses the top `N` symbols first
- Gemini only sees those selected candidates, not the whole universe
- the prompt includes recent daily bars, Chronos high/low/close context, the selected strategy, and recent strategy-PnL history
- Gemini can tighten the long entry / take-profit, skip the trade, and suggest a smaller per-name allocation percent
- all Gemini-adjusted prices still go through the same long-price safety validation, including `sell > buy`
- Gemini sizing is capped by `--max-pair-notional-fraction` so it cannot silently exceed the configured per-order risk bound
- `compare` mode backtests the base plan and the Gemini-refined plan, then only uses the Gemini version if replay metrics are better

Relevant CLI flags:

- `--gemini-overlay-mode off|compare|force`
- `--gemini-model gemini-3.1-flash-lite-preview`
- `--gemini-min-confidence 0.35`
- `--gemini-cache-dir trade_stock_wide/gemini_cache`
- `--max-pair-notional-fraction 0.50`
- `--rl-prior-leaderboard trade_stock_wide/rl_group_sweep_latest.csv`
- `--rl-prior-weight 0.05`

## Usage

```bash
python -m trade_stock_wide.run \
  --limit-symbols 100 \
  --top-k 6 \
  --selection-objective hybrid \
  --account-equity 100000 \
  --backtest-days 30 \
  --num-simulations 60 \
  --watch-activation-pct 0.005 \
  --steal-protection-pct 0.004 \
  --gemini-overlay-mode compare \
  --hourly-root trainingdatahourly
```

Use `--daily-only-replay` to compare against the simpler daily replay.

## Tuning Sweep

Use the sweep helper to search the ranking objective and work-steal knobs on the same replay path:

```bash
python scripts/sweep_trade_stock_wide.py \
  --limit-symbols 20 \
  --backtest-days 30 \
  --selection-objectives pnl,sortino,hybrid,tiny_net,torch_mlp \
  --top-ks 4,6,8 \
  --watch-activation-pcts 0.003,0.005,0.0075 \
  --steal-protection-pcts 0.003,0.004,0.006 \
  --hourly-root trainingdatahourly \
  --output trade_stock_wide/sweeps/latest.csv
```

The `tiny_net` selector is intentionally small and fast. The `torch_mlp` selector uses the same feature set and augmentation scheme but trains through `torch` and prefers CUDA when available, so it is the better option when you want to search learned rankers on the GPU before committing to a larger RL path.

## RL Prior Sweep

For short bounded end-to-end learning experiments on single names or symbol pairs:

```bash
python scripts/run_autoresearch_stock_group_sweep.py \
  --data-root trainingdatahourly/stocks \
  --group-size 1 \
  --limit-symbols 8 \
  --max-groups 8 \
  --experiments base,dynamic_score_floor,soft_rank_sizing,budget_entropy_confidence \
  --time-budget-seconds 60 \
  --output trade_stock_wide/rl_group_sweep_latest.csv
```

Then feed the resulting leaderboard back into the wide selector:

```bash
python -m trade_stock_wide.run \
  --limit-symbols 100 \
  --selection-objective hybrid \
  --rl-prior-leaderboard trade_stock_wide/rl_group_sweep_latest.csv \
  --rl-prior-weight 0.05
```
