# PufferLib Market (Daily) Modern RL Sweep - 2026-02-10

Goal: re-run the daily `pufferlib_market` PPO experiment with:
- Correct terminal boundaries (PPO/GAE sees `done` properly).
- Reward shaping knobs to jointly optimize PnL + Sortino:
  - `downside_penalty` (penalize negative return variance)
  - `trade_penalty` (discourage churn / over-trading)
- Hourly replay evaluation on unseen days to catch intraday path risk and excessive order counts.

This uses the MKTD v2 daily exporter (`pufferlib_market/export_data_daily.py`) with a `tradable` mask
to prevent stock trades on market-closed days.

## Data Export

```bash
source .venv/bin/activate

python pufferlib_market/export_data_daily.py \
  --symbols BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AAPL,TSLA \
  --data-root trainingdata/train \
  --output experiments/pufferlib_market_daily_modernrl_20260210/train_mktd_v2.bin \
  --start-date 2022-02-08 \
  --end-date 2025-12-16 \
  --min-days 200

python pufferlib_market/export_data_daily.py \
  --symbols BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AAPL,TSLA \
  --data-root trainingdata/train \
  --output experiments/pufferlib_market_daily_modernrl_20260210/eval50d_mktd_v2.bin \
  --start-date 2025-12-17 \
  --end-date 2026-02-05 \
  --min-days 51
```

## Sweep (Train + Daily Eval + Hourly Replay Eval)

```bash
source .venv/bin/activate

python experiments/pufferlib_market_daily_modernrl_20260210/run_sweep.py \
  --total-timesteps 250000 \
  --device cuda
```

Outputs:
- `experiments/pufferlib_market_daily_modernrl_20260210/sweep_results.json` (tracked)
- `experiments/pufferlib_market_daily_modernrl_20260210/checkpoints/...` (ignored)

