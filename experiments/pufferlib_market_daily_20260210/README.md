# PufferLib Market (Daily) Experiment - 2026-02-10

Goal: train/evaluate the `pufferlib_market` C environment on **daily** bars from `trainingdata/train`,
with a 50-calendar-day holdout window and a tradable-mask that prevents stock trades on market-closed days.

This experiment uses the new daily exporter:
`pufferlib_market/export_data_daily.py` (MKTD v2, includes `tradable` mask).

## Data Export

```bash
source .venv/bin/activate

# Train window (ends before holdout)
python pufferlib_market/export_data_daily.py \
  --symbols BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AAPL,TSLA \
  --data-root trainingdata/train \
  --output experiments/pufferlib_market_daily_20260210/train_mktd_v2.bin \
  --start-date 2022-02-08 \
  --end-date 2025-12-16 \
  --min-days 200

# Holdout window (51 calendar days = 50 step episode)
python pufferlib_market/export_data_daily.py \
  --symbols BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AAPL,TSLA \
  --data-root trainingdata/train \
  --output experiments/pufferlib_market_daily_20260210/eval50d_mktd_v2.bin \
  --start-date 2025-12-17 \
  --end-date 2026-02-05 \
  --min-days 51
```

## Train

```bash
source .venv/bin/activate

python pufferlib_market/train.py \
  --data-path experiments/pufferlib_market_daily_20260210/train_mktd_v2.bin \
  --max-steps 50 \
  --periods-per-year 365 \
  --fee-rate 0.001 \
  --max-leverage 1.0 \
  --num-envs 64 \
  --rollout-len 256 \
  --total-timesteps 2000000 \
  --anneal-lr \
  --checkpoint-dir experiments/pufferlib_market_daily_20260210/checkpoints/daily_mix8_v1
```

## Evaluate (Holdout)

```bash
source .venv/bin/activate

python pufferlib_market/evaluate.py \
  --checkpoint experiments/pufferlib_market_daily_20260210/checkpoints/daily_mix8_v1/best.pt \
  --data-path experiments/pufferlib_market_daily_20260210/eval50d_mktd_v2.bin \
  --max-steps 50 \
  --periods-per-year 365 \
  --num-envs 1 \
  --num-episodes 5 \
  --deterministic
```

## Sweep Runner

`run_sweep.py` runs a small hyperparam sweep (intended as a starting point; scale `--total-timesteps` up for real training).

