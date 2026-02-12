# Alpaca Hourly Crypto PPO Sweep (CUDA + C Env)

This experiment trains PPO agents on hourly Alpaca crypto bars using the C-backed `pufferlib_market` environment, then evaluates holdout PnL on unseen dates.

## What It Does

- Exports hourly MKTD v2 binaries from `trainingdatahourly`:
  - `train_hourly_mktd_v2.bin`
  - `eval_holdout_hourly_mktd_v2.bin`
- Runs a hyperparameter sweep over reward-shaping variants.
- Evaluates each run on unseen holdout episodes (`episode_steps=720` hours).
- Writes ranked outputs to `sweep_results.json`.
- Optionally appends:
  - successful runs -> `alpacaprogress.md`
  - unsuccessful runs -> `unsuccessfulalpacaprogress.md`

## Run

```bash
source .venv/bin/activate

python experiments/alpaca_pufferlib_hourly_crypto_cuda_ppo_20260211/run_sweep.py \
  --total-timesteps 300000 \
  --num-envs 96 \
  --rollout-len 256 \
  --device cuda \
  --record-progress
```

## Config

Edit `experiments/alpaca_pufferlib_hourly_crypto_cuda_ppo_20260211/config.json` to adjust:

- symbol universe (e.g. `BTCUSD`, `ETHUSD`, `LINKUSD`, ...)
- train/holdout date windows
- environment params (fees, leverage, reward scale/clip)

## Notes

- MKTD v2 includes a tradable mask, so missing bars are explicitly non-tradable.
- Exporter expects `open/high/low/close/volume` columns in hourly CSV files.
