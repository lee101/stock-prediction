# Chronos Walk-Forward Hourly Crypto PPO (CUDA)

This experiment trains PPO on hourly crypto data using Chronos forecast features and evaluates on strict walk-forward holdout folds.

## Pipeline

1. Export MKTD binaries per fold from:
   - Forecast cache (`h1`, `h24`) via `pufferlib_market/export_data.py`
   - Hourly prices under `trainingdatahourly`
2. Train multiple PPO configs per fold on C `pufferlib_market` env.
3. Evaluate each checkpoint on unseen holdout episodes for that fold.
4. Rank by holdout annualized return and Sortino.
5. Aggregate fold-best metrics and optionally append to progress markdown files.

## Run

```bash
source .venv/bin/activate

python experiments/alpaca_pufferlib_hourly_crypto_chronos_walkforward_20260211/run_walkforward.py \
  --total-timesteps 600000 \
  --num-envs 96 \
  --rollout-len 256 \
  --device cuda \
  --record-progress
```

## Smoke Run

```bash
source .venv/bin/activate

python experiments/alpaca_pufferlib_hourly_crypto_chronos_walkforward_20260211/run_walkforward.py \
  --total-timesteps 120000 \
  --num-envs 32 \
  --rollout-len 128 \
  --max-folds 1 \
  --max-runs 2 \
  --device cuda \
  --force-export
```

## Outputs

- `experiments/alpaca_pufferlib_hourly_crypto_chronos_walkforward_20260211/walkforward_results.json`
- `experiments/alpaca_pufferlib_hourly_crypto_chronos_walkforward_20260211/folds/*/fold_results.json`
- `experiments/alpaca_pufferlib_hourly_crypto_chronos_walkforward_20260211/folds/*/checkpoints/*`

## Risk-Control Action Grid

The C env supports richer discrete actions that jointly encode:
- symbol
- side (long/short)
- allocation bin (position size)
- limit-level bin (entry offset around close in bps, filled only if inside bar high/low)

Configured via `config.json`:
- `env.action_allocation_bins`
- `env.action_level_bins`
- `env.action_max_offset_bps`

Legacy behavior is preserved with `1/1/0.0` (full-size market-at-close actions).

## Focused Comparisons

Run a single fold, a single run config, and optional symbol override:

```bash
source .venv/bin/activate

python experiments/alpaca_pufferlib_hourly_crypto_chronos_walkforward_20260211/run_walkforward.py \
  --fold-names wf4_202602 \
  --run-names r4_chronos_longshort_cash01_down5_smooth2_t01 \
  --symbols BTCUSD \
  --total-timesteps 600000 \
  --num-envs 96 \
  --rollout-len 256 \
  --device cuda \
  --record-progress
```
