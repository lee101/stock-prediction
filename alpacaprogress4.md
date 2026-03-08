# Alpaca Progress 4

Date: 2026-03-08

## Summary

I added a reproducible Chronos feature sweep path for the BTC/ETH/SOL selector and ran a real seeded-baseline experiment end to end.

Current conclusion: do not deploy. The robust selector objective is still not good enough, and the baseline trio remains best under the realistic seeded-start test.

## Code Added

- `src/chronos2_params.py`
  - Added env-driven Chronos overrides for:
    - `CHRONOS2_FORCE_MULTIVARIATE`
    - `CHRONOS2_FORCE_CROSS_LEARNING`
    - `CHRONOS2_CONTEXT_LENGTH`
    - `CHRONOS2_BATCH_SIZE`
    - `CHRONOS2_SKIP_RATES`
    - `CHRONOS2_AGGREGATION_METHOD`
    - `CHRONOS2_FORCE_MULTISCALE`
- `binanceexp1/data.py`
  - Added `CHRONOS2_CONTEXT_HOURS` / `CHRONOS2_CONTEXT_LENGTH` / `CHRONOS2_BATCH_SIZE` support so the existing crypto training path can actually vary forecast context and batch settings.
- `binanceexp1/joint_chronos_forecast_cache.py`
  - Added grouped BTC/ETH/SOL Chronos cache generation for multi-symbol experiments.
- `binanceexp1/sweep_chronos_feature_configs_robust.py`
  - Added a top-level Chronos feature sweep runner around the existing robust train/search pipeline.
  - Supports cache seeding from an existing forecast root to avoid recomputing baseline horizons.

## Tests

Targeted validation passed:

```bash
source .venv313/bin/activate
pytest \
  tests/test_chronos2_params_overrides.py \
  tests/test_binanceexp1_data_chronos_env_overrides.py \
  tests/test_binance_joint_chronos_forecast_cache.py \
  tests/test_binance_sweep_chronos_feature_configs_robust.py \
  tests/test_binance_train_multiasset_selector_robust.py \
  tests/test_binance_checkpoint_set_search.py
```

Result: `15 passed`

## Completed Experiment

Experiment root:

- `experiments/binance_selector_chronos_features_targeted2/baseline_h1_24_ctx336`

This run used:

- existing `1h` and `24h` forecast caches
- realistic selection
- seeded starts: `flat`, `BTCUSD`, `ETHUSD`, `SOLUSD`
- shallow robust fine-tunes with preload from the current best trio

Artifacts:

- `experiments/binance_selector_chronos_features_targeted2/baseline_h1_24_ctx336/manifest.json`
- `experiments/binance_selector_chronos_features_targeted2/baseline_h1_24_ctx336/search/ranking.csv`
- `experiments/binance_selector_chronos_features_targeted2/baseline_h1_24_ctx336/search/scenarios.csv`

## Result

Best combo remained the existing baseline trio:

- `BTCUSD = btcusd_h1only_ft30_20260208/epoch_029.pt`
- `ETHUSD = seed42_ethusd_ft30_20260209_014309/epoch_028.pt`
- `SOLUSD = seed42_solusd_ft30_20260209_014309/epoch_030.pt`

Best robust metrics from `ranking.csv`:

- `selection_score = -18.135370362946947`
- `return_mean_pct = +2.6904847193944725`
- `return_worst_pct = -0.7796781213067521`
- `max_drawdown_worst_pct = 8.63161583098301`
- `all_profitable = false`

14d seeded-start breakdown for the winning combo:

- `flat = +4.5057%`
- `BTCUSD = +5.3958%`
- `ETHUSD = +1.6401%`
- `SOLUSD = -0.7797%`

Interpretation:

- the smooth baseline still makes money from `flat` / `BTC` / `ETH`
- it still fails the robustness bar because starting from `SOLUSD` remains negative
- retraining on the same `1h/24h` feature family did not beat the baseline trio

## Chronos 6h / Grouped Follow-Up

I also added targeted config files for:

- `experiments/chronos_feature_configs_targeted_small.json`
- `experiments/chronos_feature_configs_grouped_126.json`

I started real `1h,6h,24h` runs, including grouped multi-symbol cache generation with and without cross-learning, but stopped them after confirming the remaining bottleneck is still Chronos cache generation for the missing `6h` crypto forecasts. That work is resumable with the new sweep tooling and seeded cache roots.

Useful rerun entrypoints:

```bash
source .venv313/bin/activate
python -m binanceexp1.sweep_chronos_feature_configs_robust \
  --experiment-name binance_selector_chronos_features_targeted3_grouped \
  --feature-configs-json experiments/chronos_feature_configs_grouped_126.json \
  --seed-forecast-cache-root binanceneural/forecast_cache \
  --preload-checkpoints 'BTCUSD=binanceneural/checkpoints/btcusd_h1only_ft30_20260208/epoch_029.pt;ETHUSD=binanceneural/checkpoints/seed42_ethusd_ft30_20260209_014309/epoch_028.pt;SOLUSD=binanceneural/checkpoints/seed42_solusd_ft30_20260209_014309/epoch_030.pt' \
  --baseline-candidates 'BTCUSD=binanceneural/checkpoints/btcusd_h1only_ft30_20260208/epoch_029.pt;ETHUSD=binanceneural/checkpoints/seed42_ethusd_ft30_20260209_014309/epoch_028.pt;SOLUSD=binanceneural/checkpoints/seed42_solusd_ft30_20260209_014309/epoch_030.pt' \
  --max-train-configs 1 \
  --seeds 42 \
  --dry-train-steps 8 \
  --no-compile \
  --realistic-selection \
  --validation-use-binary-fills \
  --run-prefix chronos_feature_grouped
```

## Deployment Status

Do not deploy.

Reasons:

- no completed Chronos `6h` or grouped variant has yet beaten the seeded baseline result
- even the baseline still fails the seeded `SOLUSD` start-state requirement
- the strategy is not yet robust enough for live rebalance promotion
