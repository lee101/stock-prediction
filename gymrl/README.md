# GymRL Experiment Overview

The `gymrl/` package contains a reinforcement-learning experiment that builds on the existing forecasting stack (`predict_stock_forecasting.py`, Toto, Chronos) and the live trading loop (`predict_stock_e2e.py`). It introduces:

- **Feature engineering** that compresses probabilistic forecasts and realised market context into a dense tensor (`FeatureBuilder`).
- **A Gymnasium environment** (`PortfolioEnv`) that converts those features into a reward signal aligned with portfolio growth, turnover, and optional risk penalties (CVaR, forecast uncertainty).
- **Baseline behaviour policies** for offline RL pre-training and evaluation.
- **Offline dataset tooling** to produce replay buffers compatible with algorithms such as IQL/CQL (via d3rlpy) and PPO fine-tuning (via Stable-Baselines3).
- **A training script** (`gymrl/train_ppo_allocator.py`) that learns a first-pass allocator with PPO and keeps artefacts ready for downstream integration (e.g., `predict_stock_gymrl.py`).

## Setup

All dependencies are listed in `requirements.in`. If you only need the RL stack, you can install the relevant pieces with:

```bash
uv pip install gymnasium stable-baselines3 d3rlpy
```

The feature builder will automatically try Toto first, then Chronos, then fall back to a bootstrap sampler. To use Toto or Chronos you must have their assets available locally (the existing repo already vendors Chronos; Toto should be placed under `/mnt/fast/code/chronos-forecasting/toto` as per `src/models/toto_wrapper.py`).

## Quick Start

1. **Build the feature cube and (optionally) an offline dataset** using the packaged helpers:

   ```bash
   python -m gymrl.train_ppo_allocator \
       --data-dir tototraining/trainingdata/train \
       --behaviour-dataset data/rl/behaviour_topk.npz \
       --num-timesteps 200000
   ```

   This script:
   - Reads the per-symbol CSV history under `tototraining/trainingdata/train`.
   - Generates forecast statistics via Toto/Chronos/bootstrap.
   - Creates the Gym environment and trains a PPO allocator.
   - Saves checkpoints under `gymrl/artifacts/` and writes training metadata for reproducibility.

2. **Inspect artefacts**:
   - PPO checkpoints live in `gymrl/artifacts/`.
   - Behaviour dataset (`.npz`) contains `observations`, `actions_weights`, `rewards`, etc., ready for d3rlpy.
   - Metadata (`training_metadata.json`) captures CLI arguments, environment config, and dataset dimensions.

3. **Integrate into the live pipeline**:
   - Use `FeatureBuilder` inside a future `predict_stock_gymrl.py` to mirror the production feature computation.
   - Load the trained model with `PPO.load("gymrl/artifacts/ppo_allocator_final.zip")`.
   - Feed live forecasts into `PortfolioEnv` (or a thin runtime wrapper) to generate target weights.

## Key Modules

| File | Purpose |
| --- | --- |
| `gymrl/config.py` | Dataclasses configuring feature generation, environment dynamics, and dataset export. |
| `gymrl/feature_pipeline.py` | Builds a `FeatureCube` (`features`, `realized_returns`, `forecast_cvar`, `timestamps`). |
| `gymrl/wrappers.py` | Optional `ObservationNormalizer` wrapper with online mean/std normalization. |
| `gymrl/portfolio_env.py` | Gymnasium environment with turnover costs, drawdown penalties, and optional CVaR/uncertainty shaping. |
| `gymrl/behaviour.py` | Heuristic policies (`topk_equal_weight`, `kelly_fractional`, etc.) for offline RL. |
| `gymrl/offline_dataset.py` | Converts the feature cube + behaviour policy into replay buffers. |
| `gymrl/train_ppo_allocator.py` | End-to-end PPO training entry point with checkpointing and evaluation callbacks. |

## Next Steps

- Add a `predict_stock_gymrl.py` runner that mirrors `predict_stock_e2e.py` but sources allocations from a trained RL policy.
- Expand the environment to support shorting with explicit leverage limits (currently `step_with_weights` is long-only).
- Integrate d3rlpy offline pre-training (IQL/CQL) before PPO fine-tuning by consuming the saved `.npz` datasets.
- Wire monitoring hooks (TensorBoard/Datadog) once the PPO model runs in paper trading.
