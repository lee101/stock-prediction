# Differentiable Market + Kronos

This module fuses the differentiable market research stack with frozen Kronos
forecasts. Kronos provides Monte Carlo path statistics while the downstream head
(trainable RL or differentiable Sharpe optimisation) remains lightweight,
stable, and fully differentiable.

## Components

- **`kronos_embedder.py`** – wraps the upstream Kronos tokenizer/model, samples
  price paths, and summarises them into rich features (mu/sigma/quantiles/path
  stats) for multiple horizons.
- **`adapter.py`** – aligns Kronos features with the multi-asset
  `differentiable_market` trainer so the GRPO policy sees both classic OHLC
  features and Kronos-derived summaries.
- **`envs/dm_env.py`** – minimal Gymnasium environment for single-asset RL
  experiments over Kronos features.
- **`train_sb3.py` / `eval_sb3.py`** – PPO training + evaluation with Stable
  Baselines3.
- **`train_sharpe_diff.py`** – optional differentiable Sharpe objective without
  RL, useful for ablations.
- **`speedrun.sh`** – nanochat-style end-to-end script using `uv` environments.

## Quick Start

```bash
uv sync
source .venv/bin/activate
uv pip install -e .[hf,sb3]
python -m differentiable_market_kronos.train_sb3 --ohlcv data/BTCUSD.csv --save-dir runs/dmk_ppo
```

To plug Kronos into the differentiable market trainer:

```python
from differentiable_market_kronos import KronosFeatureConfig, DifferentiableMarketKronosTrainer
from differentiable_market import config

trainer = DifferentiableMarketKronosTrainer(
    data_cfg=config.DataConfig(root=Path("trainingdata")),
    env_cfg=config.EnvironmentConfig(),
    train_cfg=config.TrainingConfig(lookback=192, batch_windows=64),
    eval_cfg=config.EvaluationConfig(),
    kronos_cfg=KronosFeatureConfig(model_path="NeoQuasar/Kronos-small", horizons=(1, 12, 48)),
)
trainer.fit()
```

## Testing

Lightweight tests live under `tests/differentiable_market_kronos`. They stub the
Kronos embedder to keep runtime manageable while exercising the feature plumbing
into the differentiable market trainer. Run them via:

```bash
pytest tests/differentiable_market_kronos -q
```
