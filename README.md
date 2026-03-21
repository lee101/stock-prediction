# stock-prediction

A production RL trading system for crypto and equities. The core loop is:
fast C market environment + PPO policy training + Chronos2 time-series
forecasting + live execution on Binance (FDUSD/USDT) and Alpaca (US stocks).

## Architecture

```
Raw OHLCV data (Binance / Alpaca / yfinance)
        |
        v
  Binary export (export_data_*.py)
        |
        v
  C market environment (pufferlib_market/market_env.c)
        |
        v
  PPO training (pufferlib_market/train.py)
        |
        v
  Evaluate OOS (pufferlib_market/evaluate.py)
        |
        v
  Deploy (trade_execution_listener.py / binan/)
```

Chronos2 LoRA forecasts feed in as additional features at inference time
(see `forecast_cache_lookup.py`, `retrain_chronos2_hourly_loras.py`).

Remote GPU training dispatches to RunPod pods and stores checkpoints in
Cloudflare R2 (see `src/remote_training_pipeline.py`,
`scripts/dispatch_rl_training.py`).

## Quick start

```bash
git clone <repo>
cd stock-prediction
source .venv313/bin/activate
python -m pytest tests/ -x -q
```

Python 3.13 is the primary version. The `.venv313` environment should be
pre-populated. To rebuild it:

```bash
uv venv .venv313 --python 3.13
source .venv313/bin/activate
uv pip install -e ".[dev]"
```

GPU (CUDA 12.8) is required for training. CPU-only runs work for unit tests
and backtests.

## Key scripts

| Script | Purpose |
|--------|---------|
| `export_data_hourly_price.py` | Export hourly OHLCV to `.bin` for RL training |
| `export_data_daily.py` | Export daily OHLCV to `.bin` |
| `pufferlib_market/train.py` | Train PPO policy on the C market env |
| `pufferlib_market/evaluate.py` | Evaluate a checkpoint OOS |
| `pufferlib_market/autoresearch_rl.py` | Automated hyperparameter sweep with OOS eval |
| `retrain_chronos2_hourly_loras.py` | Retrain Chronos2 LoRA adapters per symbol |
| `run_crypto_lora_batch.py` | Batch LoRA evaluation across symbols |
| `forecast_cache_lookup.py` | Disk-cached Chronos2 inference |
| `trade_execution_listener.py` | Live trade execution loop |
| `binan/binance_wrapper.py` | Binance API wrapper (FDUSD + USDT pairs) |
| `scripts/dispatch_rl_training.py` | Dispatch training jobs to RunPod |
| `src/remote_training_pipeline.py` | Remote training pipeline (R2 storage) |
| `smart_test_runner.py` | Change-aware test runner |

## C environment

The market simulator is written in C for speed:

```bash
cd pufferlib_market
python setup.py build_ext --inplace
```

This must be compiled with `.venv313` active.

## Remote training

Training jobs can be dispatched to RunPod GPU pods:

```bash
python scripts/dispatch_rl_training.py --config <config>
```

Checkpoints are synced to Cloudflare R2. See `CONTRIBUTING.md` for the
SSH `StrictHostKeyChecking=no` rationale.

## Testing

```bash
# Unit tests (fast, no GPU needed)
python -m pytest tests/ -x -q -k "not slow" --ignore=tests/integration

# Full suite (requires self-hosted GPU runner)
python -m pytest tests/ -x -q
```

## Requirements

- Python 3.13 (`.venv313`)
- CUDA 12.8 for GPU training
- `uv` for package management (`uv pip install`, never plain `pip`)

## License

MIT. See [LICENSE](LICENSE).
