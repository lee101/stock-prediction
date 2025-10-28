# PufferLib 3.0 Trading PPO

Ultra-fast single-GPU reinforcement learning for the differentiable stock trading environment.  
The stack targets NVIDIA RTX 5090 boxes and mirrors the latest PufferLib 3.0 best practices:

- shared-memory vectorisation via `pufferlib.vector`
- bf16 autocast with fused optimisers and optional `torch.compile`
- CUDA-graph friendly fixed rollout shapes
- differentiable PnL with smooth transaction/slippage costs

---

## Layout

| Path | Purpose |
| ---- | ------- |
| `pufferlibtraining/market_env.py` | Torch-first MarketEnv with batched OHLCV windows, differentiable rewards, and smooth leverage penalties. |
| `pufferlibtraining/pufferrl.py` | PPO-style training loop built on `pufferlib.vector`. Provides CLI/config, bf16 autocast, `torch.compile`, and GAE. |
| `pufferlibtraining/config/rl.ini` | Single-RTX 5090 defaults (4k envs, 128-step rollouts, bf16, fused AdamW). |
| `gym/__init__.py` | Lightweight shim that exposes `gym` API via Gymnasium so PufferLib ≥3 imports cleanly on Python 3.12. |

---

## Quick start

```bash
# (optional) pin tools from PyPI
uv pip install --upgrade torch pufferlib gymnasium

# run PPO with the default 5090 preset
python -m pufferlibtraining.pufferrl --config pufferlibtraining/config/rl.ini
```

The trainer logs progress to stdout. Swap `--config` for a custom `.ini` file or override individual entries by editing `rl.ini` (see below).

---

## Configuration

`pufferlibtraining/config/rl.ini` is standard INI:

```ini
[vec]
num_envs = 4096
num_workers = 8

[train]
rollout_len   = 128
minibatches   = 8
update_iters  = 2
learning_rate = 3e-4
entropy_coef  = 0.01
gamma         = 0.99
gae_lambda    = 0.95
mixed_precision = bf16
torch_compile   = true

[policy]
hidden_size = 256

[env]
context_len    = 128
episode_len    = 256
fee_bps        = 0.5
slippage_bps   = 1.5
leverage_limit = 1.5
```

*Raise* `rollout_len` or `num_envs` carefully—keep the token budget (`rollout_len × num_envs`) within your GPU memory envelope.

---

## Market environment details

- Loads aligned OHLCV arrays for all available tickers under `trainingdata/`.
- Observation: `[context_len, feature_dim + 2]` with centred log prices, log returns, volume z-score, broadcast position/value diagnostics.
- Action: tanh-squashed scalar ∈ [-1, 1] interpreted as target leverage.
- Reward:  
  `reward = position[t-1] * return[t] - smooth_fee(|Δposition|) - smooth_slip(|Δposition|)`  
  implemented in torch for differentiability; metrics keep running PnL and Sharpe-proxy buffers.
- Episodes sample random ticker/start offsets to generate millions of windows per epoch without Python object churn.

All tensors live on the requested device (defaults to `cuda` when available) and degrade gracefully to CPU if needed.

---

## Integrating with the rest of the repo

- **Supervised pretrain (hftraining/):** export encoders and load the state dict into the PPO policy before calling `train`.
- **Post-training evaluation:** drop the learned policy into `marketsimulator/` or your custom risk dashboards for out-of-sample Sharpe/drawdown metrics.
- **Devcontainer:** the optional PufferTank image (`pufferai/puffertank:latest`) provides CUDA 12.x + PyTorch 2.4—ideal for replicating 5090 numbers on fresh hardware.

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `ModuleNotFoundError: gym` | Ensure the repo’s `gym/` shim is on `PYTHONPATH` (automatic when running from repo root). |
| Throughput < 1M steps/s | Warm up the policy with 2–3 dummy updates (`torch_compile=true`), or lower `num_envs` to avoid VRAM pressure. |
| Rewards NaN | Check data for zeros/NaNs; ensure tickers have ≥ `context_len + episode_len` rows post alignment. |

---

Happy trading 🐡🚀

