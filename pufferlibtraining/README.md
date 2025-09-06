PufferLib RL Training (Stock Trading)

Overview

- Goal: Train a stock-trading RL agent using a Gymnasium-compatible environment, set up to work with GPU and PufferLib vectorization.
- Status: Uses a clean Gymnasium env and SB3 PPO by default; integrates with PufferLib when available.

Install

- Python 3.10+ recommended with GPU PyTorch installed.
- From repo root:
- pip install -r requirements.txt
- Optional: verify GPU: python -c "import torch; print('CUDA:', torch.cuda.is_available())"

Quick Start

- Single stock PPO:
- python pufferlibtraining/train_ppo.py --symbol AAPL --data-dir data --total-timesteps 500000 --device cuda

Notes

- Data: expects CSVs in `data/` with OHLCV columns; picks the first matching `*{symbol}*.csv` else first CSV.
- Outputs: models and logs saved under `pufferlibtraining/models` and `pufferlibtraining/logs` (git‑ignored).
- PufferLib: If `pufferlib` is installed, the script will use simple vectorization hooks; otherwise it falls back to Gymnasium vector envs.

