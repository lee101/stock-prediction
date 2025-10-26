#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; please install https://github.com/astral-sh/uv" >&2
  exit 1
fi

uv venv .venv
source .venv/bin/activate
uv pip install -e .[hf,sb3]

if [ ! -d external/kronos ]; then
  git clone https://github.com/shiyu-coder/Kronos external/kronos
fi

python -m differentiable_market_kronos.train_sb3 --ohlcv data/sample_ohlcv.csv --save-dir runs/differentiable_market_kronos
