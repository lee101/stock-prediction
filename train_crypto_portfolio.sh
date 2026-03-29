#!/usr/bin/env bash
set -euo pipefail
cd /home/lee/code/stock
source .venv312/bin/activate
export TORCH_NO_COMPILE=1

SYMBOLS="${1:-BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD}"
EPOCHS="${2:-15}"
RUN_NAME="${3:-crypto_portfolio_$(date +%Y%m%d_%H%M%S)}"

python scripts/train_crypto_portfolio.py \
    --symbols "$SYMBOLS" \
    --epochs "$EPOCHS" \
    --batch-size 16 \
    --lr 1e-5 \
    --weight-decay 0.04 \
    --transformer-dim 384 \
    --transformer-layers 6 \
    --transformer-heads 6 \
    --sequence-length 48 \
    --lr-schedule cosine \
    --maker-fee 0.001 \
    --margin-rate 0.0625 \
    --max-leverage 5.0 \
    --fill-buffer 0.0005 \
    --return-weight 0.10 \
    --no-compile \
    --run-name "$RUN_NAME" \
    --seed 1337
