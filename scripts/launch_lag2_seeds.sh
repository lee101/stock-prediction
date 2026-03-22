#!/bin/bash
# Launch lag2 realistic training for seeds 42 and 7
# Run AFTER seed 1337 completes
# Usage: bash scripts/launch_lag2_seeds.sh

cd /home/lee/code/stock
source .venv312/bin/activate

echo "Launching seed 42..."
TORCH_NO_COMPILE=1 PYTHONUNBUFFERED=1 nohup python scripts/train_crypto_portfolio.py \
    --symbols BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD \
    --epochs 20 \
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
    --decision-lag 2 \
    --no-compile \
    --run-name crypto_portfolio_lag2_seed42 \
    --seed 42 \
    > /tmp/crypto_lag2_seed42.log 2>&1 &
echo "Seed 42 PID: $!"

echo "Launching seed 7..."
TORCH_NO_COMPILE=1 PYTHONUNBUFFERED=1 nohup python scripts/train_crypto_portfolio.py \
    --symbols BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD \
    --epochs 20 \
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
    --decision-lag 2 \
    --no-compile \
    --run-name crypto_portfolio_lag2_seed7 \
    --seed 7 \
    > /tmp/crypto_lag2_seed7.log 2>&1 &
echo "Seed 7 PID: $!"

echo "Both seeds launched. Monitor with:"
echo "  tail -f /tmp/crypto_lag2_seed42.log"
echo "  tail -f /tmp/crypto_lag2_seed7.log"
