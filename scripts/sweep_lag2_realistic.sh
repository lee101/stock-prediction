#!/bin/bash
# Sweep lag2 realistic checkpoints after training completes
cd /home/lee/code/stock
source .venv312/bin/activate

for seed_dir in crypto_portfolio_lag2_realistic crypto_portfolio_lag2_seed42 crypto_portfolio_lag2_seed7; do
    ckpt_dir="binanceneural/checkpoints/$seed_dir"
    if [ -d "$ckpt_dir" ] && ls "$ckpt_dir"/epoch_*.pt 1>/dev/null 2>&1; then
        echo "=== Sweeping $seed_dir ==="
        TORCH_NO_COMPILE=1 python binanceneural/sweep_crypto_portfolio.py \
            --checkpoint-dir "$ckpt_dir" \
            --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD \
            --lags 1 2 3 \
            --fee-rate 0.001 \
            --max-hold 24 \
            --fill-buffer-bps 5.0 \
            --seq-len 48
        echo ""
    else
        echo "=== $seed_dir: no checkpoints yet ==="
    fi
done
