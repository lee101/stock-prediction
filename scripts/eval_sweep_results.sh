#!/bin/bash
# Evaluate all checkpoints from overfit control sweep
set -e
cd /home/lee/code/stock
source .venv/bin/activate
export PYTHONPATH=/home/lee/code/stock

VAL=pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin
BASE_DIR=pufferlib_market/checkpoints/overfit_control_sweep

echo "name,ann_return_8bps,sortino,calmar,max_dd,profitable_pct,p05"

for dir in "$BASE_DIR"/*/; do
    name=$(basename "$dir")
    # prefer val_best.pt, fallback to best.pt
    ckpt="$dir/val_best.pt"
    [ -f "$ckpt" ] || ckpt="$dir/best.pt"
    [ -f "$ckpt" ] || continue

    # detect hidden size from name
    hs=256
    [[ "$name" == *h512* ]] && hs=512
    [[ "$name" == *h1024* ]] && hs=1024

    result=$(python -u pufferlib_market/evaluate_sliding.py \
        --checkpoint "$ckpt" \
        --data-path "$VAL" \
        --hidden-size $hs \
        --episode-len 90 --stride 15 \
        --slippage-bps 8 \
        --periods-per-year 365 \
        --deterministic 2>&1)

    ann=$(echo "$result" | grep "Annualized:" | awk '{print $2}')
    sortino=$(echo "$result" | grep "Sortino:" | awk '{print $2}')
    calmar=$(echo "$result" | grep "Calmar:" | awk '{print $2}')
    maxdd=$(echo "$result" | grep "MaxDD:" | grep worst | awk '{print $2}')
    prof=$(echo "$result" | grep "profitable=" | grep -oP 'profitable=\K[0-9.]+')
    p05=$(echo "$result" | grep "p05:" | awk '{print $2}')

    echo "$name,$ann,$sortino,$calmar,$maxdd,$prof,$p05"
done
