#!/bin/bash
set -e
cd /home/lee/code/stock
export PYTHONUNBUFFERED=1

REMOTE="sshpass -p 'ka3iMI4OSNvgFcREuDaQyLguFxuP' ssh -o StrictHostKeyChecking=no administrator@93.127.141.100"
REMOTE_DIR="/nvme0n1-disk/code/stock-prediction"

echo "=== Syncing code to remote ==="
sshpass -p 'ka3iMI4OSNvgFcREuDaQyLguFxuP' rsync -avz --exclude='.venv*' --exclude='__pycache__' --exclude='*.pt' --exclude='checkpoints' --exclude='modded-nanogpt' \
    -e "ssh -o StrictHostKeyChecking=no" \
    /home/lee/code/stock/binanceneural \
    /home/lee/code/stock/unified_hourly_experiment \
    /home/lee/code/stock/differentiable_loss_utils.py \
    /home/lee/code/stock/traininglib \
    /home/lee/code/stock/src \
    administrator@93.127.141.100:$REMOTE_DIR/

$REMOTE "cd $REMOTE_DIR && bash -c '
export PYTHONUNBUFFERED=1
PYTHON=\".venv312/bin/python -u\"
TRAIN=\"unified_hourly_experiment/train_unified_policy.py\"
SWEEP=\"unified_hourly_experiment/sweep_epoch_portfolio.py\"

COMMON=\"--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --crypto-symbols \\\"\\\" --epochs 20 --batch-size 64 --sequence-length 48 --hidden-dim 512 --num-layers 6 --num-heads 8 --warmup-steps 100 --dropout 0.1 --grad-clip 1.0 --fill-temperature 5e-4 --logits-softcap 12.0 --decision-lag-bars 1 --forecast-horizons 1 --no-compile --maker-fee 0.001 --max-leverage 2.0 --margin-annual-rate 0.0625 --fill-buffer-pct 0.0005 --validation-use-binary-fills --return-weight 0.15 --no-amp\"

EVAL_BASE=\"--symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT --fee-rate 0.001 --margin-rate 0.0625 --no-close-at-eod --max-positions 7 --max-hold-hours 6 --min-edge 0.0\"

for WD in 0.03 0.04 0.05 0.06 0.07 0.08; do
    echo \"=== AdamW lr=1e-5 wd=\$WD seed=42 ===\"
    \$PYTHON \$TRAIN \$COMMON --lr 1e-5 --weight-decay \$WD --seed 42 --checkpoint-name wd_\${WD}_s42
    echo \"--- Eval: wd_\${WD}_s42 ---\"
    \$PYTHON \$SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/wd_\${WD}_s42 \$EVAL_BASE --holdout-days 30 2>&1 | tail -5
done

echo \"=== Remote AdamW wd sweep complete ===\"
'"
