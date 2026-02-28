#!/bin/bash
set -e
cd /nvme0n1-disk/code/stock-prediction
export PYTHONUNBUFFERED=1

PYTHON=".venv312/bin/python -u"
TRAIN="unified_hourly_experiment/train_unified_policy.py"
BASE="--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --crypto-symbols '' --epochs 50 --batch-size 64 --lr 1e-5 --warmup-steps 100 --dropout 0.1 --grad-clip 1.0 --fill-temperature 5e-4 --logits-softcap 12.0 --decision-lag-bars 1 --forecast-horizons 1 --no-compile --maker-fee 0.001 --max-leverage 2.0 --margin-annual-rate 0.0625 --fill-buffer-pct 0.0005 --validation-use-binary-fills --return-weight 0.15 --sequence-length 48 --weight-decay 0.03"

# Axis 1: Architecture size (smaller = less overfit?)
echo "=== h256 4L ==="
$PYTHON $TRAIN $BASE --hidden-dim 256 --num-layers 4 --num-heads 4 --seed 1337 --checkpoint-name sweep_h256_4L
echo "=== h256 6L ==="
$PYTHON $TRAIN $BASE --hidden-dim 256 --num-layers 6 --num-heads 4 --seed 1337 --checkpoint-name sweep_h256_6L
echo "=== h384 6L ==="
$PYTHON $TRAIN $BASE --hidden-dim 384 --num-layers 6 --num-heads 6 --seed 1337 --checkpoint-name sweep_h384_6L
echo "=== h512 4L ==="
$PYTHON $TRAIN $BASE --hidden-dim 512 --num-layers 4 --num-heads 8 --seed 1337 --checkpoint-name sweep_h512_4L
echo "=== h512 8L ==="
$PYTHON $TRAIN $BASE --hidden-dim 512 --num-layers 8 --num-heads 8 --seed 1337 --checkpoint-name sweep_h512_8L

# Axis 2: Seed sensitivity
echo "=== seed=42 ==="
$PYTHON $TRAIN $BASE --hidden-dim 512 --num-layers 6 --num-heads 8 --seed 42 --checkpoint-name sweep_seed42
echo "=== seed=123 ==="
$PYTHON $TRAIN $BASE --hidden-dim 512 --num-layers 6 --num-heads 8 --seed 123 --checkpoint-name sweep_seed123
echo "=== seed=7 ==="
$PYTHON $TRAIN $BASE --hidden-dim 512 --num-layers 6 --num-heads 8 --seed 7 --checkpoint-name sweep_seed7

echo "=== Remote sweep complete ==="
