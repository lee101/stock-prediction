#!/bin/bash
set -e
cd /home/lee/code/stock
export PYTHONUNBUFFERED=1

PYTHON=".venv312/bin/python -u"
TRAIN="unified_hourly_experiment/train_unified_policy.py"
BASE="--symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT --crypto-symbols '' --epochs 50 --batch-size 64 --lr 1e-5 --warmup-steps 100 --dropout 0.1 --grad-clip 1.0 --fill-temperature 5e-4 --logits-softcap 12.0 --decision-lag-bars 1 --forecast-horizons 1 --no-compile --maker-fee 0.001 --max-leverage 2.0 --margin-annual-rate 0.0625 --fill-buffer-pct 0.0005 --validation-use-binary-fills --sequence-length 48 --weight-decay 0.03 --hidden-dim 512 --num-layers 6 --num-heads 8 --seed 1337"

echo "=== 7sym rw=0.10 ==="
$PYTHON $TRAIN $BASE --return-weight 0.10 --checkpoint-name 7sym_rw010

echo "=== 7sym rw=0.15 ==="
$PYTHON $TRAIN $BASE --return-weight 0.15 --checkpoint-name 7sym_rw015

echo "=== 7sym rw=0.20 ==="
$PYTHON $TRAIN $BASE --return-weight 0.20 --checkpoint-name 7sym_rw020

echo "=== 7sym rw=0.15 wd=0.05 ==="
$PYTHON $TRAIN $BASE --return-weight 0.15 --weight-decay 0.05 --checkpoint-name 7sym_rw015_wd05

echo "=== All 7sym training complete ==="
