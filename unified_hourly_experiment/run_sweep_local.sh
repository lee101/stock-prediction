#!/bin/bash
set -e
cd /home/lee/code/stock
export PYTHONUNBUFFERED=1

PYTHON=".venv312/bin/python -u"
TRAIN="unified_hourly_experiment/train_unified_policy.py"
BASE="--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --crypto-symbols '' --epochs 50 --batch-size 64 --lr 1e-5 --hidden-dim 512 --num-layers 6 --num-heads 8 --seed 1337 --warmup-steps 100 --dropout 0.1 --grad-clip 1.0 --fill-temperature 5e-4 --logits-softcap 12.0 --decision-lag-bars 1 --forecast-horizons 1 --no-compile --maker-fee 0.001 --max-leverage 2.0 --margin-annual-rate 0.0625 --fill-buffer-pct 0.0005 --validation-use-binary-fills --return-weight 0.15"

# Axis 1: Weight decay
echo "=== wd=0.05 ==="
$PYTHON $TRAIN $BASE --sequence-length 48 --weight-decay 0.05 --checkpoint-name sweep_wd05
echo "=== wd=0.08 ==="
$PYTHON $TRAIN $BASE --sequence-length 48 --weight-decay 0.08 --checkpoint-name sweep_wd08
echo "=== wd=0.10 ==="
$PYTHON $TRAIN $BASE --sequence-length 48 --weight-decay 0.10 --checkpoint-name sweep_wd10

# Axis 2: Sequence length
echo "=== seq32 ==="
$PYTHON $TRAIN $BASE --sequence-length 32 --weight-decay 0.03 --checkpoint-name sweep_seq32
echo "=== seq64 ==="
$PYTHON $TRAIN $BASE --sequence-length 64 --weight-decay 0.03 --checkpoint-name sweep_seq64

# Axis 3: Higher dropout
echo "=== dropout=0.2 ==="
$PYTHON $TRAIN $BASE --sequence-length 48 --weight-decay 0.03 --dropout 0.2 --checkpoint-name sweep_drop02

# Axis 4: Return weight fine-tune
echo "=== rw=0.12 ==="
$PYTHON $TRAIN $BASE --sequence-length 48 --weight-decay 0.03 --return-weight 0.12 --checkpoint-name sweep_rw012
echo "=== rw=0.18 ==="
$PYTHON $TRAIN $BASE --sequence-length 48 --weight-decay 0.03 --return-weight 0.18 --checkpoint-name sweep_rw018

echo "=== Local sweep complete ==="
