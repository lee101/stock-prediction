#!/bin/bash
set -e
cd /home/lee/code/stock
export PYTHONUNBUFFERED=1

PYTHON=".venv312/bin/python -u"
TRAIN="unified_hourly_experiment/train_unified_policy.py"
BASE="--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --crypto-symbols '' --epochs 50 --batch-size 64 --lr 1e-5 --sequence-length 48 --hidden-dim 512 --num-layers 6 --num-heads 8 --seed 1337 --warmup-steps 100 --weight-decay 0.03 --dropout 0.1 --grad-clip 1.0 --fill-temperature 5e-4 --logits-softcap 12.0 --decision-lag-bars 1 --forecast-horizons 1 --no-compile --maker-fee 0.001 --max-leverage 2.0 --margin-annual-rate 0.0625 --fill-buffer-pct 0.0005 --validation-use-binary-fills"

echo "=== Run 1: base rw=0.10 ==="
$PYTHON $TRAIN $BASE --return-weight 0.10 --checkpoint-name realistic_rw010

echo "=== Run 2: rw=0.15 ==="
$PYTHON $TRAIN $BASE --return-weight 0.15 --checkpoint-name realistic_rw015

echo "=== Run 3: rw=0.20 ==="
$PYTHON $TRAIN $BASE --return-weight 0.20 --checkpoint-name realistic_rw020

echo "=== Run 4: cosine lr schedule ==="
$PYTHON $TRAIN $BASE --return-weight 0.10 --lr-schedule cosine --lr-min-ratio 0.01 --checkpoint-name realistic_cosine

echo "=== Run 5: sortino_dd loss ==="
$PYTHON $TRAIN $BASE --return-weight 0.10 --loss-type sortino_dd --dd-penalty 2.0 --checkpoint-name realistic_sortino_dd

echo "=== Run 6: fill_buffer=0.001 ==="
$PYTHON $TRAIN $BASE --return-weight 0.10 --fill-buffer-pct 0.001 --checkpoint-name realistic_fillbuf001

echo "=== All training runs complete ==="
