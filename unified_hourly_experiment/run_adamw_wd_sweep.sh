#!/bin/bash
set -e
cd /home/lee/code/stock
export PYTHONUNBUFFERED=1

PYTHON=".venv312/bin/python -u"
TRAIN="unified_hourly_experiment/train_unified_policy.py"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"

COMMON="--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --crypto-symbols '' --epochs 20 --batch-size 64 --sequence-length 48 --hidden-dim 512 --num-layers 6 --num-heads 8 --seed 1337 --warmup-steps 100 --dropout 0.1 --grad-clip 1.0 --fill-temperature 5e-4 --logits-softcap 12.0 --decision-lag-bars 1 --forecast-horizons 1 --no-compile --maker-fee 0.001 --max-leverage 2.0 --margin-annual-rate 0.0625 --fill-buffer-pct 0.0005 --validation-use-binary-fills --return-weight 0.15 --no-amp"

EVAL_BASE="--symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT --fee-rate 0.001 --margin-rate 0.0625 --no-close-at-eod --max-positions 7 --max-hold-hours 6 --min-edge 0.0"

eval_run() {
    local name=$1
    echo "--- Eval: $name ---"
    $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/$name $EVAL_BASE --holdout-days 30 2>&1 | tail -5
}

for WD in 0.03 0.04 0.05 0.06 0.07 0.08; do
    echo "=== AdamW lr=1e-5 wd=$WD ==="
    $PYTHON $TRAIN $COMMON --lr 1e-5 --weight-decay $WD --checkpoint-name wd_${WD}
    eval_run wd_${WD}
done

echo "=== AdamW wd sweep complete ==="
