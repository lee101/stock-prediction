#!/bin/bash
set -e
cd /home/lee/code/stock

PYTHON=".venv312/bin/python"
TRAIN="unified_hourly_experiment/train_unified_policy.py"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"

TRAIN_COMMON="--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --forecast-horizons 1 --epochs 50 --batch-size 64 --lr 1e-5 --sequence-length 48 --hidden-dim 512 --num-layers 6 --weight-decay 0.03 --return-weight 0.10 --decision-lag-bars 1 --seed 1337 --no-compile"

EVAL_COMMON="--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --max-positions 9 --decision-lag-bars 1 --bar-margin 0.0013"

run_eval() {
    local ckpt_dir=$1
    local name=$2
    for days in 30 60 90 120; do
        echo "--- $name ${days}d ---"
        $PYTHON $SWEEP --checkpoint-dir "$ckpt_dir" $EVAL_COMMON --holdout-days $days 2>&1 | grep -E "Best Sortino|Best Return"
    done
}

echo "=== EXP 1: Cosine LR schedule (lr-min-ratio=0.1) ==="
$PYTHON $TRAIN $TRAIN_COMMON --lr-schedule cosine --lr-min-ratio 0.1 --checkpoint-name top9_lag1_cosine_lr 2>&1 | tail -5
run_eval "unified_hourly_experiment/checkpoints/top9_lag1_cosine_lr" "cosine_lr"

echo "=== EXP 2: Feature noise std=0.01 ==="
$PYTHON $TRAIN $TRAIN_COMMON --feature-noise-std 0.01 --checkpoint-name top9_lag1_fnoise01 2>&1 | tail -5
run_eval "unified_hourly_experiment/checkpoints/top9_lag1_fnoise01" "fnoise01"

echo "=== EXP 3: Dropout 0.2 ==="
$PYTHON $TRAIN $TRAIN_COMMON --dropout 0.2 --checkpoint-name top9_lag1_dropout02 2>&1 | tail -5
run_eval "unified_hourly_experiment/checkpoints/top9_lag1_dropout02" "dropout02"

echo "=== ALL EXPERIMENTS COMPLETE ==="
