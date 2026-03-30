#!/bin/bash
# Sweep around dd002 best config with varied regularization + seeds
# Base: wd=0.05, tp=0.005, dd=0.02, slip=8bps, obs_norm
# Goal: find configs that are even more robust (6/6 positive windows, higher Calmar)

set -e
cd /home/lee/code/stock
source .venv/bin/activate
export PYTHONPATH=/home/lee/code/stock

TRAIN=pufferlib_market/data/mixed23_latest_train_20260320.bin
VAL=pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin
BASE_DIR=pufferlib_market/checkpoints/overfit_control_sweep

COMMON="--data-path $TRAIN --val-data-path $VAL \
  --total-timesteps 5000000 --num-envs 128 --rollout-len 256 \
  --ppo-epochs 4 --minibatch-size 2048 --lr 3e-4 --hidden-size 256 \
  --periods-per-year 365 --obs-norm --val-eval-interval 25 --val-eval-windows 6 \
  --save-every 50 --max-periodic-checkpoints 5 --wandb-mode disabled"

run_trial() {
    local name=$1; shift
    local dir="$BASE_DIR/$name"
    if [ -f "$dir/val_best.pt" ] || [ -f "$dir/best.pt" ]; then
        echo "SKIP $name (already done)"
        return
    fi
    echo "=== START $name ==="
    python -u -m pufferlib_market.train --checkpoint-dir "$dir" $COMMON "$@" 2>&1 | tail -5
    echo "=== DONE $name ==="
}

# dd002 baseline with different seeds
run_trial dd002_s42  --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --fill-slippage-bps 8 --seed 42
run_trial dd002_s123 --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --fill-slippage-bps 8 --seed 123
run_trial dd002_s777 --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --fill-slippage-bps 8 --seed 777

# Stronger regularization
run_trial wd10_tp01_dd005_s42  --weight-decay 0.10 --trade-penalty 0.01 --drawdown-penalty 0.05 --fill-slippage-bps 8 --seed 42
run_trial wd10_tp01_dd005_s123 --weight-decay 0.10 --trade-penalty 0.01 --drawdown-penalty 0.05 --fill-slippage-bps 8 --seed 123

# Higher slippage training (12bps) - forces wider edges
run_trial dd002_slip12_s42  --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --fill-slippage-bps 12 --seed 42
run_trial dd002_slip12_s123 --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --fill-slippage-bps 12 --seed 123

# Entropy annealing (explore more early, exploit later)
run_trial dd002_entanneal_s42  --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --fill-slippage-bps 8 --ent-coef 0.05 --anneal-ent --ent-coef-end 0.005 --seed 42
run_trial dd002_entanneal_s123 --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --fill-slippage-bps 8 --ent-coef 0.05 --anneal-ent --ent-coef-end 0.005 --seed 123

# Cosine LR schedule
run_trial dd002_cosine_s42  --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --fill-slippage-bps 8 --lr-schedule cosine --seed 42
run_trial dd002_cosine_s123 --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --fill-slippage-bps 8 --lr-schedule cosine --seed 123

# Downside penalty variant (Sortino-like training signal)
run_trial dd002_downside_s42  --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --downside-penalty 0.5 --fill-slippage-bps 8 --seed 42

# Larger model (h512) with same reg
run_trial dd002_h512_s42  --weight-decay 0.05 --trade-penalty 0.005 --drawdown-penalty 0.02 --fill-slippage-bps 8 --hidden-size 512 --seed 42

echo "=== ALL DONE ==="
echo "Evaluate with:"
echo "  for d in $BASE_DIR/*/; do"
echo "    echo \$d; python -u pufferlib_market/evaluate_sliding.py --checkpoint \$d/val_best.pt --data-path $VAL --hidden-size 256 --episode-len 90 --stride 15 --slippage-bps 8 --periods-per-year 365 --deterministic 2>&1 | grep -E 'Return:|Sortino:|Calmar:|profitable'"
echo "  done"
