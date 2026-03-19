#!/bin/bash
set -e
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

echo "=== Long Training: tp=0.15, seed=314, 30min ==="
timeout 1800 python -u -m pufferlib_market.train \
  --data-path pufferlib_market/data/crypto5_daily_train.bin \
  --total-timesteps 999999999 --max-steps 720 \
  --hidden-size 1024 --lr 3e-4 --ent-coef 0.05 \
  --gamma 0.99 --gae-lambda 0.95 \
  --num-envs 128 --rollout-len 256 --ppo-epochs 4 \
  --seed 314 --reward-scale 10.0 --reward-clip 5.0 \
  --cash-penalty 0.01 --fee-rate 0.001 \
  --trade-penalty 0.15 --anneal-lr \
  --checkpoint-dir pufferlib_market/checkpoints/long_daily/tp0.15_s314_30min \
  --periods-per-year 365.0 || true

echo ""
echo "=== Long Training: tp=0.10, seed=42, 30min ==="
timeout 1800 python -u -m pufferlib_market.train \
  --data-path pufferlib_market/data/crypto5_daily_train.bin \
  --total-timesteps 999999999 --max-steps 720 \
  --hidden-size 1024 --lr 3e-4 --ent-coef 0.05 \
  --gamma 0.99 --gae-lambda 0.95 \
  --num-envs 128 --rollout-len 256 --ppo-epochs 4 \
  --seed 42 --reward-scale 10.0 --reward-clip 5.0 \
  --cash-penalty 0.01 --fee-rate 0.001 \
  --trade-penalty 0.10 --anneal-lr \
  --checkpoint-dir pufferlib_market/checkpoints/long_daily/tp0.10_s42_30min \
  --periods-per-year 365.0 || true

echo ""
echo "=== Evaluating long-trained checkpoints ==="
python -u eval_long_daily.py
