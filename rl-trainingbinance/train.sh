#!/bin/bash
# RL Training for Binance 6-symbol crypto trader
# Symbols: BTC, ETH, SUI, SOL, AAVE, DOGE
#
# Best practices from crypto12 experiments:
# - anneal-lr is critical (nearly doubles returns)
# - h1024 is sweet spot
# - ent-coef 0.05 optimal
# - reward: ret*10, clip[-5,5], cash-penalty 0.01
# - More steps = super-linear returns

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv313/bin/activate

DATA=rl-trainingbinance/data/binance6_data.bin
CKPT_DIR=rl-trainingbinance/checkpoints

# Phase 1: 100M steps baseline
echo "=== Phase 1: 100M steps ==="
python -u -m pufferlib_market.train \
    --data-path "$DATA" \
    --max-steps 720 \
    --num-envs 64 \
    --hidden-size 1024 \
    --arch mlp \
    --lr 3e-4 \
    --anneal-lr \
    --ent-coef 0.05 \
    --reward-scale 10.0 \
    --reward-clip 5.0 \
    --cash-penalty 0.01 \
    --drawdown-penalty 0.0 \
    --fee-rate 0.001 \
    --max-leverage 1.0 \
    --total-timesteps 100000000 \
    --checkpoint-dir "${CKPT_DIR}/binance6_ppo_v1_h1024_100M" \
    --save-every 50 \
    --seed 42 \
    "$@"
