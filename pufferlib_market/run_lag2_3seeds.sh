#!/usr/bin/env bash
# 3-seed training run with realistic decision_lag=2 on mixed23 data.
# Addresses 2026-04-07 finding: prior "champion" checkpoints were all validated at
# decision_lag=0 (lookahead bias) and collapse at lag>=2 in realistic eval.
#
# Each seed trains for 5M timesteps with the same config. Val uses the SAME
# lag=2 setting the C env now supports, so train/val/production-validation
# should finally agree.
#
# Requires GPU headroom. Check `nvidia-smi` first.
set -euo pipefail

cd /home/lee/code/stock

TRAIN_DATA=pufferlib_market/data/mixed23_latest_train_20260320.bin
VAL_DATA=pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin
OUT=pufferlib_market/checkpoints/lag2_mixed23_3seeds

mkdir -p "$OUT"

for seed in 42 1337 2024; do
  echo "=== seed $seed ==="
  /home/lee/code/stock/.venv313/bin/python -m pufferlib_market.train \
    --data-path "$TRAIN_DATA" \
    --val-data-path "$VAL_DATA" \
    --checkpoint-dir "$OUT/seed_$seed" \
    --total-timesteps 5000000 \
    --num-envs 64 --rollout-len 256 \
    --seed $seed \
    --decision-lag 2 \
    --max-steps 180 --periods-per-year 365 \
    --hidden-size 1024 --arch mlp \
    --fee-rate 0.001 --max-leverage 1.0 \
    --fill-slippage-bps 5 --fill-buffer-bps 5 \
    --obs-norm --cosine-lr --warmup-frac 0.05 \
    2>&1 | tee "$OUT/seed_${seed}.log"
done

# Post-training realistic holdout sweep
for seed in 42 1337 2024; do
  ckpt="$OUT/seed_$seed/best.pt"
  [ -f "$ckpt" ] || continue
  for slip in 0 5 10 20; do
    /home/lee/code/stock/.venv313/bin/python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" \
      --data-path "$VAL_DATA" \
      --eval-hours 30 --n-windows 50 --seed 42 \
      --fee-rate 0.001 --slippage-bps $slip --fill-buffer-bps 5 \
      --max-leverage 1.0 --periods-per-year 365 --decision-lag 2 \
      --deterministic \
      --out "$OUT/seed_${seed}_slip${slip}.json" 2>&1 | tail -3
  done
done

echo "Done. Summarize with: python -c 'import json; ...' over $OUT/seed_*_slip*.json"
