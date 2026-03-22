#!/bin/bash
# Phase 2: L-block + cross-feature sweeps
# Run after stocks10 sweep (PID 4014799) finishes
set -euo pipefail

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

echo "[$(date)] Stocks10 sweep done. Starting L-block sweep (s1137_lr2e4, h2048, obs_norm, etc)..."

# L-block: 6 named configs + 2 random mutations
python3 -u -m pufferlib_market.autoresearch_rl \
  --stocks \
  --train-data pufferlib_market/data/stocks11_daily_train_2012.bin \
  --val-data pufferlib_market/data/stocks11_daily_val_2012.bin \
  --holdout-data pufferlib_market/data/stocks11_daily_val_2012.bin \
  --time-budget 600 \
  --max-timesteps-per-sample 700 \
  --max-trials 13 \
  --start-from 161 \
  --rank-metric holdout_robust_score \
  --holdout-eval-steps 90 \
  --holdout-n-windows 20 \
  --fee-rate-override 0.001 \
  --leaderboard autoresearch_stocks11_lblock_sweep.csv \
  --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock_lblock \
  2>&1 | tee -a autoresearch_lblock.log

echo "[$(date)] L-block sweep done. Starting cross-feature sweep..."

# Cross-feature: test if extra features (relative_return, beta, corr, breadth_rank) help
python3 -u -m pufferlib_market.autoresearch_rl \
  --stocks \
  --train-data pufferlib_market/data/stocks11_daily_train_2012_cross.bin \
  --val-data pufferlib_market/data/stocks11_daily_val_2012_cross.bin \
  --holdout-data pufferlib_market/data/stocks11_daily_val_2012_cross.bin \
  --time-budget 600 \
  --max-timesteps-per-sample 700 \
  --max-trials 5 \
  --descriptions "lr1e4_anneal_s1137,lr1e4_anneal_s5678,s1137_sdp01_t001,s5678_sdp01_t001,s1137_transformer" \
  --rank-metric holdout_robust_score \
  --holdout-eval-steps 90 \
  --holdout-n-windows 20 \
  --fee-rate-override 0.001 \
  --leaderboard autoresearch_stocks11_cross_sweep.csv \
  --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock_cross \
  2>&1 | tee -a autoresearch_cross.log

echo "[$(date)] All Phase 2 sweeps done."
