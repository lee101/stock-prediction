#!/bin/bash
# Retrain the top "lost" configs that showed Sortino 3.7+ in fresh runs
# but never had their checkpoints saved properly.
# These configs have NEVER been trained on the standard mixed23_daily data.

set -e
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

echo "=== Retraining top daily configs on mixed23 ==="
echo "Target configs: robust_reg_tp01, gspo_like_drawdown_mix15, robust_reg_tp005, gspo_like_smooth_mix15, gspo_like"
echo ""

PYTHONPATH="/nvme0n1-disk/code/stock-prediction/PufferLib:/nvme0n1-disk/code/stock-prediction" \
python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/mixed23_daily_train.bin \
  --val-data pufferlib_market/data/mixed23_daily_val.bin \
  --time-budget 600 \
  --max-trials 10 \
  --periods-per-year 365 \
  --max-steps-override 90 \
  --fee-rate-override 0.001 \
  --leaderboard pufferlib_market/autoresearch_mixed23_daily_retrain_leaderboard.csv \
  --checkpoint-root pufferlib_market/checkpoints/autoresearch_mixed23_daily_retrain \
  --descriptions "robust_reg_tp01,robust_reg_tp005,gspo_like_drawdown_mix15,gspo_like_smooth_mix15,gspo_like,robust_reg_tp005_dd002,robust_reg_tp005_sds02,per_env_adv_smooth"

echo ""
echo "=== Done! Check leaderboard ==="
cat pufferlib_market/autoresearch_mixed23_daily_retrain_leaderboard.csv
