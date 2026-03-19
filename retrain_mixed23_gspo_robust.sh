#!/bin/bash
set -e
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

echo "=== Retraining GSPO/Robust configs on mixed23 daily ==="

PYTHONPATH="$PWD/PufferLib:$PWD" python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/mixed23_daily_train.bin \
  --val-data pufferlib_market/data/mixed23_daily_val.bin \
  --time-budget 600 \
  --max-trials 10 \
  --descriptions "gspo_like,gspo_like_mix15,gspo_like_smooth_mix15,gspo_like_drawdown_mix15,robust_reg_tp005,robust_reg_tp01,robust_reg_tp005_dd002,per_env_adv_smooth" \
  --checkpoint-root pufferlib_market/checkpoints/mixed23_gspo_v2 \
  --leaderboard pufferlib_market/mixed23_gspo_v2_leaderboard.csv \
  --periods-per-year 365.0 \
  --max-steps-override 90 \
  --fee-rate-override 0.001

echo ""
echo "=== Results ==="
cat pufferlib_market/mixed23_gspo_v2_leaderboard.csv
