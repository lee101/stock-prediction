#!/usr/bin/env bash
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
export PYTHONPATH="$PWD:$PWD/PufferLib:${PYTHONPATH:-}"
mkdir -p analysis/remote_runs/drytest_001
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting remote pipeline"
echo "+ rm -rf pufferlib_market/build pufferlib_market/binding*.so"
rm -rf pufferlib_market/build pufferlib_market/binding*.so
echo "+ python pufferlib_market/setup.py build_ext --inplace --force"
python pufferlib_market/setup.py build_ext --inplace --force
echo "+ python -c 'import pufferlib_market.binding; print(\"binding OK\")'"
python -c 'import pufferlib_market.binding; print("binding OK")'
echo "+ python -u -m pufferlib_market.autoresearch_rl --train-data pufferlib_market/data/stocks11_daily_train_2012.bin --val-data pufferlib_market/data/stocks11_daily_val_2012.bin --time-budget 200 --max-trials 5 --leaderboard autoresearch_stocks11_2012_leaderboard.csv --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock --rank-metric holdout_robust_score --stocks --max-timesteps-per-sample 200 --start-from 233 --seed-only --no-poly-prune --periods-per-year 252.0 --max-steps-override 252 --fee-rate-override 0.001 --holdout-data pufferlib_market/data/stocks11_daily_val_2012.bin --holdout-eval-steps 90 --holdout-n-windows 20 --holdout-seed 1337 --holdout-fee-rate 0.001 --holdout-fill-buffer-bps 5.0 --holdout-max-leverage 1.0 --holdout-short-borrow-apr 0.0"
python -u -m pufferlib_market.autoresearch_rl --train-data pufferlib_market/data/stocks11_daily_train_2012.bin --val-data pufferlib_market/data/stocks11_daily_val_2012.bin --time-budget 200 --max-trials 5 --leaderboard autoresearch_stocks11_2012_leaderboard.csv --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock --rank-metric holdout_robust_score --stocks --max-timesteps-per-sample 200 --start-from 233 --seed-only --no-poly-prune --periods-per-year 252.0 --max-steps-override 252 --fee-rate-override 0.001 --holdout-data pufferlib_market/data/stocks11_daily_val_2012.bin --holdout-eval-steps 90 --holdout-n-windows 20 --holdout-seed 1337 --holdout-fee-rate 0.001 --holdout-fill-buffer-bps 5.0 --holdout-max-leverage 1.0 --holdout-short-borrow-apr 0.0
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pipeline complete"
