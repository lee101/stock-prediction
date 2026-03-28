# Remote Training Runbook

This repository already uses a shared GPU box for long runs.

## Remote node

- host: `administrator@93.127.141.100`
- repo root: `/nvme0n1-disk/code/stock-prediction`
- ssh pattern: `ssh -o StrictHostKeyChecking=no administrator@93.127.141.100`

After connecting:

```bash
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
export PYTHONPATH=/nvme0n1-disk/code/stock-prediction:/nvme0n1-disk/code/stock-prediction/PufferLib:$PYTHONPATH
```

Use `.venv312` instead when the experiment already depends on it.

## Sync code to remote

Prefer `rsync` over ad hoc copy/paste:

```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  -e "ssh -o StrictHostKeyChecking=no" \
  /home/lee/code/stock/ \
  administrator@93.127.141.100:/nvme0n1-disk/code/stock-prediction/
```

## 1. Daily unified RL on remote

Current robust mixed23 sweep template:

```bash
ssh -o StrictHostKeyChecking=no administrator@93.127.141.100 '
  set -euo pipefail
  cd /nvme0n1-disk/code/stock-prediction
  source .venv313/bin/activate
  export PYTHONPATH=$PWD:$PWD/PufferLib:$PYTHONPATH
  mkdir -p analysis/remote_logs
  nohup python -u -m pufferlib_market.autoresearch_rl \
    --train-data pufferlib_market/data/mixed23_fresh_train.bin \
    --val-data pufferlib_market/data/mixed23_fresh_val.bin \
    --time-budget 180 --max-trials 8 \
    --descriptions robust_reg_wd02,robust_reg_tp005,robust_reg_tp01,robust_reg_tp005_sds02,robust_reg_tp005_dd002,robust_reg_tp005_sm001,robust_reg_tp005_ent,robust_reg_h512_tp005 \
    --periods-per-year 365 --max-steps-override 90 \
    --holdout-data pufferlib_market/data/mixed23_fresh_val.bin \
    --holdout-eval-steps 90 --holdout-n-windows 20 \
    --holdout-fee-rate 0.001 --holdout-fill-buffer-bps 5 \
    --replay-eval-data pufferlib_market/data/mixed23_fresh_val.bin \
    --replay-eval-hourly-root trainingdatahourly \
    --replay-eval-start-date 2025-06-01 \
    --replay-eval-end-date 2026-02-05 \
    --replay-eval-fill-buffer-bps 5 \
    --rank-metric replay_hourly_return_pct \
    --leaderboard pufferlib_market/autoresearch_mixed23_fresh_robust_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/mixed23_fresh_robust \
    > analysis/remote_logs/mixed23_fresh_robust.log 2>&1 &
  echo $! > analysis/remote_logs/mixed23_fresh_robust.pid
  cat analysis/remote_logs/mixed23_fresh_robust.pid
'
```

Tail the log:

```bash
ssh -o StrictHostKeyChecking=no administrator@93.127.141.100 \
  'tail -n 80 /nvme0n1-disk/code/stock-prediction/analysis/remote_logs/mixed23_fresh_robust.log'
```

## 2. Replay retest on remote

Use this to score one checkpoint on the current 60-day replay window:

```bash
ssh -o StrictHostKeyChecking=no administrator@93.127.141.100 '
  set -euo pipefail
  cd /nvme0n1-disk/code/stock-prediction
  source .venv313/bin/activate
  export PYTHONPATH=$PWD:$PWD/PufferLib:$PYTHONPATH
  python -m pufferlib_market.replay_eval \
    --checkpoint pufferlib_market/checkpoints/mixed23_fresh_targeted/reg_combo_2/best.pt \
    --daily-data-path /tmp/mixed23_val_60d_20251208_20260205.bin \
    --hourly-data-root trainingdatahourly \
    --start-date 2025-12-08 --end-date 2026-02-05 \
    --max-steps 59 --fill-buffer-bps 5 --deterministic --cpu \
    --output-json pufferlib_market/replay_eval_5bp_60d/reg_combo_2.json
'
```

## 3. Hourly Chronos2 -> RL remote launcher

Use the launcher when Chronos2 LoRA work should feed directly into forecast-cache features and then into `pufferlib_market` RL:

```bash
source .venv313/bin/activate
python scripts/launch_remote_hourly_chronos_rl.py \
  --run-id hourly_chronos_rl_$(date -u +%Y%m%d_%H%M%S) \
  --symbols BTCUSD,ETHUSD,SOLUSD,UNIUSD \
  --preaugs baseline,percent_change \
  --context-lengths 128 \
  --learning-rates 5e-5 \
  --num-steps 400 \
  --train-hours 720 \
  --val-hours 168 \
  --time-budget 1800 \
  --max-trials 4 \
  --descriptions baseline_anneal_lr,ent_anneal,clip_vloss,wd_01
```

The launcher:

- computes the latest shared hourly train/val window locally
- writes a local manifest at `analysis/remote_runs/<run-id>/launch_manifest.json`
- launches a detached remote pipeline that runs:
  - `scripts/run_crypto_lora_batch.py`
  - `scripts/promote_chronos2_lora_reports.py`
  - `scripts/build_hourly_forecast_caches.py`
  - `python -m pufferlib_market.export_data_hourly_forecast`
  - `python -m pufferlib_market.autoresearch_rl`

Monitor the run:

```bash
ssh -o StrictHostKeyChecking=no administrator@93.127.141.100 \
  'cd /nvme0n1-disk/code/stock-prediction && tail -n 80 analysis/remote_runs/<run-id>/pipeline.log'
```

Pull back the remote artifacts:

```bash
rsync -az -e "ssh -o StrictHostKeyChecking=no" \
  administrator@93.127.141.100:/nvme0n1-disk/code/stock-prediction/analysis/remote_runs/<run-id>/ \
  analysis/remote_runs/<run-id>/remote_run/
```

## 4. Shared Chronos2 -> hourly-vs-daily RL comparison

Use this when the goal is a fair downstream comparison from the same Chronos2
upstream forecasts:

- branch A: hourly forecast-cache RL
- branch B: daily RL with causal hourly forecast-context features from `export_data_daily_v4`

```bash
source .venv313/bin/activate
python scripts/launch_remote_chronos_compare_rl.py \
  --run-id chronos_compare_rl_$(date -u +%Y%m%d_%H%M%S) \
  --symbols BTCUSD,ETHUSD,SOLUSD,AVAXUSD,LINKUSD,UNIUSD,XRPUSD,DOGEUSD \
  --local-hourly-data-root trainingdatahourly \
  --remote-hourly-data-root trainingdatahourly \
  --local-daily-data-root trainingdatadaily \
  --remote-daily-data-root trainingdatadaily \
  --context-lengths 128 \
  --learning-rates 5e-5 \
  --num-steps 400 \
  --train-hours 4320 \
  --val-hours 1440 \
  --time-budget 1800 \
  --max-trials 4 \
  --descriptions sortino_rc3_tp08,sortino_rc3_tp09,robust_reg_tp01,sortino_top1_tp
```

The launcher:

- computes a shared time window that fits both hourly and daily data
- fine-tunes / promotes Chronos2 once
- builds one shared hourly forecast cache
- exports both:
  - hourly MKTD forecast features
  - daily fused MKTD with hourly Chronos context
- runs separate `pufferlib_market.autoresearch_rl` sweeps for hourly and daily branches

Monitor the run:

```bash
ssh -o StrictHostKeyChecking=no administrator@93.127.141.100 \
  'cd /nvme0n1-disk/code/stock-prediction && tail -n 80 analysis/remote_runs/<run-id>/pipeline.log'
```

Pull the leaderboards:

```bash
scp -o StrictHostKeyChecking=no \
  administrator@93.127.141.100:/nvme0n1-disk/code/stock-prediction/analysis/remote_runs/<run-id>/hourly_leaderboard.csv \
  analysis/remote_runs/<run-id>/hourly_leaderboard.csv
scp -o StrictHostKeyChecking=no \
  administrator@93.127.141.100:/nvme0n1-disk/code/stock-prediction/analysis/remote_runs/<run-id>/daily_leaderboard.csv \
  analysis/remote_runs/<run-id>/daily_leaderboard.csv
```

## 5. ETH risk-aware PPO on remote

Existing helper:

```bash
bash fastalgorithms/eth_risk_ppo/run_train_remote.sh 300000 eth_risk_ppo_remote_$(date -u +%Y%m%d_%H%M%S)
```

That helper already:

- connects to `administrator@93.127.141.100`
- checks out the current branch
- pulls latest code
- activates `.venv312` or `.venv313`
- launches detached training under `fastalgorithms/eth_risk_ppo/logs/`

## 6. Kronos fine-tuning on remote

Full-model or LoRA Kronos runs should be launched directly on the remote GPU box:

```bash
ssh -o StrictHostKeyChecking=no administrator@93.127.141.100 '
  set -euo pipefail
  cd /nvme0n1-disk/code/stock-prediction
  source .venv313/bin/activate
  export PYTHONPATH=$PWD:$PYTHONPATH
  mkdir -p analysis/remote_logs
  nohup python -m kronostraining.run_training \
    --data-dir trainingdata \
    --output-dir kronostraining/artifacts/remote_run_$(date -u +%Y%m%d_%H%M%S) \
    --lookback 64 \
    --horizon 30 \
    --validation-days 30 \
    --epochs 3 \
    > analysis/remote_logs/kronos_remote.log 2>&1 &
  echo $! > analysis/remote_logs/kronos_remote.pid
  cat analysis/remote_logs/kronos_remote.pid
'
```

LoRA example:

```bash
python -m kronostraining.run_training \
  --data-dir trainingdata/AAPL \
  --output-dir kronostraining/artifacts \
  --adapter lora \
  --adapter-name AAPL \
  --adapter-r 8 \
  --adapter-alpha 16 \
  --adapter-dropout 0.05
```

## 7. Existing remote sweep helpers

These already target the same host/path conventions and are safe starting points:

- `fastalgorithms/eth_risk_ppo/run_train_remote.sh`
- `fastalgorithms/eth_risk_ppo/run_iteration_batch_remote.sh`
- `fastalgorithms/eth_risk_ppo/run_iteration_queue_remote.sh`
- `scripts/run_sweeps_remote.sh`

## Operating style

Borrow the useful part of the `nanochat` workflow:

- change one thing per batch
- run a smaller proof batch first
- watch throughput plus validation/OOS metrics, not just training reward
- only scale to longer remote runs after the smaller batch improves the real metric

For trading work, the real metric means replay/holdout robustness and drawdown, not in-sample reward alone.
