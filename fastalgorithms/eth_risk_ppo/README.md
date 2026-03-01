# ETH Risk-Aware PPO Experiment

This experiment package focuses on ETH hourly trading research with:

- `gymrl` PPO training on `trainingdatahourly/ETHUSD.csv`
- risk-aware reward shaping (`drawdown`, `CVaR`, `turnover`, uncertainty)
- BF16 policy autocast for faster training on modern GPUs
- reproducible simulator-vs-live matching reports

## Files

- `run_train_local.sh`: launch a local ETH PPO training run.
- `run_train_remote.sh`: pull latest git on remote and launch detached training.
- `compare_live_vs_sim_eth.py`: compare simulator fills against live ETH orders.
- `configs/eth_risk_ppo_bf16.json`: baseline training hyperparameters.

## Local Training

```bash
bash fastalgorithms/eth_risk_ppo/run_train_local.sh 200000 eth_risk_ppo_$(date -u +%Y%m%d_%H%M%S)
```

Arguments:

1. timesteps (default `150000`)
2. run name (default timestamped)

Artifacts are written under:

- `fastalgorithms/eth_risk_ppo/artifacts/<run_name>/`
- `fastalgorithms/eth_risk_ppo/runs/<run_name>/` (TensorBoard)

## Remote Launch

```bash
bash fastalgorithms/eth_risk_ppo/run_train_remote.sh 300000 eth_risk_ppo_remote_$(date -u +%Y%m%d_%H%M%S)
```

Environment overrides:

- `REMOTE_HOST` (default `administrator@93.127.141.100`)
- `REMOTE_DIR` (default `/nvme0n1-disk/code/stock-prediction`)
- `REMOTE_SSH` (default `ssh -o StrictHostKeyChecking=no`)

The remote script performs `git fetch` + `git pull --rebase` and starts training with `nohup`.

No credentials are embedded in this repository.

## Live-vs-Sim Comparison

```bash
source .venv312/bin/activate
PAPER=0 python fastalgorithms/eth_risk_ppo/compare_live_vs_sim_eth.py \
  --sim-fills analysis/eth_prod_match_hourly_20260301/fills.csv \
  --hours 24 \
  --symbol ETHUSD \
  --output analysis/eth_prod_match_hourly_20260301/live_vs_sim_report_v2.json
```

