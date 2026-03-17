# HuggingFace vs PufferLib

## Goal

Run a fair comparison between the current HuggingFace-style attention trainer stack and the current PufferLib-style RL stack on the same stock subset and the same realistic execution assumptions.

This comparison started on **March 17, 2026 (UTC)** and should keep running through the day with iterative updates as new runs finish.

## Current Caveat

- The local `PufferLib4/` directory is currently empty.
- `pufferlibtraining3` therefore resolves against the installed/local `pufferlib` package path instead of a populated checked-out `PufferLib4/` tree.
- For now, this document uses **"PufferLib4-style path"** to mean the `pufferlibtraining3` PPO stack and records the exact command/env so the comparison is reproducible and the limitation is explicit.

## Benchmark Protocol

- Universe: `AAPL`, `MSFT`, `NVDA`, `AMZN`, `TSLA`
- Source data root: `trainingdata`
- HF train split: `analysis/hf_vs_pufferlib/data/train_5sym_until_2024-12-31`
  - Built with [`prepare_hf_puffer_benchmark_data.py`](/nvme0n1-disk/code/stock-prediction/prepare_hf_puffer_benchmark_data.py)
  - Source files: `trainingdata/{AAPL,MSFT,NVDA,AMZN,TSLA}.csv`
  - Train window: `2021-12-03` through `2024-12-31`
- Shared holdout window: `2025-01-02` through `2025-11-28`
- Asset class: US equities only for the first pass
- Shared simulator: [`pufferlibtraining3/envs/market_env.py`](/nvme0n1-disk/code/stock-prediction/pufferlibtraining3/envs/market_env.py)
- Shared cost assumptions:
  - Trading fee: `5 bps` (`0.0005`)
  - Slippage: `5 bps`
  - Intraday leverage cap: `4x`
  - Overnight leverage cap: `2x`
  - Annual leverage financing: `6.5%`
- Shared evaluation script: [`compare_hf_pufferlib_marketsim.py`](/nvme0n1-disk/code/stock-prediction/compare_hf_pufferlib_marketsim.py)
- Primary execution mode: `open_close`
  - Reason: aligns with day-end deleveraging and the current stock leverage plan
- Secondary execution mode: `maxdiff`
  - Reason: stresses within-bar fill realism
- Primary metric: Sortino
- Secondary metrics:
  - total return
  - annualized return
  - max drawdown
  - trading cost
  - fill rate
  - turnover
- HF launcher: [`analysis/hf_vs_pufferlib/run_hf_local.sh`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/run_hf_local.sh)
- Puffer launcher: [`analysis/hf_vs_pufferlib/run_puffer_local.sh`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/run_puffer_local.sh)
- HF configs:
  - [`analysis/hf_vs_pufferlib/configs/hf_5sym_nototo_adamw_2024train.json`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/configs/hf_5sym_nototo_adamw_2024train.json)
  - [`analysis/hf_vs_pufferlib/configs/hf_5sym_toto_muon_2024train.json`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/configs/hf_5sym_toto_muon_2024train.json)

## Local Validation

### Evaluator Tests

- Command:
```bash
source .venv313/bin/activate && python -m pytest tests/test_compare_hf_pufferlib_marketsim.py -q
```
- Result: `2 passed`

### Dataset Split Regression

- Command:
```bash
source .venv313/bin/activate && python -m pytest tests/test_prepare_hf_puffer_benchmark_data.py -q
```
- Result: `1 passed`
- Real split build:
```bash
source .venv313/bin/activate && python prepare_hf_puffer_benchmark_data.py \
  --source-root trainingdata \
  --output-dir analysis/hf_vs_pufferlib/data/train_5sym_until_2024-12-31 \
  --symbols AAPL,MSFT,NVDA,AMZN,TSLA \
  --end-date 2024-12-31
```
- Result:
  - each symbol produced `772` train rows
  - manifest written to [`analysis/hf_vs_pufferlib/data/train_5sym_until_2024-12-31/manifest.json`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/data/train_5sym_until_2024-12-31/manifest.json)

### Puffer Smoke

- Training command:
```bash
source .venv313/bin/activate && python -m pufferlibtraining3.pufferrl \
  --symbol AAPL \
  --data-root trainingdata \
  --device cpu \
  --backend Serial \
  --num-envs 2 \
  --total-timesteps 1024 \
  --batch-size 1024 \
  --minibatch-size 256 \
  --update-epochs 1 \
  --bptt-horizon 64 \
  --learning-rate 1e-4 \
  --mode open_close \
  --slip-bps 5 \
  --trading-fee 0.0005 \
  --log-json /tmp/pufferlib_hf_compare_smoke_puffer.json
```
- Shared-simulator eval command:
```bash
source .venv313/bin/activate && python compare_hf_pufferlib_marketsim.py evaluate-puffer \
  --summary-json /tmp/pufferlib_hf_compare_smoke_puffer.json \
  --symbol AAPL \
  --data-root trainingdata \
  --device cpu \
  --output-json /tmp/pufferlib_hf_compare_smoke_eval_puffer.json
```
- Result:
  - `total_return=-3.79%`
  - `sortino=-1.2384`
  - `max_drawdown=-4.11%`

### HF Smoke

- Training command:
```bash
source .venv313/bin/activate && python - <<'PY'
from hftraining.config import create_config
from hftraining.run_training import run_training

cfg = create_config('quick_test')
cfg.data.symbols = ['AAPL', 'MSFT', 'NVDA']
cfg.data.data_dir = 'trainingdata/puffer_subset'
cfg.data.use_toto_forecasts = False
cfg.data.sequence_length = 32
cfg.data.prediction_horizon = 1
cfg.training.max_steps = 16
cfg.training.batch_size = 4
cfg.training.gradient_accumulation_steps = 1
cfg.training.learning_rate = 1e-4
cfg.training.optimizer = 'adamw'
cfg.training.transaction_cost_bps = 5.0
cfg.training.profit_loss_weight = 0.2
cfg.evaluation.eval_steps = 8
cfg.evaluation.save_steps = 8
cfg.evaluation.logging_steps = 4
cfg.output.output_dir = '/tmp/hf_vs_pufferlib_smoke_subset'
cfg.output.logging_dir = '/tmp/hf_vs_pufferlib_smoke_subset/logs'
cfg.output.cache_dir = '/tmp/hf_vs_pufferlib_smoke_subset/cache'
cfg.system.device = 'cpu'
cfg.system.auto_batch_size = False
run_training(cfg)
PY
```
- Shared-simulator compare command:
```bash
source .venv313/bin/activate && python compare_hf_pufferlib_marketsim.py compare \
  --symbol AAPL \
  --data-root trainingdata \
  --mode open_close \
  --hf-checkpoint /tmp/hf_vs_pufferlib_smoke_subset/final_model.pth \
  --hf-processor-path /tmp/hf_vs_pufferlib_smoke_subset/data_processor.pkl \
  --puffer-summary-json /tmp/pufferlib_hf_compare_smoke_puffer.json \
  --device cpu \
  --output-json /tmp/hf_vs_pufferlib_smoke_compare.json
```
- Result:
  - HF: `total_return=-65.97%`, `sortino=-0.7157`
  - Puffer smoke reference: `total_return=-3.79%`, `sortino=-1.2384`
  - This smoke is only a pipeline validation. The training budgets are too small to use the absolute metrics as a real model ranking.

### HF Config Smoke

- Stable config smoke:
```bash
source .venv313/bin/activate && python - <<'PY'
from hftraining.config import ExperimentConfig
from hftraining.run_training import run_training

cfg = ExperimentConfig.load('analysis/hf_vs_pufferlib/configs/hf_5sym_nototo_adamw_2024train.json')
cfg.training.max_steps = 4
cfg.training.batch_size = 2
cfg.training.dataloader_num_workers = 0
cfg.evaluation.eval_steps = 2
cfg.evaluation.save_steps = 4
cfg.evaluation.logging_steps = 1
cfg.output.output_dir = '../tmp/hf_5sym_nototo_smoke'
cfg.output.logging_dir = '../tmp/hf_5sym_nototo_smoke/logs'
cfg.output.cache_dir = '../tmp/hf_5sym_nototo_smoke/cache'
cfg.system.device = 'cpu'
cfg.system.auto_batch_size = False
run_training(cfg)
PY
```
- Result: completed successfully in about `8s`
- Toto note:
  - the Toto config starts and writes config output, but its warm-up/bootstrap stage is much slower than the no-Toto config
  - I kept the Toto config in the queued long-run path, but the no-Toto config is the stable first live benchmark

### Puffer Date-Filter Regression

- Bug found on March 17, 2026:
  - `pufferlibtraining3` and `fastmarketsim` both compared naive filter dates against UTC-indexed CSVs and crashed
- Fix:
  - [`pufferlibtraining3/envs/market_env.py`](/nvme0n1-disk/code/stock-prediction/pufferlibtraining3/envs/market_env.py)
  - [`fastmarketsim/env.py`](/nvme0n1-disk/code/stock-prediction/fastmarketsim/env.py)
- Regression test:
```bash
source .venv313/bin/activate && python -m pytest tests/test_market_env_date_filters.py -q
```
- Result: `2 passed`
- Filtered fast-backend smoke:
```bash
source .venv313/bin/activate && python -m pufferlibtraining3.pufferrl \
  --symbol AAPL \
  --data-root trainingdata \
  --start-date 2021-12-03 \
  --end-date 2024-12-31 \
  --mode open_close \
  --device cpu \
  --backend Serial \
  --env-backend fast \
  --num-envs 2 \
  --total-timesteps 256 \
  --batch-size 256 \
  --minibatch-size 128 \
  --update-epochs 1 \
  --bptt-horizon 64 \
  --learning-rate 1e-4 \
  --slip-bps 5 \
  --trading-fee 0.0005 \
  --log-json /tmp/puffer_fast_filtered_smoke.json
```
- Result: completed successfully with `SPS≈631`

## Live Run Ledger

### Access Mode

- Remote SSH attempts to `administrator@93.127.141.100` failed earlier with `Permission denied (publickey,password)`.
- Because this local box already exposes an `RTX 5090`, the live benchmark is running locally instead of remotely.

### HF Queue

- `tmux` session: `hf_vs_pufferlib_hf_20260317`
- Launch command:
```bash
tmux new-session -d -s hf_vs_pufferlib_hf_20260317 \
  'cd /nvme0n1-disk/code/stock-prediction && ./analysis/hf_vs_pufferlib/run_hf_local.sh > analysis/remote_logs/hf_vs_pufferlib_hf_queue_20260317.log 2>&1'
```
- Queue wrapper log: [`analysis/remote_logs/hf_vs_pufferlib_hf_queue_20260317.log`](/nvme0n1-disk/code/stock-prediction/analysis/remote_logs/hf_vs_pufferlib_hf_queue_20260317.log)
  - Note: the wrapper log is intentionally quiet because each run writes to its own `train.log`
- Current active run: `hf_5sym_nototo_adamw_2024train`
  - Run directory: [`analysis/hf_vs_pufferlib/runs/hf_5sym_nototo_adamw_2024train`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/runs/hf_5sym_nototo_adamw_2024train)
  - Train log: [`analysis/hf_vs_pufferlib/runs/hf_5sym_nototo_adamw_2024train/train.log`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/runs/hf_5sym_nototo_adamw_2024train/train.log)
  - Exact train command: [`analysis/hf_vs_pufferlib/runs/hf_5sym_nototo_adamw_2024train/train_command.txt`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/runs/hf_5sym_nototo_adamw_2024train/train_command.txt)
  - Current status as of **March 17, 2026 09:56 UTC**:
    - still running
    - passed step `2000` and wrote [`analysis/hf_vs_pufferlib/runs/hf_5sym_nototo_adamw_2024train/checkpoint_step_2000.pth`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/runs/hf_5sym_nototo_adamw_2024train/checkpoint_step_2000.pth)
    - later reached step `4500`, where validation logged `eval_loss=1.2664`, `action_loss=1.0357`, `price_loss=0.4614`
    - throughput is roughly `13-15` train steps/s between eval blocks
    - one transient large negative training loss spike appeared around step `2700`, so the profit-weighted loss needs monitoring even though validation stayed stable
- Queued next in the same HF session:
  - `hf_5sym_toto_muon_2024train`
  - output path will be [`analysis/hf_vs_pufferlib/runs/hf_5sym_toto_muon_2024train`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/runs/hf_5sym_toto_muon_2024train)

### Puffer Queue

- `tmux` session: `hf_vs_pufferlib_puffer_20260317`
- Launch command:
```bash
tmux new-session -d -s hf_vs_pufferlib_puffer_20260317 \
  'cd /nvme0n1-disk/code/stock-prediction && ./analysis/hf_vs_pufferlib/run_puffer_local.sh > analysis/remote_logs/hf_vs_pufferlib_puffer_queue_20260317.log 2>&1'
```
- Queue wrapper log: [`analysis/remote_logs/hf_vs_pufferlib_puffer_queue_20260317.log`](/nvme0n1-disk/code/stock-prediction/analysis/remote_logs/hf_vs_pufferlib_puffer_queue_20260317.log)
  - Note: like the HF queue, progress is recorded in each run directory rather than the wrapper log
- Current active run: `puffer_open_close_AAPL_2024train`
  - Run directory: [`analysis/hf_vs_pufferlib/runs/puffer_open_close_AAPL_2024train`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/runs/puffer_open_close_AAPL_2024train)
  - Train log: [`analysis/hf_vs_pufferlib/runs/puffer_open_close_AAPL_2024train/train.log`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/runs/puffer_open_close_AAPL_2024train/train.log)
  - Exact train command: [`analysis/hf_vs_pufferlib/runs/puffer_open_close_AAPL_2024train/train_command.txt`](/nvme0n1-disk/code/stock-prediction/analysis/hf_vs_pufferlib/runs/puffer_open_close_AAPL_2024train/train_command.txt)
  - Current status as of **March 17, 2026 09:56 UTC**:
    - still running
    - reached epoch `2` / `global_step=196608`
    - logged `SPS=1533.56`
    - dashboard ETA for the first run fell to about `19m35s`
    - latest environment snapshot: `equity=0.4840`, `trading_cost=0.00115`, `financing_cost=0.00015`, `deleverage_notional=0.4753`
- Remaining queue after the active run:
  - `AAPL maxdiff`
  - `MSFT open_close`
  - `MSFT maxdiff`
  - `NVDA open_close`
  - `NVDA maxdiff`
  - `AMZN open_close`
  - `AMZN maxdiff`
  - `TSLA open_close`
  - `TSLA maxdiff`

## Next Steps

- Let the current HF and Puffer queues continue running through March 17, 2026 UTC.
- When the first non-smoke checkpoints finish, run shared-simulator evaluation with:
  - `open_close`
  - `maxdiff`
  - HF action modes `alloc_only` and `alloc_signed_by_logits`
- Update this document with per-symbol and average metrics, then promote the strongest configuration.
