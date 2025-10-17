# Trading Bot Scripts

A collection of scripts for trading stocks and crypto on Alpaca Markets and Binance.

## History & Background

This neural network trading bot trades stocks (long/short) and crypto (long-only) daily at market open/close. It successfully grew my portfolio from $38k to $66k over several months in favorable conditions at the end of 2024.

The bot uses the Amazon Chronos model for time series forecasting.

Breakdown of how it works

https://www.youtube.com/watch?v=56c3OhqJDJk&list=PLVovYLPm_feCybDdwSeXUCCTHZaLPoXZJ&index=9

## Getting Started

```bash
npm install -g selenium-side-runner
npm install -g chromedriver
```

### Prepare Machine

```bash
sudo apt-get install libsqlite3-dev -y
sudo apt-get update
sudo apt-get install libxml2-dev
sudo apt-get install libxslt1-dev
```

### Scripts

Clear out positions at bid/ask (much more cost-effective than market orders):

```bash
PYTHONPATH=$(pwd) python ./scripts/alpaca_cli.py close_all_positions
```

Cancel an order with a linear ramp:

```bash
PYTHONPATH=$(pwd) python scripts/alpaca_cli.py backout_near_market BTCUSD
```

Ramp into a position:

```bash
PYTHONPATH=$(pwd) python scripts/alpaca_cli.py ramp_into_position ETHUSD
```

## Training Pipelines

### PufferLib multi-stage RL (Toto-assisted)

- Boot the UV-managed environment and launch the end-to-end RL stack (generic forecaster → specialists → vectorised RL) with sensible defaults:

  ```bash
  uv run python pufferlibtraining/train_ppo.py \
    --trainingdata-dir trainingdata \
    --output-dir pufferlibtraining/models/$(date +%Y%m%d)_puffer_stage \
    --tensorboard-dir pufferlibtraining/logs/$(date +%Y%m%d)_puffer_stage
  ```

- The command checks your CSV coverage before training and drops a JSON summary plus checkpoints under `pufferlibtraining/models/…`. Override `--skip-base/--skip-specialists/--skip-rl` or supply `--base-checkpoint` to resume midway. For the ultra-long run we previously used, combine the flags shown below:

  ```bash
  uv run python pufferlibtraining/train_ppo.py \
    --skip-base \
    --base-checkpoint pufferlibtraining/models/toto_run/base_models/base_checkpoint_20251017_162009.pth \
    --skip-specialists \
    --trainingdata-dir trainingdata \
    --output-dir pufferlibtraining/models/toto_run_rl_lowcost_400 \
    --tensorboard-dir pufferlibtraining/logs/toto_run_rl_lowcost_400 \
    --rl-epochs 400 \
    --rl-batch-size 256 \
    --rl-learning-rate 7e-4 \
    --rl-optimizer muon_mix \
    --rl-warmup-steps 400 \
    --rl-min-lr 1e-5 \
    --rl-grad-clip 0.25 \
    --transaction-cost-bps 1 \
    --risk-penalty 0.0 \
    --summary-path pufferlibtraining/models/toto_run_rl_lowcost_400/summary.json
  ```

### HFTraining RL + Kronos adapters

- Quick CPU/GPU smoke test (validated on October 18 2025 at 08:51 NZDT) that runs 10 short episodes against `trainingdata/SPY.csv` and prints validation metrics:

  ```bash
  uv run python hftraining/quick_realistic_test.py
  ```

- Expected log tail from the latest run:

  ```
  Final Validation Results:
  Return: 0.00%
  Sharpe Ratio: 0.000
  Max Drawdown: 2.30%
  ```

- For configurable HuggingFace-style fine-tuning (including Kronos/Toto feature injection), pass a JSON config. The bundled AAPL smoke config writes artefacts under `hftraining/output/quick_test_aapl`:

  ```bash
  uv run python -m hftraining.run_training \
    --config hftraining/configs/quick_test_aapl.json
  ```

  Adjust `data.symbols`, `data.use_toto_forecasts`, and the `output.*` paths inside the JSON to target Kronos-assisted runs with your own symbols.

### Kronos fine-tuning (kronostraining)

- Install Kronos extras once:

  ```bash
  uv pip install -r external/kronos/requirements.txt
  ```

- Launch the CUDA-only fine-tuner (defaults to NeoQuasar/Kronos-small and writes to `kronostraining/artifacts/`):

  ```bash
  uv run python -m kronostraining.run_training \
    --data-dir trainingdata \
    --output-dir kronostraining/artifacts \
    --lookback 64 \
    --horizon 30 \
    --validation-days 30 \
    --epochs 3
  ```

  The script automatically evaluates the final checkpoint on the held-out window and emits MAE/RMSE/MAPE summaries.

### Toto fine-tuning (tototraining)

- Point the trainer at your train/val splits (GPU strongly recommended). The options below freeze the backbone and keep checkpoints plus HF exports under `tototraining/checkpoints/demo_run`:

  ```bash
  uv run python tototraining/train.py \
    --train-root trainingdata/train \
    --val-root trainingdata/val \
    --output-dir tototraining/checkpoints/demo_run \
    --epochs 1 \
    --batch-size 2 \
    --context-length 1024 \
    --prediction-length 64 \
    --loss heteroscedastic
  ```

- For the full-production recipe with Muon + cosine schedule, switch to the bundled helper:

  ```bash
  uv run python tototraining/run_gpu_training.py
  ```

  This keeps the top checkpoints, logs metrics to `tensorboard_logs/`, and records validation/test summaries in `tototraining/checkpoints/gpu_run/final_metrics.json`.

### Schedule Tasks

Using the Linux `at` command:

```bash
echo "PYTHONPATH=$(pwd) python ./scripts/alpaca_cli.py ramp_into_position TSLA" | at 3:30
```

Show/cancel jobs with `atq`:

```bash
atq
atrm 1
atq
```

Cancel any duplicate orders/(need to run this incase of bugs):

```bash
PYTHONPATH=$(pwd) python ./scripts/cancel_multi_orders.py
```

### Stock CLI

The `stock_cli.py` utility provides a consolidated interface for observing live risk state, probes, and portfolio history without opening dashboards.

```bash
# Show account status, positions, and risk thresholds
PYTHONPATH=$(pwd) python stock_cli.py status

# Render an ASCII portfolio value graph in the terminal
PYTHONPATH=$(pwd) python stock_cli.py risk-text --limit 120 --width 80

# Generate a PNG chart of portfolio value and global risk thresholds
PYTHONPATH=$(pwd) python stock_cli.py plot-risk --output portfolio_risk.png

# Inspect current probe trades and learning state (optionally target a specific suffix)
PYTHONPATH=$(pwd) python stock_cli.py probe-status --tz UTC --suffix sim
```

### Notes and todos

- Proper datastores refreshed data
- Dynamic config

Neural networks:
- Select set of trades to make
- Margin
- Take profit
- Roughly at EOD only to close stock positions violently

Check if numbers are flipped and if so, do something?

### Crypto Issues

Crypto can only be traded non-margin for some time, cant be shorted in alpaca, so this server should be used that loops/does market orders in Binance instead which is also better low fee:

```bash
./.env/bin/gunicorn -k uvicorn.workers.UvicornWorker -b :5050 src.crypto_loop.crypto_order_loop_server:app --timeout 1800 --workers 1
```

### Install Requirements

```bash
uv pip install requirements.txt
```

### Run the Stock Trading Bot

```bash
python trade_stock_e2e.py
```

### Run the Tests

```bash
pytest .
```

## Training Optimizer Toolkit

Advanced optimizer experiments now live in `traininglib/`. The module ships with:

- A registry that exposes Adam/AdamW, SGD, Lion, Adafactor, Shampoo, and Muon with sensible defaults.
- `traininglib.benchmarking.RegressionBenchmark` for quick, repeatable regression checks (including multi-seed summaries).
- Hugging Face helpers (`traininglib.hf_integration`) so you can wire the same optimizer choices into a `Trainer`.
- A CLI (`python -m traininglib.benchmark_cli`) that prints aggregated losses so you can compare optimizers the moment new ideas land.

Example usage inside a Hugging Face script:

```python
from transformers import Trainer, TrainingArguments
from traininglib.hf_integration import build_hf_optimizers

optimizer, scheduler = build_hf_optimizers(model, "shampoo")
trainer = Trainer(model=model, args=training_args, optimizers=(optimizer, scheduler))
trainer.train()
```

The accompanying tests in `tests/traininglib/` run a small benchmark to confirm Shampoo and Muon at least match AdamW before you swap anything into production training loops.

You can also evaluate the wider optimizer set locally:

```bash
python -m traininglib.benchmark_cli --optimizers adamw shampoo muon lion --runs 3
```

### Run a Simulation

```bash
PYTHONPATH=$(pwd) python backtest_test3_inline.py
```

## HFTraining Quick Run (2025-10-17)

The `hftraining` pipeline now has a reproducible quick-test recipe paired with a fast RL sanity check. The latest run used seeded configs so replays line up with the logged metrics.

- Prepare a single-symbol data slice for the quick test (already staged under `hftraining/local_data/quick_aapl/AAPL.csv`).
- Launch the lightweight transformer training run:

```bash
python hftraining/run_training.py \
  --config_file hftraining/configs/quick_test_aapl.json \
  --experiment_name hf_quick_aapl_20251017
```

  - Artifacts land in `hftraining/output/quick_test_aapl/` (config snapshot, `final_model.pth`, `run_report.md` with the full metric dump).
  - The Oct 17 run reached a best eval loss of 3.2758 at step 300, with final training loss 2.0366 over ~6 s on a single RTX 3090 Ti.

- Evaluate the seeded RL backtest that consumes the same feature stack and emits P&L diagnostics:

```bash
python hftraining/quick_realistic_test.py
python - <<'PY'
from hftraining.quick_realistic_test import quick_test
import numpy as np, json
model, metrics, equity_curve = quick_test()
initial = equity_curve[0]
returns = np.diff(equity_curve) / equity_curve[:-1]
avg_daily_return = returns.mean()
annual_return = (1 + avg_daily_return) ** 252 - 1
payload = {
    "initial_capital": float(initial),
    "final_capital": float(equity_curve[-1]),
    "avg_daily_return_pct": float(avg_daily_return * 100),
    "avg_daily_pnl": float(avg_daily_return * initial),
    "annual_return_pct": float(annual_return * 100),
    "annual_pnl": float(annual_return * initial),
}
with open(\"hftraining/output/rl_quick_test_metrics_20251017.json\", \"w\") as f:
    json.dump(payload, f, indent=2)
PY
```

  - The scripted run (seed = 42) starts at \$25 000 and finishes at \$24 425, so the simulated strategy is currently losing ≈\$5.04 per trading day on average (−0.020% daily, −4.96% annualized, ≈\$1.24 k/year) with a Sharpe of −1.50 and drawdown capped near 2.3%.
  - Detailed P&L figures live in `hftraining/output/rl_quick_test_metrics_20251017.json` alongside commission (\$0.56) and slippage (\$569.71) estimates.

# todos

better forecasting transformers
better trading alg as on avg its good and accurate but following the sign is loosing some of the power of the model - need a better fuzzy strategy that better exploits the fact that the model is correct on average

### Please Support Me!

You can support us by purchasing [Netwrck](https://netwrck.com/) an AI agent maker and art generator.

- Art Generator/photo editor: [AIArt-Generator.art](https://AIArt-Generator.art)
- [Helix.app.nz](https://helix.app.nz) a dashboard builder
- [Text-Generator.io](https://text-generator.io) an API server for vision language and speech models




### extra stuff

trade simulator

TRADE_STATE_SUFFIX=sim python marketsimulator/run_trade_loop.py --symbols AAPL MSFT NVDA BTCUSD ETHUSD --steps 10 --step-size 6 --initial-cash 100000



Have a thing that like a bindary step for the whole portfolil and probe trades
  Tests

  - python -m pytest tests/test_portfolio_risk.py tests/test_probe_transitions.py

  Next Steps

      1. Run python stock_cli.py plot-risk -o portfolio_risk.png to confirm the chart output.


REAL_TESTING=true flag uses faster hparams and torch compile and bf16 so its faster during marketsimulator tests but its turned off for trade_stock_e2e.py
 python stock_cli.py status

train a new toto

uv run python -m tototraining.run_gpu_training \
  --compile \
  --optim muon_mix \
  --device_bs 4 \
  --grad_accum 4 \
  --lr 3e-4 \
  --warmup_steps 2000 \
  --max_epochs 24 \
  --report runs/toto_gpu_report.md

## StockAgent Allocation Workflows

### `stockagentcombined` (Toto + Kronos blend)

- **Environment prep**
  - `uv sync`
  - Install the vendored Toto package once: `uv pip install -e toto`
  - Ensure `external/kronos` is populated (already vendored) and GPU drivers are available if you intend to run the full models.
- **Refresh hyperparameters**
  - Regenerate Toto/Kronos sweeps as needed (see `hftraining/` or your preferred pipeline) and drop the resulting JSON artefacts under `hyperparams/{toto,kronos,best}`. The combined generator will hot-load whatever is present in that tree.
  - Sanity check the loader with a quick unit run: `uv run pytest tests/test_stockagentcombined.py -k forecast`
- **Run the market simulator**
  - Full speed (requires real models):
    ```bash
    uv run python -m stockagentcombined.simulation \
      --symbols AAPL MSFT NVDA AMD \
      --lookback-days 180 \
      --simulation-days 5 \
      --starting-cash 1000000 \
      --local-data-dir trainingdata \
      --allow-remote-data  # drop this flag to stay offline
    ```
  - Smoke testing without GPU: follow the inline stub recipe in `stockagentcombined/results.md` (uses synthetic Toto/Kronos adapters and runs entirely on CPU).
- **Record outcomes**
  - Persist headline metrics and command lines in `stockagentcombined/results.md`. The file currently captures the 2025-10-17 stub run used while drafting this guide.

### `stockagent2` (Black–Litterman allocator)

- **Prerequisites**
  - Complete the `stockagentcombined` setup above so the combined forecasts are available.
  - Run allocator unit tests: `uv run pytest tests/test_stockagent2`
- **Allocate with real forecasts**
  - You can drive the end-to-end allocator via `stockagent2.agentsimulator.runner.run_pipeline_simulation`. Example:
    ```bash
    uv run python - <<'PY'
    from stockagent2.agentsimulator.runner import run_pipeline_simulation, RunnerConfig
    from stockagent2.config import OptimizationConfig, PipelineConfig

    simulator = run_pipeline_simulation(
        runner_config=RunnerConfig(
            symbols=("AAPL", "MSFT", "NVDA", "AMD"),
            lookback_days=200,
            simulation_days=5,
            starting_cash=1_000_000,
            allow_remote_data=False,
        ),
        optimisation_config=OptimizationConfig(),
        pipeline_config=PipelineConfig(),
    )
    if simulator is not None:
        print(f\"trades={len(simulator.trade_log)} total_fees={simulator.total_fees:.2f}\")
    PY
    ```
  - The helper returns an `AgentSimulator`; inspect `simulator.trade_log`, `simulator.equity_curve`, or rerun the plans against a fresh simulator for analytics.
- **Market simulator hooks**
  - Both agents share `stockagent.agentsimulator.AgentSimulator`. You can pass the trading plans emitted by `PipelinePlanBuilder` straight into it to benchmark execution paths side-by-side.
- **Document results**
  - Append the latest allocator runs to `stockagent2/results.md`. The current entry shows the CPU-only stub run (net-neutral configuration) executed on 2025-10-17; swap the stub adapters with real models when running in production.
