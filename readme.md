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

### Long-horizon RL training (torch.compile + muon)

Activate the PyTorch 2.9.0 CUDA environment and launch a 400‑epoch low-cost RL run:

```bash
source .venv312uv/bin/activate
python -m pufferlibtraining.train_ppo \
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

trainings:
add training data in traininddata/ dir - we have a script for this

source .venv312uv/bin/activate

    python -m pufferlibtraining.train_ppo \
      --skip-base \
      --base-checkpoint pufferlibtraining/models/toto_run/base_models/base_checkpoint_20251017_162009.pth \
      --skip-specialists \
      --trainingdata-dir trainingdata \
      --output-dir pufferlibtraining/models/toto_run_rl_lowcost_resume \
      --tensorboard-dir pufferlibtraining/logs/toto_run_rl_lowcost_resume \
      --rl-epochs 200 \
      --rl-batch-size 256 \
      --rl-learning-rate 3e-4 \
      --rl-optimizer muon_mix \
      --rl-warmup-steps 200 \
      --rl-min-lr 1e-5 \
      --rl-grad-clip 0.25 \
      --transaction-cost-bps 1 \
      --risk-penalty 0.0 \
      --rl-initial-checkpoint-dir pufferlibtraining/models/toto_run_rl_lowcost_400v2/finetuned/portfolio_pairs \
      --summary-path pufferlibtraining/models/toto_run_rl_lowcost_resume/summary.json

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
