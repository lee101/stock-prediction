# Trading Bot Scripts

A collection of scripts for trading stocks and crypto on Alpaca Markets and Binance.

## History & Background

This neural network trading bot trades stocks (long/short) and crypto (long-only) daily at market open/close. It successfully grew my portfolio from $38k to $66k over several months in favorable conditions at the end of 2024.

The bot uses the Amazon Chronos model for time series forecasting.

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
