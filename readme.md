
A collection of scripts for trading stocks and crypto on alpaca markets and binance.

## History & Background

Neural network trading bot that trades stocks (long/short) and crypto (long-only) daily at market open/close. Successfully grew my portfolio from $38k to $66k over several months in favorable conditions at the end of 2024.

Uses Amazon Chronos model for time series forecasting.

## Getting Started

npm install -g selenium-side-runner
npm install -g chromedriver

# prepare machine
sudo apt-get install libsqlite3-dev -y

sudo apt-get update
sudo apt-get install libxml2-dev
sudo apt-get install libxslt1-dev


### Scripts
clear out positions at bid/ask (much more cost effective than market orders)

```
PYTHONPATH=$(pwd) python ./scripts/alpaca_cli.py close_all_positions
```

##### cancel an order with a linear ramp

```
PYTHONPATH=$(pwd) python scripts/alpaca_cli.py backout_near_market BTCUSD
```
##### ramp into a position

```
PYTHONPATH=$(pwd) python scripts/alpaca_cli.py ramp_into_position ETHUSD
```

# at a time e.g. sudo apt install at

using linux command at

```
echo "PYTHONPATH=$(pwd) python ./scripts/alpaca_cli.py ramp_into_position TSLA" | at 3:30
```

show/cancel jobs with atq

(.env) (base) lee@lee-top:/media/lee/crucial1/code/stock$ atq
1       Fri Oct 18 03:00:00 2024 a lee
2       Fri Oct 18 03:30:00 2024 a lee
(.env) (base) lee@lee-top:/media/lee/crucial1/code/stock$ atrm 1
(.env) (base) lee@lee-top:/media/lee/crucial1/code/stock$ atq
2       Fri Oct 18 03:30:00 2024 a lee

##### cancel any duplicate orders/bugs

PYTHONPATH=$(pwd) python ./scripts/cancel_multi_orders.py


- proper datastores refreshed data
- dynamic config

neural networks
- select set of trades to make
- margin
- takeprofit
- roughly at eod only to close stock positions violently



check if numbers are flipped and if so do something?

### crypto issues
crypto can be only traded non margin for some time so this server should be used that loops/does market orders:

now they do?

fees though so
use binance for crypto try not trade it on alpaca?

 ./.env/bin/gunicorn -k uvicorn.workers.UvicornWorker -b :5050 src.crypto_loop.crypto_order_loop_server:app --timeout 1800 --workers 1


### install requirements

```
uv pip install requirements.txt
```

Run the stock trading bot 
``` 
python trade_stock_e2e.py
```

Run the tests

```
pytest .
```

Run a simulation

```
PYTHONPATH=$(pwd) python backtest_test3_inline.py
```

### Please support me!

You can support us by purchasing [Netwrck](https://netwrck.com/).

Also checkout [AIArt-Generator.art](https://AIArt-Generator.art) and [Netwrck.com](https://netwrck.com)
Also checkout [Helix.app.nz](https://helix.app.nz)
