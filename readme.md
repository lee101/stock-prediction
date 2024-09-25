


## readme

npm install -g selenium-side-runner
npm install -g chromedriver

# prepare machine
sudo apt-get install libsqlite3-dev -y

sudo apt-get update
sudo apt-get install libxml2-dev
sudo apt-get install libxslt1-dev


### Scripts
clear out positions at bid/ask (much more cost effective than market orders)

PYTHONPATH=$(pwd) python ./scripts/alpaca_cli.py close_all_positions

##### cancel an order with a linear ramp

PYTHONPATH=$(pwd) python scripts/alpaca_cli.py backout_near_market BTCUSD

##### ramp into a position

PYTHONPATH=$(pwd) python scripts/alpaca_cli.py ramp_into_position ETHUSD

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

with a pip cache local dir

```
pip install --cache_dir=/media/lee/crucial/pipcache -r requirements.txt
```


add these lines for cache
vi ~/.config/pip/pip.conf
[global]
cache-dir = /media/lee/crucial/pipcache
no-cache-dir = false
