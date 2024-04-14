


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

order canceller that cancels duplicate orders

### cancel any duplicate orders/bugs

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
fees though
use binance for crypto try not trade it on alpaca?

 ./.env/bin/gunicorn -k uvicorn.workers.UvicornWorker -b :5050 src.crypto_loop.crypto_order_loop_server:app --timeout 1800 --workers 1
