# API Reference

This service exposes the trading bot order loop plus forecast and recommendation data over HTTP. The API is a lightweight JSON service running locally; it does **not** use Firebase.

## Base URL

```
http://localhost:5050
```

## Authentication

None by default. Deploy behind your own auth/reverse proxy if you need external access.

## Conventions

- All responses are JSON.
- `symbols` query parameters accept a comma-separated list (e.g. `AAPL,MSFT,BTCUSD`).
- When no forecast file is available, the forecast endpoints return an empty list with `count: 0`.

## Forecasts API

### Forecast data source

The API reads the latest `predictions-*.csv` (or `predictions.csv`) from the following locations, in order:

- `strategy_state/results/`
- `results/`

This mirrors the bot’s live and simulated forecast output files.

### GET `/api/v1/forecasts/latest`
Return the most recent per-symbol forecast snapshot.

Query params:
- `symbols` (optional): comma-separated symbols to include.

Example response:

```json
{
  "generated_at": "2025-02-21T02:53:14.669196+00:00",
  "source_file": "/home/lee/code/stock/results/predictions-sim.csv",
  "count": 2,
  "forecasts": [
    {
      "symbol": "AAPL",
      "predicted": {"close": 202.9, "high": 210.24, "low": 200.72},
      "last": {"close": 199.88},
      "strategy": {
        "entry_takeprofit": {"profit": 0.0362, "high_price": 210.24, "low_price": 200.72},
        "maxdiffprofit": {"profit": 0.0469, "high_price": 210.24, "low_price": 200.72},
        "takeprofit": {"profit": 0.0109, "high_price": 210.24, "low_price": 200.72}
      }
    }
  ]
}
```

Notes:
- `predicted.*` are the latest model price targets when available (close/high/low).
- `last.close` is the most recent observed close when available.
- `strategy.*.profit` is the model-estimated profitability for each strategy.
- `strategy.*.high_price` / `low_price` are the suggested take-profit or range targets.

### GET `/api/v1/forecasts/prices`
Return just the trading bot prices (latest observed and predicted price targets).

Query params:
- `symbols` (optional): comma-separated symbols to include.

Example response:

```json
{
  "generated_at": "2025-02-21T02:53:14.669196+00:00",
  "source_file": "/home/lee/code/stock/results/predictions-sim.csv",
  "count": 2,
  "prices": [
    {
      "symbol": "AAPL",
      "last": {"close": 199.88},
      "predicted": {"close": 202.9, "high": 210.24, "low": 200.72}
    }
  ]
}
```

### GET `/api/v1/bot/forecasts`
Return the overall bot forecast per symbol, including a buy list and price targets.

Query params:
- `symbols` (optional): comma-separated symbols to include.
- `min_profit` (optional, default `0.0`): minimum expected profit to issue a `BUY` recommendation.

Example response:

```json
{
  "generated_at": "2025-02-21T02:53:14.669196+00:00",
  "source_file": "/home/lee/code/stock/results/predictions-sim.csv",
  "count": 2,
  "buy_list": ["AAPL"],
  "forecasts": [
    {
      "symbol": "AAPL",
      "recommendation": "BUY",
      "strategy": "maxdiffprofit",
      "expected_profit": 0.0469,
      "price_targets": {"low": 200.72, "high": 210.24, "close": 202.9}
    }
  ]
}
```

## Order Loop API

### POST `/api/v1/stock_order`
Queue a crypto order for the loop (it will submit when the loop picks it up).

Request body:

```json
{
  "symbol": "BTCUSD",
  "side": "buy",
  "price": 62000.5,
  "qty": 0.05
}
```

### GET `/api/v1/stock_orders`
Return all queued orders.

### GET `/api/v1/stock_order/{symbol}`
Return the queued order for a single symbol.

### DELETE `/api/v1/stock_order/{symbol}`
Clear the queued order for a symbol.

### GET `/api/v1/stock_order/cancel_all`
Clear all queued orders.

## Running the API server

```bash
source .venv312/bin/activate
./.env/bin/gunicorn -k uvicorn.workers.UvicornWorker -b :5050 src.crypto_loop.crypto_order_loop_server:app --timeout 1800 --workers 1
```

## Quick curl checks

```bash
curl http://localhost:5050/api/v1/forecasts/latest
curl \"http://localhost:5050/api/v1/forecasts/prices?symbols=AAPL,MSFT\"
curl \"http://localhost:5050/api/v1/bot/forecasts?min_profit=0.02\"
```
