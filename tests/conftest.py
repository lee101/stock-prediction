#!/usr/bin/env python3
"""Pytest configuration for environments with real PyTorch installed."""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# Provide a harmless env_real stub during tests so we never import the real
# credentials or accidentally place live trades. Set USE_REAL_ENV=1 to bypass.
if os.getenv("USE_REAL_ENV", "0") not in ("1", "true", "TRUE", "yes", "YES"):
    env_stub = types.ModuleType("env_real")
    env_stub.ALP_KEY_ID = "test-key"
    env_stub.ALP_SECRET_KEY = "test-secret"
    env_stub.ALP_KEY_ID_PROD = "test-key-prod"
    env_stub.ALP_SECRET_KEY_PROD = "test-secret-prod"
    env_stub.ALP_ENDPOINT = "paper"
    env_stub.PAPER = True
    env_stub.ADD_LATEST = False
    env_stub.BINANCE_API_KEY = "test-binance-key"
    env_stub.BINANCE_SECRET = "test-binance-secret"
    env_stub.CLAUDE_API_KEY = "test-claude-key"
    env_stub.SIMULATE = True
    sys.modules["env_real"] = env_stub

# Lightweight stubs for optional third-party dependencies so unit tests never
# reach external services when the packages are missing locally.
if "loguru" not in sys.modules:
    loguru_mod = types.ModuleType("loguru")
    loguru_mod.logger = MagicMock()
    sys.modules["loguru"] = loguru_mod

if "cachetools" not in sys.modules:
    cachetools_mod = types.ModuleType("cachetools")

    def cached(**kwargs):
        def decorator(func):
            return func

        return decorator

    class TTLCache(dict):
        def __init__(self, maxsize, ttl):
            super().__init__()

    cachetools_mod.cached = cached
    cachetools_mod.TTLCache = TTLCache
    sys.modules["cachetools"] = cachetools_mod

sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("requests.exceptions", types.ModuleType("requests.exceptions"))
requests_exceptions = sys.modules["requests.exceptions"]
if not hasattr(requests_exceptions, "HTTPError"):
    requests_exceptions.HTTPError = Exception
if not hasattr(requests_exceptions, "ConnectionError"):
    requests_exceptions.ConnectionError = Exception
requests_mod = sys.modules["requests"]
if not hasattr(requests_mod, "RequestException"):
    requests_mod.RequestException = Exception
if not hasattr(requests_mod, "HTTPError"):
    requests_mod.HTTPError = requests_exceptions.HTTPError
if not hasattr(requests_mod, "Response"):
    class _Response: ...
    requests_mod.Response = _Response

if "retry" not in sys.modules:
    retry_mod = types.ModuleType("retry")

    def _retry(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    retry_mod.retry = _retry
    sys.modules["retry"] = retry_mod

if "alpaca" not in sys.modules:
    alpaca_mod = types.ModuleType("alpaca")
    alpaca_data = types.ModuleType("alpaca.data")
    alpaca_trading = types.ModuleType("alpaca.trading")
    alpaca_trading.client = types.ModuleType("client")
    alpaca_trading.enums = types.ModuleType("enums")
    alpaca_trading.requests = types.ModuleType("requests")

    alpaca_data.StockLatestQuoteRequest = MagicMock()
    alpaca_data.StockHistoricalDataClient = MagicMock()
    alpaca_data.CryptoHistoricalDataClient = MagicMock()
    alpaca_data.CryptoLatestQuoteRequest = MagicMock()
    alpaca_data.CryptoBarsRequest = MagicMock()
    alpaca_data.StockBarsRequest = MagicMock()
    alpaca_data.TimeFrame = MagicMock()
    alpaca_data.TimeFrameUnit = MagicMock()
    alpaca_data_historical = types.ModuleType("alpaca.data.historical")
    alpaca_data_historical.StockHistoricalDataClient = MagicMock()
    alpaca_data_historical.CryptoHistoricalDataClient = MagicMock()
    sys.modules["alpaca.data.historical"] = alpaca_data_historical

    alpaca_trading.OrderType = MagicMock()
    alpaca_trading.LimitOrderRequest = MagicMock()
    alpaca_trading.GetOrdersRequest = MagicMock()
    alpaca_trading.Order = MagicMock()
    alpaca_trading.client.TradingClient = MagicMock()
    alpaca_trading.TradingClient = MagicMock()
    alpaca_trading.enums.OrderSide = MagicMock()
    alpaca_trading.requests.MarketOrderRequest = MagicMock()

    sys.modules["alpaca"] = alpaca_mod
    sys.modules["alpaca.data"] = alpaca_data
    sys.modules["alpaca.trading"] = alpaca_trading
    sys.modules["alpaca.trading.client"] = alpaca_trading.client
    sys.modules["alpaca.trading.enums"] = alpaca_trading.enums
    sys.modules["alpaca.trading.requests"] = alpaca_trading.requests

sys.modules.setdefault("alpaca_trade_api", types.ModuleType("alpaca_trade_api"))
alpaca_rest = sys.modules.setdefault(
    "alpaca_trade_api.rest", types.ModuleType("alpaca_trade_api.rest")
)

if not hasattr(alpaca_rest, "APIError"):
    alpaca_rest.APIError = Exception

if "data_curate_daily" not in sys.modules:
    data_curate_daily_stub = types.ModuleType("data_curate_daily")
    _latest_prices = {}

    def download_exchange_latest_data(client, symbol):
        # store deterministic bid/ask defaults for tests
        _latest_prices[symbol] = {
            "bid": _latest_prices.get(symbol, {}).get("bid", 99.0),
            "ask": _latest_prices.get(symbol, {}).get("ask", 101.0),
        }

    def get_bid(symbol):
        return _latest_prices.get(symbol, {}).get("bid", 99.0)

    def get_ask(symbol):
        return _latest_prices.get(symbol, {}).get("ask", 101.0)

    data_curate_daily_stub.download_exchange_latest_data = download_exchange_latest_data
    data_curate_daily_stub.get_bid = get_bid
    data_curate_daily_stub.get_ask = get_ask
    sys.modules["data_curate_daily"] = data_curate_daily_stub

if "backtest_test3_inline" not in sys.modules:
    backtest_stub = types.ModuleType("backtest_test3_inline")

    def backtest_forecasts(symbol, num_simulations=10):
        import pandas as pd

        return pd.DataFrame(
            {
                "simple_strategy_return": [0.01] * num_simulations,
                "all_signals_strategy_return": [0.01] * num_simulations,
                "entry_takeprofit_return": [0.01] * num_simulations,
                "highlow_return": [0.01] * num_simulations,
                "predicted_close": [1.0] * num_simulations,
                "predicted_high": [1.2] * num_simulations,
                "predicted_low": [0.8] * num_simulations,
                "close": [1.0] * num_simulations,
            }
        )

    backtest_stub.backtest_forecasts = backtest_forecasts
    sys.modules["backtest_test3_inline"] = backtest_stub

# Allow skipping the hard PyTorch requirement for lightweight coverage runs.
if os.getenv("SKIP_TORCH_CHECK", "0") not in ("1", "true", "TRUE", "yes", "YES"):
    # Ensure PyTorch is available; fail fast if not.
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "PyTorch must be installed for this test suite."
        ) from e
