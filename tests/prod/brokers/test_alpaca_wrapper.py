import sys
import types
import pytest
from unittest.mock import patch, MagicMock

# Create dummy modules so alpaca_wrapper can be imported without the real
# dependencies installed in the test environment.
sys.modules.setdefault("cachetools", types.ModuleType("cachetools"))
cachetools_mod = sys.modules["cachetools"]
def cached(**kwargs):
    def decorator(func):
        return func
    return decorator
class TTLCache(dict):
    def __init__(self, maxsize, ttl):
        super().__init__()
cachetools_mod.cached = cached
cachetools_mod.TTLCache = TTLCache
sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("requests.exceptions", types.ModuleType("requests.exceptions"))
loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = MagicMock()
sys.modules.setdefault("loguru", loguru_mod)
retry_mod = types.ModuleType("retry")
def _retry(*a, **kw):
    def decorator(func):
        return func
    return decorator
retry_mod.retry = _retry
sys.modules.setdefault("retry", retry_mod)
try:
    import pytz as pytz_mod  # type: ignore
except ModuleNotFoundError:
    pytz_mod = types.ModuleType("pytz")

    def timezone(name):
        return name

    pytz_mod.timezone = timezone
    pytz_mod.UTC = object()

    class _Exc(Exception):
        pass

    class _Ex:
        UnknownTimeZoneError = _Exc

    pytz_mod.exceptions = _Ex()
    sys.modules["pytz"] = pytz_mod
else:
    sys.modules["pytz"] = pytz_mod

alpaca = types.ModuleType("alpaca")
alpaca_data = types.ModuleType("alpaca.data")
alpaca_trading = types.ModuleType("alpaca.trading")
alpaca_trading.client = types.ModuleType("client")
alpaca_trading.enums = types.ModuleType("enums")
alpaca_trading.requests = types.ModuleType("requests")

alpaca_data.StockLatestQuoteRequest = MagicMock()
alpaca_data.StockHistoricalDataClient = MagicMock()
alpaca_data.CryptoHistoricalDataClient = MagicMock()
alpaca_data.CryptoLatestQuoteRequest = MagicMock()
alpaca_data.StockBarsRequest = MagicMock()
alpaca_data.CryptoBarsRequest = MagicMock()
alpaca_data.TimeFrame = MagicMock()
alpaca_data.TimeFrameUnit = MagicMock()

alpaca_data_enums = types.ModuleType("alpaca.data.enums")
alpaca_data_enums.DataFeed = MagicMock()

alpaca_trading.OrderType = MagicMock()
alpaca_trading.LimitOrderRequest = MagicMock()
alpaca_trading.GetOrdersRequest = MagicMock()
alpaca_trading.Order = MagicMock()
alpaca_trading.client.TradingClient = MagicMock()
alpaca_trading.enums.OrderSide = MagicMock()
alpaca_trading.requests.MarketOrderRequest = MagicMock()

sys.modules["alpaca"] = alpaca
sys.modules["alpaca.data"] = alpaca_data
sys.modules["alpaca.data.enums"] = alpaca_data_enums
sys.modules["alpaca.trading"] = alpaca_trading
sys.modules["alpaca.trading.client"] = alpaca_trading.client
sys.modules["alpaca.trading.enums"] = alpaca_trading.enums
sys.modules["alpaca.trading.requests"] = alpaca_trading.requests

alpaca_trade_api = types.ModuleType("alpaca_trade_api.rest")
alpaca_trade_api.APIError = Exception
sys.modules["alpaca_trade_api"] = types.ModuleType("alpaca_trade_api")
sys.modules["alpaca_trade_api.rest"] = alpaca_trade_api

env_real = types.ModuleType("env_real")
env_real.ALP_KEY_ID = "key"
env_real.ALP_SECRET_KEY = "secret"
env_real.ALP_KEY_ID_PROD = "key"
env_real.ALP_SECRET_KEY_PROD = "secret"
env_real.ALP_ENDPOINT = "paper"
sys.modules["env_real"] = env_real

from alpaca_wrapper import (
    latest_data,
    has_current_open_position,
    execute_portfolio_orders,
    open_order_at_price_or_all,
)


@pytest.mark.skip(reason="Requires network access")
def test_get_latest_data():
    data = latest_data('BTCUSD')
    print(data)
    data = latest_data('COUR')
    print(data)


@pytest.mark.skip(reason="Requires network access")
def test_has_current_open_position():
    has_position = has_current_open_position('BTCUSD', 'buy')  # real
    assert has_position is True
    has_position = has_current_open_position('BTCUSD', 'sell')  # real
    assert has_position is False
    has_position = has_current_open_position('LTCUSD', 'buy')  # real
    assert has_position is False


def test_execute_portfolio_orders_handles_errors():
    orders = [
        {"symbol": "AAA", "qty": 1, "side": "buy", "price": 10},
        {"symbol": "BBB", "qty": 1, "side": "buy", "price": 20},
    ]

    with patch("alpaca_wrapper.open_order_at_price_or_all") as mock_open:
        mock_open.side_effect = [Exception("rejected"), "ok"]
        results = execute_portfolio_orders(orders)

    assert results["AAA"] is None
    assert results["BBB"] == "ok"
    assert mock_open.call_count == 2


def test_open_order_at_price_or_all_adjusts_on_insufficient_balance():
    with patch("alpaca_wrapper.get_orders", return_value=[]), \
         patch("alpaca_wrapper.has_current_open_position", return_value=False), \
         patch("alpaca_wrapper.LimitOrderRequest", side_effect=lambda **kw: kw) as req, \
         patch("alpaca_wrapper.alpaca_api.submit_order") as submit:

        submit.side_effect = [
            Exception('{"available": 50, "message": "insufficient balance"}'),
            "ok",
        ]

        result = open_order_at_price_or_all("AAA", 10, "buy", 10)

    assert result == "ok"
    assert submit.call_count == 2
    first_qty = submit.call_args_list[0].kwargs["order_data"]["qty"]
    second_qty = submit.call_args_list[1].kwargs["order_data"]["qty"]
    assert first_qty == 10
    assert second_qty == 4
