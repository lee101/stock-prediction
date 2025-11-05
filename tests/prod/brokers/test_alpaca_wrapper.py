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
    open_market_order_violently,
    close_position_violently,
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


def test_market_order_blocked_when_market_closed():
    """Market orders should be blocked when market is closed."""
    # Create a mock clock that says market is closed
    mock_clock = MagicMock()
    mock_clock.is_open = False

    with patch("alpaca_wrapper.get_clock", return_value=mock_clock), \
         patch("alpaca_wrapper.alpaca_api.submit_order") as submit:

        result = open_market_order_violently("AAPL", 10, "buy")

        # Should return None and not call submit_order
        assert result is None
        assert submit.call_count == 0


def test_crypto_market_order_always_blocked():
    """Market orders should NEVER be allowed for crypto (Alpaca executes at bid/ask midpoint, not market price)."""
    # Create a mock clock that says market is open
    mock_clock = MagicMock()
    mock_clock.is_open = True

    with patch("alpaca_wrapper.get_clock", return_value=mock_clock), \
         patch("alpaca_wrapper.alpaca_api.submit_order") as submit:

        # Even with market open, crypto market orders should be blocked
        # because Alpaca will execute them at the bid/ask midpoint instead of market price
        result = open_market_order_violently("BTCUSD", 0.01, "buy")

        # Should return None and not call submit_order
        assert result is None
        assert submit.call_count == 0


def test_market_order_allowed_when_market_open():
    """Market orders should work when market is open."""
    # Create a mock clock that says market is open
    mock_clock = MagicMock()
    mock_clock.is_open = True

    with patch("alpaca_wrapper.get_clock", return_value=mock_clock), \
         patch("alpaca_wrapper.MarketOrderRequest", side_effect=lambda **kw: kw), \
         patch("alpaca_wrapper.alpaca_api.submit_order", return_value="order_ok") as submit:

        result = open_market_order_violently("AAPL", 10, "buy")

        # Should succeed
        assert result == "order_ok"
        assert submit.call_count == 1


def test_market_order_blocked_when_spread_too_high():
    """Market orders should be blocked when spread > 1%, but fallback to limit order at midpoint."""
    # Create a mock position
    mock_position = MagicMock()
    mock_position.symbol = "AAPL"
    mock_position.side = "long"
    mock_position.qty = 10

    # Create a mock clock that says market is open
    mock_clock = MagicMock()
    mock_clock.is_open = True

    # Mock quote with high spread (2%)
    mock_quote = MagicMock()
    mock_quote.ask_price = 102.0
    mock_quote.bid_price = 100.0  # 2% spread

    with patch("alpaca_wrapper.get_clock", return_value=mock_clock), \
         patch("alpaca_wrapper.latest_data", return_value=mock_quote), \
         patch("alpaca_wrapper.LimitOrderRequest", side_effect=lambda **kw: kw), \
         patch("alpaca_wrapper.alpaca_api.submit_order", return_value="limit_order_ok") as submit:

        result = close_position_violently(mock_position)

        # Should fallback to limit order at midpoint (101.0)
        assert result == "limit_order_ok"
        assert submit.call_count == 1
        # Verify it used a limit order, not market order
        order_data = submit.call_args.kwargs["order_data"]
        assert order_data["limit_price"] == "101.0"  # midpoint of 100 and 102


def test_market_order_allowed_when_spread_acceptable():
    """Market orders should work when spread <= 1% and closing position."""
    # Create a mock position
    mock_position = MagicMock()
    mock_position.symbol = "AAPL"
    mock_position.side = "long"
    mock_position.qty = 10

    # Create a mock clock that says market is open
    mock_clock = MagicMock()
    mock_clock.is_open = True

    # Mock quote with acceptable spread (0.5%)
    mock_quote = MagicMock()
    mock_quote.ask_price = 100.5
    mock_quote.bid_price = 100.0  # 0.5% spread

    with patch("alpaca_wrapper.get_clock", return_value=mock_clock), \
         patch("alpaca_wrapper.latest_data", return_value=mock_quote), \
         patch("alpaca_wrapper.MarketOrderRequest", side_effect=lambda **kw: kw), \
         patch("alpaca_wrapper.alpaca_api.submit_order", return_value="order_ok") as submit:

        result = close_position_violently(mock_position)

        # Should succeed
        assert result == "order_ok"
        assert submit.call_count == 1


def test_limit_order_allowed_when_market_closed():
    """Limit orders should work even when market is closed (out-of-hours trading)."""
    # Create a mock clock that says market is closed
    mock_clock = MagicMock()
    mock_clock.is_open = False

    with patch("alpaca_wrapper.get_clock", return_value=mock_clock), \
         patch("alpaca_wrapper.get_orders", return_value=[]), \
         patch("alpaca_wrapper.has_current_open_position", return_value=False), \
         patch("alpaca_wrapper.LimitOrderRequest", side_effect=lambda **kw: kw), \
         patch("alpaca_wrapper.alpaca_api.submit_order", return_value="order_ok") as submit:

        result = open_order_at_price_or_all("AAPL", 10, "buy", 150.0)

        # Should succeed - limit orders work out of hours
        assert result == "order_ok"
        assert submit.call_count == 1


def test_crypto_position_closes_with_limit_order():
    """Crypto positions should always close with limit orders (no market orders)."""
    # Create a mock crypto position
    mock_position = MagicMock()
    mock_position.symbol = "BTCUSD"
    mock_position.side = "long"
    mock_position.qty = 0.5

    # Create a mock clock that says market is open (doesn't matter for crypto)
    mock_clock = MagicMock()
    mock_clock.is_open = True

    # Mock quote with reasonable spread
    mock_quote = MagicMock()
    mock_quote.ask_price = 50100.0
    mock_quote.bid_price = 50000.0  # 0.2% spread (under 1%)

    with patch("alpaca_wrapper.get_clock", return_value=mock_clock), \
         patch("alpaca_wrapper.latest_data", return_value=mock_quote), \
         patch("alpaca_wrapper.LimitOrderRequest", side_effect=lambda **kw: kw), \
         patch("alpaca_wrapper.alpaca_api.submit_order", return_value="crypto_limit_ok") as submit:

        result = close_position_violently(mock_position)

        # Should use limit order at midpoint, NOT market order
        assert result == "crypto_limit_ok"
        assert submit.call_count == 1
        # Verify it used a limit order
        order_data = submit.call_args.kwargs["order_data"]
        assert "limit_price" in order_data
        assert order_data["limit_price"] == "50050.0"  # midpoint


def test_force_open_clock_allows_out_of_hours_trading():
    """When force_open_the_clock is set, we can trade out of hours with limit orders."""
    import alpaca_wrapper

    # Save original value
    original_force = alpaca_wrapper.force_open_the_clock

    try:
        # Set force_open_the_clock
        alpaca_wrapper.force_open_the_clock = True

        # Create a mock clock that says market is closed
        mock_clock = MagicMock()
        mock_clock.is_open = False

        with patch("alpaca_wrapper.get_clock_internal", return_value=mock_clock), \
             patch("alpaca_wrapper.get_orders", return_value=[]), \
             patch("alpaca_wrapper.has_current_open_position", return_value=False), \
             patch("alpaca_wrapper.LimitOrderRequest", side_effect=lambda **kw: kw), \
             patch("alpaca_wrapper.alpaca_api.submit_order", return_value="order_ok") as submit:

            # get_clock should return market as open due to force flag
            clock = alpaca_wrapper.get_clock()
            assert clock.is_open is True

            result = open_order_at_price_or_all("AAPL", 10, "buy", 150.0)

            # Should succeed
            assert result == "order_ok"
            assert submit.call_count == 1
    finally:
        # Restore original value
        alpaca_wrapper.force_open_the_clock = original_force
