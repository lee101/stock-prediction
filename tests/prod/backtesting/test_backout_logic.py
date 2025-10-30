import sys
import types
from types import SimpleNamespace
from datetime import datetime, timedelta

import pytest

# Create dummy modules so alpaca_cli can be imported without real dependencies
sys.modules.setdefault("alpaca_trade_api", types.ModuleType("alpaca_trade_api"))
sys.modules.setdefault("alpaca_trade_api.rest", types.ModuleType("alpaca_trade_api.rest"))

alpaca_module = sys.modules["alpaca_trade_api.rest"]
alpaca_module.APIError = Exception
sys.modules["alpaca_trade_api"].REST = lambda *a, **k: types.SimpleNamespace()

sys.modules.setdefault("alpaca", types.ModuleType("alpaca"))
sys.modules.setdefault("alpaca.data", types.ModuleType("alpaca.data"))
sys.modules.setdefault("alpaca.data.enums", types.ModuleType("alpaca.data.enums"))
sys.modules.setdefault("alpaca.trading", types.ModuleType("alpaca.trading"))
sys.modules.setdefault("alpaca.trading.client", types.ModuleType("client"))
sys.modules.setdefault("alpaca.trading.enums", types.ModuleType("enums"))
sys.modules.setdefault("alpaca.trading.requests", types.ModuleType("requests"))
alpaca_data = sys.modules["alpaca.data"]
alpaca_data.StockHistoricalDataClient = lambda *a, **k: None
sys.modules["alpaca.data"].StockHistoricalDataClient = lambda *a, **k: None
alpaca_data.StockLatestQuoteRequest = lambda *a, **k: None
alpaca_data.CryptoHistoricalDataClient = lambda *a, **k: None
alpaca_data.CryptoLatestQuoteRequest = lambda *a, **k: None
sys.modules["alpaca.data.enums"].DataFeed = types.SimpleNamespace()
alpaca_trading = sys.modules["alpaca.trading"]
alpaca_trading.OrderType = types.SimpleNamespace(LIMIT='limit', MARKET='market')
alpaca_trading.LimitOrderRequest = lambda **kw: kw
alpaca_trading.GetOrdersRequest = object
alpaca_trading.Order = object
alpaca_trading.client = types.ModuleType("client")
alpaca_trading.enums = types.ModuleType("enums")
alpaca_trading.requests = types.ModuleType("requests")
class DummyTradingClient:
    def __init__(self, *a, **k):
        self.orders = []
    def get_all_positions(self):
        return []
    def get_account(self):
        return types.SimpleNamespace(equity=0, cash=0, multiplier=1)
    def get_clock(self):
        return types.SimpleNamespace(is_open=True)
    def cancel_orders(self):
        self.orders.clear()
    def submit_order(self, order_data):
        self.orders.append(order_data)
        return order_data
alpaca_trading.client.TradingClient = DummyTradingClient
alpaca_trading.enums.OrderSide = types.SimpleNamespace(BUY='buy', SELL='sell')
alpaca_trading.requests.MarketOrderRequest = object
sys.modules["alpaca.trading.client"].TradingClient = DummyTradingClient
sys.modules["alpaca.trading.enums"].OrderSide = types.SimpleNamespace(BUY='buy', SELL='sell')
sys.modules["alpaca.trading.requests"].MarketOrderRequest = object
sys.modules.setdefault("typer", types.ModuleType("typer"))
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
sys.modules["requests"].exceptions = sys.modules["requests.exceptions"]
sys.modules["requests.exceptions"].ConnectionError = Exception
loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = types.SimpleNamespace(info=lambda *a, **k: None)
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
    pytz_mod.exceptions = types.SimpleNamespace(UnknownTimeZoneError=Exception)
    sys.modules["pytz"] = pytz_mod
else:
    sys.modules["pytz"] = pytz_mod
env_real = types.ModuleType("env_real")
env_real.ALP_KEY_ID = "key"
env_real.ALP_SECRET_KEY = "secret"
env_real.ALP_KEY_ID_PROD = "key"
env_real.ALP_SECRET_KEY_PROD = "secret"
env_real.ALP_ENDPOINT = "paper"
sys.modules.setdefault("env_real", env_real)
sys.modules.setdefault("data_curate_daily", types.ModuleType("data_curate_daily"))
data_curate_daily = sys.modules["data_curate_daily"]
data_curate_daily.download_exchange_latest_data = lambda *a, **k: None
data_curate_daily.get_bid = lambda *a, **k: 0
data_curate_daily.get_ask = lambda *a, **k: 0
jsonshelve_mod = types.ModuleType("jsonshelve")
class FlatShelf(dict):
    def __init__(self, *a, **k):
        super().__init__()
    def load(self):
        pass
jsonshelve_mod.FlatShelf = FlatShelf
sys.modules.setdefault("jsonshelve", jsonshelve_mod)
sys.modules.setdefault("src.fixtures", types.ModuleType("fixtures"))
sys.modules["src.fixtures"].crypto_symbols = []
logging_utils_mod = types.ModuleType("logging_utils")

def _stub_logger(*args, **kwargs):
    return types.SimpleNamespace(
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )

logging_utils_mod.setup_logging = _stub_logger
sys.modules.setdefault("src.logging_utils", logging_utils_mod)
sys.modules.setdefault("src.stock_utils", types.ModuleType("stock_utils"))
sys.modules["src.stock_utils"].pairs_equal = lambda a,b: a==b
sys.modules["src.stock_utils"].remap_symbols = lambda s: s
sys.modules.setdefault("src.trading_obj_utils", types.ModuleType("trading_obj_utils"))
sys.modules["src.trading_obj_utils"].filter_to_realistic_positions = lambda x: x

import scripts.alpaca_cli as alpaca_cli


class DummyData:
    def __init__(self, bid, ask):
        self.bid_price = bid
        self.ask_price = ask


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(alpaca_cli, 'sleep', lambda *a, **k: None)


def test_close_position_near_market_short_uses_ask(monkeypatch):
    position = SimpleNamespace(symbol='META', side='short', qty=1)
    dummy_quote = DummyData(99, 100)
    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'latest_data', lambda s: dummy_quote)

    captured = {}

    def fake_submit(order_data):
        captured['price'] = order_data['limit_price']
        return 'ok'

    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'alpaca_api', types.SimpleNamespace(submit_order=fake_submit))

    result = alpaca_cli.alpaca_wrapper.close_position_near_market(position, pct_above_market=0)
    assert result == 'ok'
    assert captured['price'] == '100.0'


def test_close_position_near_market_long_uses_bid(monkeypatch):
    position = SimpleNamespace(symbol='META', side='long', qty=1)
    dummy_quote = DummyData(98, 99)
    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'latest_data', lambda s: dummy_quote)

    captured = {}

    def fake_submit(order_data):
        captured['price'] = order_data['limit_price']
        return 'ok'

    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'alpaca_api', types.SimpleNamespace(submit_order=fake_submit))

    result = alpaca_cli.alpaca_wrapper.close_position_near_market(position, pct_above_market=0)
    assert result == 'ok'
    assert captured['price'] == '98.0'


def test_backout_near_market_switches_to_market(monkeypatch):
    start = datetime.now() - timedelta(minutes=16)
    position = SimpleNamespace(symbol='META', side='short', qty=1)

    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'filter_to_realistic_positions', lambda pos: pos)
    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'get_open_orders', lambda: [])
    monkeypatch.setattr(alpaca_cli, '_minutes_until_market_close', lambda *a, **k: 120.0)

    called = {}

    def fake_market(pos):
        called['called'] = True
        return True

    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'close_position_near_market', lambda *a, **k: pytest.fail('limit order used'))
    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'close_position_violently', fake_market)

    # Sequence: first call returns position, second returns empty list to exit loop
    call_count = {'n': 0}

    def get_positions():
        call_count['n'] += 1
        return [position] if call_count['n'] == 1 else []

    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'get_all_positions', get_positions)

    alpaca_cli.backout_near_market('META', start_time=start, ramp_minutes=10, market_after=15, sleep_interval=0)

    assert called.get('called')


def test_backout_near_market_ramp_progress(monkeypatch):
    start = datetime.now() - timedelta(minutes=14)
    position = SimpleNamespace(symbol='META', side='short', qty=1)

    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'filter_to_realistic_positions', lambda pos: pos)
    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'get_open_orders', lambda: [])
    monkeypatch.setattr(alpaca_cli, '_minutes_until_market_close', lambda *a, **k: 120.0)

    captured = {}

    def fake_close(pos, *, pct_above_market):
        captured['pct'] = pct_above_market
        return True

    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'close_position_near_market', fake_close)

    call_count = {'n': 0}

    def get_positions():
        call_count['n'] += 1
        return [position] if call_count['n'] == 1 else []

    monkeypatch.setattr(alpaca_cli.alpaca_wrapper, 'get_all_positions', get_positions)

    ramp_minutes = 30
    alpaca_cli.backout_near_market(
        'META',
        start_time=start,
        ramp_minutes=ramp_minutes,
        market_after=50,
        market_close_buffer_minutes=0,
        sleep_interval=0,
    )

    minutes_since_start = 14
    pct_offset = -0.003
    pct_final_offset = 0.02
    progress = min(minutes_since_start / ramp_minutes, 1.0)
    expected_pct = pct_offset + (pct_final_offset - pct_offset) * progress

    assert pytest.approx(captured['pct'], rel=1e-6) == pytest.approx(expected_pct, rel=1e-6)
