from datetime import datetime, timedelta
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def cli(monkeypatch) -> ModuleType:
    rest_stub = lambda *args, **kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "alpaca_trade_api", SimpleNamespace(REST=rest_stub))
    data_module = ModuleType("alpaca.data")
    data_module.StockHistoricalDataClient = lambda *args, **kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "alpaca.data", data_module)
    module = importlib.import_module("scripts.alpaca_cli")
    yield module
    sys.modules.pop("scripts.alpaca_cli", None)


class StubWrapper:
    def __init__(self):
        self._position_calls = 0
        self.limit_calls = 0
        self.market_calls = 0

    def get_all_positions(self):
        self._position_calls += 1
        if self._position_calls == 1:
            return [SimpleNamespace(symbol="AAPL", side="long", qty="1")]
        return []

    def get_open_orders(self):
        return []

    def cancel_order(self, order):
        return None

    def close_position_near_market(self, position, *, pct_above_market):
        self.limit_calls += 1
        return True

    def close_position_violently(self, position):
        self.market_calls += 1
        return True


def _setup_common(cli_module, monkeypatch, spread_value):
    wrapper = StubWrapper()
    monkeypatch.setattr(cli_module, "alpaca_wrapper", wrapper)
    monkeypatch.setattr(cli_module, "filter_to_realistic_positions", lambda positions: positions)
    monkeypatch.setattr(cli_module, "pairs_equal", lambda left, right: left == right)
    monkeypatch.setattr(cli_module, "_current_spread_pct", lambda symbol: spread_value)
    monkeypatch.setattr(cli_module, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "BACKOUT_MARKET_MAX_SPREAD_PCT", 0.01)
    return wrapper


def test_backout_near_market_skips_market_when_spread_high(cli, monkeypatch):
    wrapper = _setup_common(cli, monkeypatch, spread_value=0.02)  # 2%
    start_time = datetime.now() - timedelta(minutes=60)

    cli.backout_near_market(
        "AAPL",
        start_time=start_time,
        ramp_minutes=1,
        market_after=1,
        sleep_interval=0,
    )

    assert wrapper.market_calls == 0
    assert wrapper.limit_calls >= 1


def test_backout_near_market_uses_market_when_spread_ok(cli, monkeypatch):
    wrapper = _setup_common(cli, monkeypatch, spread_value=0.005)  # 0.5%
    start_time = datetime.now() - timedelta(minutes=60)

    cli.backout_near_market(
        "AAPL",
        start_time=start_time,
        ramp_minutes=1,
        market_after=1,
        sleep_interval=0,
    )

    assert wrapper.market_calls == 1
    assert wrapper.limit_calls == 0
