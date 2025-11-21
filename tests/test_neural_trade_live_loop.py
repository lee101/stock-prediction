import types
from datetime import datetime, timedelta, timezone

import pytest

from neural_trade_stock_live import NeuralTradingLoop, NeuralPlan
from neuraldailytraining.runtime import TradingPlan


class DummyRuntime:
    def __init__(self, plans):
        self._plans = plans
        self.risk_threshold = 1.0

    def generate_plans(self, symbols):
        return self._plans


def _make_plan(symbol: str, buy: float, sell: float, amt: float) -> TradingPlan:
    return TradingPlan(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc).isoformat(),
        buy_price=buy,
        sell_price=sell,
        trade_amount=amt,
        reference_close=buy,
    )


def test_min_trade_amount_filters_small_plans():
    plans = [
        _make_plan("SMALL", 10.0, 10.5, 0.04),
        _make_plan("KEEP", 20.0, 20.3, 0.06),
    ]
    loop = NeuralTradingLoop(
        runtime=DummyRuntime(plans),
        symbols=["SMALL", "KEEP"],
        min_trade_amount=0.05,
        max_plans=5,
        skip_equity_weekends=False,
    )

    selected = loop._generate_plans()

    assert [p.symbol for p in selected] == ["KEEP"]


def test_max_plans_uses_edge_priority():
    plans = [
        _make_plan("MID", 100.0, 101.0, 1.0),   # 1% edge
        _make_plan("LOW", 50.0, 50.1, 1.0),     # 0.2% edge
        _make_plan("HIGH", 10.0, 10.5, 1.0),    # 5% edge
    ]
    loop = NeuralTradingLoop(
        runtime=DummyRuntime(plans),
        symbols=["MID", "LOW", "HIGH"],
        max_plans=2,
        min_trade_amount=0.0,
        skip_equity_weekends=False,
    )

    selected = loop._generate_plans()

    # Expect HIGH then MID (largest edge then next largest)
    assert [p.symbol for p in selected] == ["HIGH", "MID"]
    assert [p.priority for p in selected] == [1, 2]


def test_weekend_skip_equity(monkeypatch):
    plans = [_make_plan("AAPL", 100.0, 100.5, 0.5)]
    loop = NeuralTradingLoop(
        runtime=DummyRuntime(plans),
        symbols=["AAPL"],
        skip_equity_weekends=True,
        min_trade_amount=0.0,
    )

    called = {}

    def fake_spawn(*args, **kwargs):  # pragma: no cover - monkeypatched hook
        called["hit"] = True

    # Force weekend (Saturday)
    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2025, 11, 22, 12, 0, 0, tzinfo=timezone.utc)  # Saturday
            return base if tz is None else base.astimezone(tz)

    monkeypatch.setattr("neural_trade_stock_live.spawn_open_position_at_maxdiff_takeprofit", fake_spawn)
    monkeypatch.setattr("neural_trade_stock_live.datetime", FakeDateTime)

    loop._dispatch_plan(loop._generate_plans()[0], account_equity=1000.0)

    assert "hit" not in called, "Equity orders should be skipped on weekends"


def test_weekend_allows_crypto(monkeypatch):
    plans = [_make_plan("BTC-USD", 29000.0, 29100.0, 0.5)]
    loop = NeuralTradingLoop(
        runtime=DummyRuntime(plans),
        symbols=["BTC-USD"],
        skip_equity_weekends=True,
        min_trade_amount=0.0,
    )

    calls = []

    def fake_spawn(symbol, side, limit_price, target_qty, tolerance_pct=None, entry_strategy=None, force_immediate=None, priority_rank=None, crypto_rank=None, **kwargs):  # pragma: no cover - monkeypatched
        calls.append((symbol, side, limit_price, target_qty))

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2025, 11, 22, 12, 0, 0, tzinfo=timezone.utc)  # Saturday
            return base if tz is None else base.astimezone(tz)

    monkeypatch.setattr("neural_trade_stock_live.spawn_open_position_at_maxdiff_takeprofit", fake_spawn)
    monkeypatch.setattr("neural_trade_stock_live.datetime", FakeDateTime)

    loop._dispatch_plan(loop._generate_plans()[0], account_equity=1000.0)

    assert calls, "Crypto should still dispatch on weekends"
    assert calls[0][0] == "BTC-USD"

