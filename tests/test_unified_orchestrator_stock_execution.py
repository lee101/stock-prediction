from __future__ import annotations

import sys
import types
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import alpaca.trading.client as trading_client
import alpaca.trading.enums as trading_enums
import alpaca.trading.requests as trading_requests
import pandas as pd

from llm_hourly_trader.gemini_wrapper import TradePlan
from unified_orchestrator import orchestrator
from unified_orchestrator.state import UnifiedPortfolioSnapshot


def _patch_stock_execution_dependencies(monkeypatch, tmp_path) -> None:
    class _OrderSide:
        BUY = "buy"
        SELL = "sell"

    class _TimeInForce:
        DAY = "day"

    fake_enums = types.ModuleType("alpaca.trading.enums")
    fake_enums.OrderSide = _OrderSide
    fake_enums.TimeInForce = _TimeInForce
    fake_requests = types.ModuleType("alpaca.trading.requests")
    fake_requests.LimitOrderRequest = lambda **kwargs: kwargs

    monkeypatch.setattr(orchestrator, "STOCK_PEAKS_FILE", tmp_path / "stock_peaks.json")
    monkeypatch.setattr(orchestrator, "check_recent_splits", lambda symbols, lookback_days=7: {})
    monkeypatch.setattr(trading_client, "TradingClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(trading_enums, "OrderSide", _OrderSide, raising=False)
    monkeypatch.setattr(trading_enums, "TimeInForce", _TimeInForce, raising=False)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", fake_enums)
    monkeypatch.setattr(trading_requests, "LimitOrderRequest", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", fake_requests)


def test_execute_stock_signals_respects_150pct_gross_exposure(monkeypatch, tmp_path) -> None:
    _patch_stock_execution_dependencies(monkeypatch, tmp_path)

    snapshot = UnifiedPortfolioSnapshot(
        alpaca_cash=10_000.0,
        alpaca_buying_power=40_000.0,
        regime="STOCK_HOURS",
        minutes_to_close=30,
    )
    plan = TradePlan(
        direction="long",
        buy_price=100.0,
        sell_price=104.0,
        confidence=0.8,
        reasoning="best setup",
        allocation_pct=150.0,
    )

    orders = orchestrator.execute_stock_signals({"CRWD": plan}, snapshot, dry_run=True)

    assert orders == [
        {
            "symbol": "CRWD",
            "action": "buy",
            "price": 100.0,
            "qty": 150.0,
            "dry_run": True,
        }
    ]


def test_execute_stock_signals_caps_single_name_stock_exposure_at_2x(monkeypatch, tmp_path) -> None:
    _patch_stock_execution_dependencies(monkeypatch, tmp_path)

    snapshot = UnifiedPortfolioSnapshot(
        alpaca_cash=10_000.0,
        alpaca_buying_power=40_000.0,
        regime="STOCK_HOURS",
        minutes_to_close=30,
    )
    plan = TradePlan(
        direction="long",
        buy_price=100.0,
        sell_price=104.0,
        confidence=0.8,
        reasoning="pack the account",
        allocation_pct=250.0,
    )

    orders = orchestrator.execute_stock_signals({"COIN": plan}, snapshot, dry_run=True)

    assert orders == [
        {
            "symbol": "COIN",
            "action": "buy",
            "price": 100.0,
            "qty": 200.0,
            "dry_run": True,
        }
    ]


def test_fetch_stock_history_frames_uses_wide_enough_calendar_lookback(monkeypatch) -> None:
    requests_seen: list[dict] = []
    data_enums = sys.modules.setdefault("alpaca.data.enums", types.ModuleType("alpaca.data.enums"))
    data_requests = sys.modules.setdefault("alpaca.data.requests", types.ModuleType("alpaca.data.requests"))
    data_timeframe = sys.modules.setdefault("alpaca.data.timeframe", types.ModuleType("alpaca.data.timeframe"))

    monkeypatch.setattr(data_enums, "DataFeed", SimpleNamespace(IEX="iex"), raising=False)
    monkeypatch.setattr(data_timeframe, "TimeFrame", SimpleNamespace(Hour="hour"), raising=False)
    monkeypatch.setattr(data_requests, "StockBarsRequest", lambda **kwargs: requests_seen.append(kwargs) or kwargs, raising=False)

    class _Client:
        def get_stock_bars(self, request):
            frame = pd.DataFrame(
                {
                    "timestamp": [request["end"] - timedelta(hours=1)],
                    "symbol": [request["symbol_or_symbols"]],
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1_000.0],
                }
            )
            return SimpleNamespace(df=frame)

    now = datetime(2026, 4, 12, 11, 40, tzinfo=UTC)
    frames = orchestrator._fetch_stock_history_frames(_Client(), ["CRWD"], now)

    assert "CRWD" in frames
    assert len(requests_seen) == 1
    request = requests_seen[0]
    assert request["limit"] == 72
    assert request["timeframe"] == "hour"
    assert request["feed"] == "iex"
    assert request["end"] == now
    assert now - request["start"] >= timedelta(days=20)
