from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import tests.conftest as conftest


def test_alpaca_stub_repair_restores_request_constructors(monkeypatch) -> None:
    alpaca = types.ModuleType("alpaca")
    trading = types.ModuleType("alpaca.trading")
    requests = types.ModuleType("alpaca.trading.requests")
    requests.MarketOrderRequest = MagicMock()
    trading.MarketOrderRequest = object()
    trading.LimitOrderRequest = object()
    trading.GetOrdersRequest = object()

    monkeypatch.setitem(sys.modules, "alpaca", alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", requests)

    conftest._ensure_alpaca_test_stubs()

    assert sys.modules["alpaca"].trading is trading
    assert trading.requests is requests
    assert requests.MarketOrderRequest is not None
    assert requests.LimitOrderRequest is not None
    assert requests.GetOrdersRequest is not None
    assert trading.MarketOrderRequest is requests.MarketOrderRequest
    assert trading.LimitOrderRequest is requests.LimitOrderRequest
    assert trading.GetOrdersRequest is requests.GetOrdersRequest


def test_alpaca_stub_repair_restores_trading_client_module(monkeypatch) -> None:
    alpaca = types.ModuleType("alpaca")
    trading = types.ModuleType("alpaca.trading")
    client = types.ModuleType("alpaca.trading.client")
    trading.TradingClient = object()

    monkeypatch.setitem(sys.modules, "alpaca", alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", client)

    conftest._ensure_alpaca_test_stubs()

    assert sys.modules["alpaca"].trading is trading
    assert trading.client is client
    assert client.TradingClient is not None
    assert trading.TradingClient is client.TradingClient


def test_alpaca_stub_repair_restores_enum_value_shape(monkeypatch) -> None:
    trading = types.ModuleType("alpaca.trading")
    enums = types.ModuleType("alpaca.trading.enums")
    enums.OrderSide = types.SimpleNamespace(BUY="buy", SELL="sell")
    enums.TimeInForce = types.SimpleNamespace(DAY="day", GTC="gtc")

    monkeypatch.setitem(sys.modules, "alpaca.trading", trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums)

    conftest._ensure_alpaca_test_stubs()

    assert enums.OrderSide.BUY.value == "buy"
    assert enums.OrderSide.BUY == "buy"
    assert enums.OrderSide.SELL.value == "sell"
    assert enums.OrderSide.SELL == "sell"
    assert enums.TimeInForce.DAY.value == "day"
    assert enums.TimeInForce.DAY == "day"
    assert enums.TimeInForce.GTC.value == "gtc"
    assert enums.TimeInForce.GTC == "gtc"
    assert enums.TimeInForce.IOC.value == "ioc"
    assert enums.TimeInForce.IOC == "ioc"
    assert enums.QueryOrderStatus.OPEN.value == "open"
    assert enums.QueryOrderStatus.OPEN == "open"
    assert enums.QueryOrderStatus.CLOSED.value == "closed"
    assert enums.QueryOrderStatus.CLOSED == "closed"


def test_alpaca_stub_repair_replaces_dynamic_magicmock_enums(monkeypatch) -> None:
    trading = types.ModuleType("alpaca.trading")
    enums = types.ModuleType("alpaca.trading.enums")
    enums.OrderSide = MagicMock()
    enums.TimeInForce = MagicMock()

    monkeypatch.setitem(sys.modules, "alpaca.trading", trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums)

    conftest._ensure_alpaca_test_stubs()

    assert enums.OrderSide.BUY.value == "buy"
    assert enums.OrderSide.SELL.value == "sell"
    assert enums.TimeInForce.DAY.value == "day"
    assert enums.TimeInForce.GTC.value == "gtc"
    assert enums.TimeInForce.IOC.value == "ioc"
    assert enums.QueryOrderStatus.OPEN.value == "open"
    assert enums.QueryOrderStatus.CLOSED.value == "closed"
