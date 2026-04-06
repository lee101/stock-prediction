from __future__ import annotations

import argparse
import builtins
import gc
import json
import os
import struct
import sys
import tempfile
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trade_mixed_daily import _resolve_symbols


@pytest.fixture(autouse=True)
def _reset_pufferlib_market_modules(reset_package_submodules):
    reset_package_submodules("pufferlib_market", "binding", "train")
    yield
    reset_package_submodules("pufferlib_market", "binding", "train")


def _write_mktd(
    path: Path,
    symbols: list[str],
    *,
    features_per_sym: int = 16,
    price_features: int = 5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_symbols = len(symbols)
    num_timesteps = 2

    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD",
        2,
        num_symbols,
        num_timesteps,
        features_per_sym,
        price_features,
        b"\x00" * 40,
    )
    symbol_table = b"".join(sym.encode("ascii").ljust(16, b"\x00") for sym in symbols)
    features = np.zeros((num_timesteps, num_symbols, features_per_sym), dtype=np.float32)
    prices = np.zeros((num_timesteps, num_symbols, price_features), dtype=np.float32)
    mask = np.ones((num_timesteps, num_symbols), dtype=np.uint8)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(symbol_table)
        handle.write(features.tobytes())
        handle.write(prices.tobytes())
        handle.write(mask.tobytes())


def _write_checkpoint(
    path: Path,
    *,
    symbol_count: int = 1,
    features_per_sym: int = 16,
    hidden_size: int = 1024,
    arch: str = "mlp",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_size = symbol_count * features_per_sym + 5 + symbol_count
    num_actions = 1 + 2 * symbol_count
    actor_hidden = max(1, hidden_size // 2)
    if arch == "mlp":
        model = {
            "encoder.0.weight": torch.zeros((hidden_size, obs_size), dtype=torch.float32),
            "actor.2.weight": torch.zeros((num_actions, actor_hidden), dtype=torch.float32),
        }
    elif arch == "resmlp":
        model = {
            "input_proj.weight": torch.zeros((hidden_size, obs_size), dtype=torch.float32),
            "actor.2.weight": torch.zeros((num_actions, actor_hidden), dtype=torch.float32),
        }
    else:  # pragma: no cover - tests only use known architectures
        raise ValueError(f"unsupported checkpoint arch for test helper: {arch}")
    torch.save({"model": model}, path)


def test_resolve_symbols_prefers_matching_local_mktd(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "pufferlib_market" / "checkpoints" / "mixed23_fresh_replay" / "ent_anneal" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {"encoder.0.weight": torch.zeros((8, 39))}}, checkpoint)

    data_dir = tmp_path / "pufferlib_market" / "data"
    _write_mktd(data_dir / "legacy23_val.bin", ["OLDA", "OLDB"])
    _write_mktd(data_dir / "mixed23_fresh_val.bin", ["AAPL", "BTCUSD"])

    monkeypatch.setattr("trade_mixed_daily.REPO", tmp_path)

    args = argparse.Namespace(checkpoint=str(checkpoint), symbols=None)

    assert _resolve_symbols(args) == ["AAPL", "BTCUSD"]


def test_execute_signal_uses_limit_order_in_paper(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    submitted: list[object] = []

    class _TradingClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_account(self):
            return SimpleNamespace(portfolio_value=10_000.0, buying_power=10_000.0)

        def submit_order(self, order):
            submitted.append(order)
            return SimpleNamespace(
                id="paper-limit-1",
                status="accepted",
                side="buy",
                qty=getattr(order, "qty", None),
                symbol=getattr(order, "symbol", None),
                type="limit",
            )

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_aw = types.ModuleType("alpaca_wrapper")
    fake_aw.ALP_KEY_ID = "paper-key"
    fake_aw.ALP_SECRET_KEY = "paper-secret"
    fake_aw.TradingClient = _TradingClient
    fake_aw.OrderType = SimpleNamespace(LIMIT="limit")
    fake_aw.OrderSide = SimpleNamespace(
        BUY=SimpleNamespace(value="buy"),
        SELL=SimpleNamespace(value="sell"),
    )
    fake_aw.LimitOrderRequest = _LimitOrderRequest
    fake_aw.latest_data = lambda symbol: SimpleNamespace(ask_price=101.0, bid_price=100.0)
    fake_aw._midpoint_limit_price = lambda symbol, side, price: 100.5
    fake_aw._get_time_in_force_for_qty = lambda qty, symbol: "day"
    fake_aw.data_client = SimpleNamespace(
        get_stock_bars=lambda _request: {"AAPL": [SimpleNamespace(close=100.0)]}
    )

    fake_requests = types.ModuleType("alpaca.data.requests")
    fake_requests.StockBarsRequest = lambda **kwargs: kwargs
    fake_timeframe = types.ModuleType("alpaca.data.timeframe")
    fake_timeframe.TimeFrame = SimpleNamespace(Hour="hour")

    monkeypatch.setitem(sys.modules, "alpaca_wrapper", fake_aw)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", fake_timeframe)
    monkeypatch.chdir(tmp_path)

    changed = prod.execute_signal(
        {"symbol": "AAPL", "direction": "LONG", "confidence": 0.9},
        allocation_pct=10.0,
    )

    assert changed is True
    assert len(submitted) == 1
    assert isinstance(submitted[0], _LimitOrderRequest)
    assert submitted[0].symbol == "AAPL"
    assert submitted[0].limit_price == 100.5
    assert submitted[0].time_in_force == "day"


def test_load_mixed_daily_broker_api_builds_adapter_from_alpaca_modules(monkeypatch) -> None:
    import trade_mixed_daily_prod as prod

    class _TradingClient:
        def __init__(self, *_args, **_kwargs):
            pass

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_aw = types.ModuleType("alpaca_wrapper")
    fake_aw.ALP_KEY_ID = "paper-key"
    fake_aw.ALP_SECRET_KEY = "paper-secret"
    fake_aw.TradingClient = _TradingClient
    fake_aw.OrderType = SimpleNamespace(LIMIT="limit")
    fake_aw.OrderSide = SimpleNamespace(
        BUY=SimpleNamespace(value="buy"),
        SELL=SimpleNamespace(value="sell"),
    )
    fake_aw.LimitOrderRequest = _LimitOrderRequest
    fake_aw.latest_data = lambda symbol: symbol
    fake_aw._midpoint_limit_price = lambda symbol, side, price: price
    fake_aw._get_time_in_force_for_qty = lambda qty, symbol: "day"
    fake_aw.data_client = SimpleNamespace(name="stock-client")
    fake_aw.crypto_client = SimpleNamespace(name="crypto-client")

    fake_requests = types.ModuleType("alpaca.data.requests")
    fake_requests.StockBarsRequest = lambda **kwargs: ("stock", kwargs)
    fake_requests.CryptoBarsRequest = lambda **kwargs: ("crypto", kwargs)
    fake_timeframe = types.ModuleType("alpaca.data.timeframe")
    fake_timeframe.TimeFrame = SimpleNamespace(Hour="hour")

    monkeypatch.setitem(sys.modules, "alpaca_wrapper", fake_aw)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", fake_timeframe)

    broker = prod._load_mixed_daily_broker_api()

    assert isinstance(broker.trading_client, _TradingClient)
    assert broker.order_type_limit == "limit"
    assert broker.order_side_buy.value == "buy"
    assert broker.order_side_sell.value == "sell"
    assert broker.limit_order_request is _LimitOrderRequest
    assert broker.latest_data("AAPL") == "AAPL"
    assert broker.midpoint_limit_price("AAPL", "buy", 123.0) == 123.0
    assert broker.time_in_force_for_qty(1.0, "AAPL") == "day"
    assert broker.data_client.name == "stock-client"
    assert broker.crypto_client.name == "crypto-client"
    assert broker.stock_bars_request(symbol_or_symbols="AAPL", timeframe="hour", limit=1)[0] == "stock"
    assert broker.crypto_bars_request(symbol_or_symbols="BTC/USD", timeframe="hour", limit=1)[0] == "crypto"
    assert broker.timeframe_hour == "hour"


def test_execute_signal_rejects_when_crypto_market_data_surface_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    submitted: list[object] = []

    class _TradingClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_account(self):
            return SimpleNamespace(portfolio_value=10_000.0, buying_power=10_000.0)

        def submit_order(self, order):
            submitted.append(order)
            return SimpleNamespace(id="unexpected")

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_aw = types.ModuleType("alpaca_wrapper")
    fake_aw.ALP_KEY_ID = "paper-key"
    fake_aw.ALP_SECRET_KEY = "paper-secret"
    fake_aw.TradingClient = _TradingClient
    fake_aw.OrderType = SimpleNamespace(LIMIT="limit")
    fake_aw.OrderSide = SimpleNamespace(
        BUY=SimpleNamespace(value="buy"),
        SELL=SimpleNamespace(value="sell"),
    )
    fake_aw.LimitOrderRequest = _LimitOrderRequest
    fake_aw.latest_data = lambda symbol: SimpleNamespace(ask_price=101.0, bid_price=100.0)
    fake_aw._midpoint_limit_price = lambda symbol, side, price: 100.5
    fake_aw._get_time_in_force_for_qty = lambda qty, symbol: "day"
    fake_aw.data_client = SimpleNamespace(
        get_stock_bars=lambda _request: {"AAPL": [SimpleNamespace(close=100.0)]}
    )
    fake_aw.crypto_client = SimpleNamespace(
        get_crypto_bars=lambda _request: {"BTC/USD": [SimpleNamespace(close=50_000.0)]}
    )

    fake_requests = types.ModuleType("alpaca.data.requests")
    fake_requests.StockBarsRequest = lambda **kwargs: kwargs
    fake_timeframe = types.ModuleType("alpaca.data.timeframe")
    fake_timeframe.TimeFrame = SimpleNamespace(Hour="hour")

    monkeypatch.setitem(sys.modules, "alpaca_wrapper", fake_aw)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", fake_timeframe)
    monkeypatch.chdir(tmp_path)

    result = prod.execute_signal_result(
        {"symbol": "BTCUSD", "direction": "LONG", "confidence": 0.9},
        allocation_pct=10.0,
    )

    assert result.submitted is False
    assert result.status == "rejected"
    assert result.reason == "crypto_market_data_unavailable"
    assert submitted == []


def test_execute_signal_rejects_when_broker_api_is_unavailable(monkeypatch) -> None:
    import trade_mixed_daily_prod as prod

    monkeypatch.setattr(prod, "_load_mixed_daily_broker_api", lambda: (_ for _ in ()).throw(RuntimeError("auth failed")))

    result = prod.execute_signal_result(
        {"symbol": "AAPL", "direction": "LONG", "confidence": 0.9},
        allocation_pct=10.0,
    )

    assert result.submitted is False
    assert result.status == "rejected"
    assert result.reason == "broker_api_unavailable"
    assert result.alpaca_symbol == "AAPL"


def test_execute_signal_rejects_when_account_fetch_fails(monkeypatch) -> None:
    import trade_mixed_daily_prod as prod

    broker = SimpleNamespace(
        trading_client=SimpleNamespace(
            get_account=lambda: (_ for _ in ()).throw(ConnectionError("broker timeout"))
        ),
        order_type_limit="limit",
        order_side_buy=SimpleNamespace(value="buy"),
        order_side_sell=SimpleNamespace(value="sell"),
        limit_order_request=lambda **kwargs: kwargs,
        latest_data=lambda symbol: SimpleNamespace(ask_price=101.0, bid_price=100.0),
        midpoint_limit_price=lambda symbol, side, price: price,
        time_in_force_for_qty=lambda qty, symbol: "day",
        data_client=SimpleNamespace(get_stock_bars=lambda _request: {"AAPL": [SimpleNamespace(close=100.0)]}),
        crypto_client=None,
        stock_bars_request=lambda **kwargs: kwargs,
        crypto_bars_request=None,
        timeframe_hour="hour",
    )
    monkeypatch.setattr(prod, "_load_mixed_daily_broker_api", lambda: broker)

    result = prod.execute_signal_result(
        {"symbol": "AAPL", "direction": "LONG", "confidence": 0.9},
        allocation_pct=10.0,
    )

    assert result.submitted is False
    assert result.status == "rejected"
    assert result.reason == "account_fetch_failed"


def test_build_signal_execution_candidates_decodes_primary_and_ranked_alternatives() -> None:
    import trade_mixed_daily_prod as prod

    candidates = prod._build_signal_execution_candidates(
        {
            "direction": "SHORT",
            "symbol": "BTCUSD",
            "confidence": 0.8,
            "value": 1.25,
            "action": 4,
            "timestamp": "2026-04-03T00:00:00+00:00",
            "all_probs": [0.01, 0.6, 0.05, 0.1, 0.8],
            "sym_names": ["AAPL", "BTCUSD"],
        },
        max_alternatives=3,
    )

    assert [(candidate.attempt_kind, candidate.attempt_index) for candidate in candidates] == [
        ("primary", 0),
        ("alternative", 1),
        ("alternative", 2),
        ("alternative", 3),
    ]
    assert [(candidate.signal["direction"], candidate.signal["symbol"]) for candidate in candidates] == [
        ("SHORT", "BTCUSD"),
        ("LONG", "AAPL"),
        ("SHORT", "AAPL"),
        ("LONG", "BTCUSD"),
    ]
    assert [candidate.signal["confidence"] for candidate in candidates] == pytest.approx([0.8, 0.6, 0.1, 0.05])


def test_execute_inference_signal_uses_alternative_candidates_after_primary_skip(monkeypatch, capsys) -> None:
    import trade_mixed_daily_prod as prod

    signal = {
        "direction": "SHORT",
        "symbol": "BTCUSD",
        "confidence": 0.8,
        "value": 1.25,
        "action": 4,
        "timestamp": "2026-04-03T00:00:00+00:00",
        "all_probs": [0.01, 0.6, 0.05, 0.1, 0.8],
        "sym_names": ["AAPL", "BTCUSD"],
    }
    attempts: list[tuple[str, str, int, str, str, float]] = []
    results = iter(
        [
            prod.TradeExecutionResult(
                submitted=False,
                status="skipped",
                reason="crypto_short_without_position",
                symbol="BTCUSD",
                alpaca_symbol="BTC/USD",
                direction="SHORT",
                confidence=0.8,
                is_crypto=True,
            ),
            prod.TradeExecutionResult(
                submitted=True,
                status="submitted",
                reason="order_submitted",
                symbol="AAPL",
                alpaca_symbol="AAPL",
                direction="LONG",
                confidence=0.6,
                is_crypto=False,
                order_id="ord-2",
                order_status="accepted",
            ),
        ]
    )

    def fake_execute_signal_attempt(*, config, mode, signal, attempt_kind, attempt_index):
        del config
        attempts.append(
            (mode, attempt_kind, attempt_index, signal["direction"], signal["symbol"], signal["confidence"])
        )
        return next(results)

    monkeypatch.setattr(prod, "_execute_signal_attempt", fake_execute_signal_attempt)

    result = prod._execute_inference_signal(config=object(), mode="daemon", signal=signal)

    captured = capsys.readouterr().out
    assert result is not None
    assert result.submitted is True
    assert result.symbol == "AAPL"
    assert attempts == [
        ("daemon", "primary", 0, "SHORT", "BTCUSD", 0.8),
        ("daemon", "alternative", 1, "LONG", "AAPL", 0.6),
    ]
    assert "EXECUTING: SHORT BTCUSD" in captured
    assert "Primary signal skipped, trying alternatives..." in captured
    assert "Trying: LONG AAPL (conf=0.600)" in captured


def test_execute_signal_returns_success_when_trade_log_write_fails(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    submitted: list[object] = []

    class _TradingClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_account(self):
            return SimpleNamespace(portfolio_value=10_000.0, buying_power=10_000.0)

        def submit_order(self, order):
            submitted.append(order)
            return SimpleNamespace(
                id="paper-limit-2",
                status="accepted",
                side="buy",
                qty=getattr(order, "qty", None),
                symbol=getattr(order, "symbol", None),
                type="limit",
            )

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_aw = types.ModuleType("alpaca_wrapper")
    fake_aw.ALP_KEY_ID = "paper-key"
    fake_aw.ALP_SECRET_KEY = "paper-secret"
    fake_aw.TradingClient = _TradingClient
    fake_aw.OrderType = SimpleNamespace(LIMIT="limit")
    fake_aw.OrderSide = SimpleNamespace(
        BUY=SimpleNamespace(value="buy"),
        SELL=SimpleNamespace(value="sell"),
    )
    fake_aw.LimitOrderRequest = _LimitOrderRequest
    fake_aw.latest_data = lambda symbol: SimpleNamespace(ask_price=101.0, bid_price=100.0)
    fake_aw._midpoint_limit_price = lambda symbol, side, price: 100.5
    fake_aw._get_time_in_force_for_qty = lambda qty, symbol: "day"
    fake_aw.data_client = SimpleNamespace(
        get_stock_bars=lambda _request: {"AAPL": [SimpleNamespace(close=100.0)]}
    )

    fake_requests = types.ModuleType("alpaca.data.requests")
    fake_requests.StockBarsRequest = lambda **kwargs: kwargs
    fake_timeframe = types.ModuleType("alpaca.data.timeframe")
    fake_timeframe.TimeFrame = SimpleNamespace(Hour="hour")

    real_open = builtins.open

    def fail_trade_log_open(file, *args, **kwargs):
        if str(file).endswith("strategy_state/mixed23_daily_trades.jsonl"):
            raise OSError("disk full")
        return real_open(file, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "alpaca_wrapper", fake_aw)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", fake_timeframe)
    monkeypatch.setattr(builtins, "open", fail_trade_log_open)
    monkeypatch.chdir(tmp_path)

    result = prod.execute_signal_result(
        {"symbol": "AAPL", "direction": "LONG", "confidence": 0.9},
        allocation_pct=10.0,
    )

    assert result.submitted is True
    assert result.status == "submitted"
    assert result.reason == "order_submitted"
    assert result.log_write_error == "disk full"
    assert len(submitted) == 1


def test_execute_signal_rejects_non_finite_account_values(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    submitted: list[object] = []

    class _TradingClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_account(self):
            return SimpleNamespace(portfolio_value=float("nan"), buying_power=10_000.0)

        def submit_order(self, order):
            submitted.append(order)
            return SimpleNamespace(id="unexpected")

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_aw = types.ModuleType("alpaca_wrapper")
    fake_aw.ALP_KEY_ID = "paper-key"
    fake_aw.ALP_SECRET_KEY = "paper-secret"
    fake_aw.TradingClient = _TradingClient
    fake_aw.OrderType = SimpleNamespace(LIMIT="limit")
    fake_aw.OrderSide = SimpleNamespace(
        BUY=SimpleNamespace(value="buy"),
        SELL=SimpleNamespace(value="sell"),
    )
    fake_aw.LimitOrderRequest = _LimitOrderRequest
    fake_aw.latest_data = lambda symbol: SimpleNamespace(ask_price=101.0, bid_price=100.0)
    fake_aw._midpoint_limit_price = lambda symbol, side, price: 100.5
    fake_aw._get_time_in_force_for_qty = lambda qty, symbol: "day"
    fake_aw.data_client = SimpleNamespace(
        get_stock_bars=lambda _request: {"AAPL": [SimpleNamespace(close=100.0)]}
    )

    fake_requests = types.ModuleType("alpaca.data.requests")
    fake_requests.StockBarsRequest = lambda **kwargs: kwargs
    fake_timeframe = types.ModuleType("alpaca.data.timeframe")
    fake_timeframe.TimeFrame = SimpleNamespace(Hour="hour")

    monkeypatch.setitem(sys.modules, "alpaca_wrapper", fake_aw)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", fake_timeframe)
    monkeypatch.chdir(tmp_path)

    result = prod.execute_signal_result(
        {"symbol": "AAPL", "direction": "LONG", "confidence": 0.9},
        allocation_pct=10.0,
    )

    assert result.submitted is False
    assert result.status == "rejected"
    assert result.reason == "invalid_portfolio_value"
    assert submitted == []


def test_execute_signal_rejects_invalid_limit_price(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    submitted: list[object] = []

    class _TradingClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_account(self):
            return SimpleNamespace(portfolio_value=10_000.0, buying_power=10_000.0)

        def submit_order(self, order):
            submitted.append(order)
            return SimpleNamespace(id="unexpected")

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_aw = types.ModuleType("alpaca_wrapper")
    fake_aw.ALP_KEY_ID = "paper-key"
    fake_aw.ALP_SECRET_KEY = "paper-secret"
    fake_aw.TradingClient = _TradingClient
    fake_aw.OrderType = SimpleNamespace(LIMIT="limit")
    fake_aw.OrderSide = SimpleNamespace(
        BUY=SimpleNamespace(value="buy"),
        SELL=SimpleNamespace(value="sell"),
    )
    fake_aw.LimitOrderRequest = _LimitOrderRequest
    fake_aw.latest_data = lambda symbol: SimpleNamespace(ask_price=101.0, bid_price=100.0)
    fake_aw._midpoint_limit_price = lambda symbol, side, price: float("nan")
    fake_aw._get_time_in_force_for_qty = lambda qty, symbol: "day"
    fake_aw.data_client = SimpleNamespace(
        get_stock_bars=lambda _request: {"AAPL": [SimpleNamespace(close=100.0)]}
    )

    fake_requests = types.ModuleType("alpaca.data.requests")
    fake_requests.StockBarsRequest = lambda **kwargs: kwargs
    fake_timeframe = types.ModuleType("alpaca.data.timeframe")
    fake_timeframe.TimeFrame = SimpleNamespace(Hour="hour")

    monkeypatch.setitem(sys.modules, "alpaca_wrapper", fake_aw)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", fake_timeframe)
    monkeypatch.chdir(tmp_path)

    result = prod.execute_signal_result(
        {"symbol": "AAPL", "direction": "LONG", "confidence": 0.9},
        allocation_pct=10.0,
    )

    assert result.submitted is False
    assert result.status == "rejected"
    assert result.reason == "invalid_limit_price"
    assert submitted == []


def test_execute_signal_rejects_when_limit_price_calculation_raises(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    submitted: list[object] = []

    class _TradingClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_account(self):
            return SimpleNamespace(portfolio_value=10_000.0, buying_power=10_000.0)

        def submit_order(self, order):
            submitted.append(order)
            return SimpleNamespace(id="unexpected")

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_aw = types.ModuleType("alpaca_wrapper")
    fake_aw.ALP_KEY_ID = "paper-key"
    fake_aw.ALP_SECRET_KEY = "paper-secret"
    fake_aw.TradingClient = _TradingClient
    fake_aw.OrderType = SimpleNamespace(LIMIT="limit")
    fake_aw.OrderSide = SimpleNamespace(
        BUY=SimpleNamespace(value="buy"),
        SELL=SimpleNamespace(value="sell"),
    )
    fake_aw.LimitOrderRequest = _LimitOrderRequest
    fake_aw.latest_data = lambda symbol: SimpleNamespace(ask_price=101.0, bid_price=100.0)
    fake_aw._midpoint_limit_price = lambda symbol, side, price: (_ for _ in ()).throw(RuntimeError("quote service down"))
    fake_aw._get_time_in_force_for_qty = lambda qty, symbol: "day"
    fake_aw.data_client = SimpleNamespace(
        get_stock_bars=lambda _request: {"AAPL": [SimpleNamespace(close=100.0)]}
    )

    fake_requests = types.ModuleType("alpaca.data.requests")
    fake_requests.StockBarsRequest = lambda **kwargs: kwargs
    fake_timeframe = types.ModuleType("alpaca.data.timeframe")
    fake_timeframe.TimeFrame = SimpleNamespace(Hour="hour")

    monkeypatch.setitem(sys.modules, "alpaca_wrapper", fake_aw)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", fake_timeframe)
    monkeypatch.chdir(tmp_path)

    result = prod.execute_signal_result(
        {"symbol": "AAPL", "direction": "LONG", "confidence": 0.9},
        allocation_pct=10.0,
    )

    assert result.submitted is False
    assert result.status == "rejected"
    assert result.reason == "limit_price_calculation_failed"
    assert submitted == []


def test_execute_signal_rejects_when_time_in_force_selection_raises(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    submitted: list[object] = []

    class _TradingClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_account(self):
            return SimpleNamespace(portfolio_value=10_000.0, buying_power=10_000.0)

        def submit_order(self, order):
            submitted.append(order)
            return SimpleNamespace(id="unexpected")

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_aw = types.ModuleType("alpaca_wrapper")
    fake_aw.ALP_KEY_ID = "paper-key"
    fake_aw.ALP_SECRET_KEY = "paper-secret"
    fake_aw.TradingClient = _TradingClient
    fake_aw.OrderType = SimpleNamespace(LIMIT="limit")
    fake_aw.OrderSide = SimpleNamespace(
        BUY=SimpleNamespace(value="buy"),
        SELL=SimpleNamespace(value="sell"),
    )
    fake_aw.LimitOrderRequest = _LimitOrderRequest
    fake_aw.latest_data = lambda symbol: SimpleNamespace(ask_price=101.0, bid_price=100.0)
    fake_aw._midpoint_limit_price = lambda symbol, side, price: 100.5
    fake_aw._get_time_in_force_for_qty = lambda qty, symbol: (_ for _ in ()).throw(RuntimeError("tif unavailable"))
    fake_aw.data_client = SimpleNamespace(
        get_stock_bars=lambda _request: {"AAPL": [SimpleNamespace(close=100.0)]}
    )

    fake_requests = types.ModuleType("alpaca.data.requests")
    fake_requests.StockBarsRequest = lambda **kwargs: kwargs
    fake_timeframe = types.ModuleType("alpaca.data.timeframe")
    fake_timeframe.TimeFrame = SimpleNamespace(Hour="hour")

    monkeypatch.setitem(sys.modules, "alpaca_wrapper", fake_aw)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", fake_timeframe)
    monkeypatch.chdir(tmp_path)

    result = prod.execute_signal_result(
        {"symbol": "AAPL", "direction": "LONG", "confidence": 0.9},
        allocation_pct=10.0,
    )

    assert result.submitted is False
    assert result.status == "rejected"
    assert result.reason == "time_in_force_unavailable"
    assert submitted == []


def test_read_mktd_context_reads_symbol_table_and_last_prices(tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    path = tmp_path / "sample.bin"
    num_symbols = 2
    num_timesteps = 3
    features_per_sym = 16
    price_features = 5
    symbols = ["AAPL", "BTCUSD"]
    prices = np.zeros((num_timesteps, num_symbols, price_features), dtype=np.float32)
    prices[-1, 0] = np.array([10.0, 11.0, 9.5, 10.5, 100.0], dtype=np.float32)
    prices[-1, 1] = np.array([20.0, 21.0, 19.5, 20.5, 200.0], dtype=np.float32)

    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD",
        2,
        num_symbols,
        num_timesteps,
        features_per_sym,
        price_features,
        b"\x00" * 40,
    )
    symbol_table = b"".join(symbol.encode("ascii").ljust(16, b"\x00") for symbol in symbols)
    features = np.zeros((num_timesteps, num_symbols, features_per_sym), dtype=np.float32)
    mask = np.ones((num_timesteps, num_symbols), dtype=np.uint8)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(symbol_table)
        handle.write(features.tobytes())
        handle.write(prices.tobytes())
        handle.write(mask.tobytes())

    context = prod._read_mktd_context(path)

    assert context.num_symbols == num_symbols
    assert context.num_timesteps == num_timesteps
    assert context.features_per_sym == features_per_sym
    assert context.symbol_names == symbols
    np.testing.assert_allclose(context.last_prices, prices[-1])


def test_read_mktd_context_rejects_truncated_header(tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    path = tmp_path / "truncated.bin"
    path.write_bytes(b"MKTD")

    with pytest.raises(ValueError, match=r"invalid MKTD header: expected 64 bytes, found 4"):
        prod._read_mktd_context(path)


def test_read_mktd_context_rejects_invalid_magic(tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    path = tmp_path / "invalid.bin"
    path.write_bytes(struct.pack("<4sIIIII40s", b"NOPE", 2, 1, 1, 16, 5, b"\x00" * 40))

    with pytest.raises(ValueError, match=r"invalid MKTD magic b'NOPE'"):
        prod._read_mktd_context(path)


def test_read_mktd_context_rejects_truncated_price_section(tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    path = tmp_path / "short-prices.bin"
    header = struct.pack("<4sIIIII40s", b"MKTD", 2, 1, 1, 16, 5, b"\x00" * 40)
    symbol_table = b"AAPL".ljust(16, b"\x00")
    features = np.zeros((1, 1, 16), dtype=np.float32).tobytes()
    truncated_prices = np.zeros((1, 4), dtype=np.float32).tobytes()
    path.write_bytes(header + symbol_table + features + truncated_prices)

    with pytest.raises(ValueError, match=r"truncated MKTD price section: expected 20 bytes, found 16"):
        prod._read_mktd_context(path)


def test_load_symbol_price_frame_reloads_after_file_change(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    path = tmp_path / "AAPL.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01,1,2,0.5,1.5,100\n",
        encoding="utf-8",
    )
    prod._SYMBOL_PRICE_FRAME_CACHE.clear()

    read_calls = 0
    real_read_csv = prod.pd.read_csv

    def counting_read_csv(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(prod.pd, "read_csv", counting_read_csv)

    first = prod._load_symbol_price_frame(path, "AAPL")
    second = prod._load_symbol_price_frame(path, "AAPL")

    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01,1,2,0.5,1.5,100\n"
        "2026-04-02,2,3,1.5,2.5,150\n",
        encoding="utf-8",
    )
    stat_result = path.stat()
    os.utime(path, ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1))
    third = prod._load_symbol_price_frame(path, "AAPL")

    assert read_calls == 2
    assert second is first
    assert list(third.index.strftime("%Y-%m-%d")) == ["2026-04-01", "2026-04-02"]


def test_load_symbol_price_frame_caches_validation_errors(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    path = tmp_path / "AAPL.csv"
    path.write_text(
        "timestamp,open,high,low,close\n"
        "2026-04-01,1,2,0.5,1.5\n",
        encoding="utf-8",
    )
    prod._SYMBOL_PRICE_FRAME_CACHE.clear()

    read_calls = 0
    real_read_csv = prod.pd.read_csv

    def counting_read_csv(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(prod.pd, "read_csv", counting_read_csv)

    with pytest.raises(ValueError, match="missing required columns: volume"):
        prod._load_symbol_price_frame(path, "AAPL")

    with pytest.raises(ValueError, match="missing required columns: volume"):
        prod._load_symbol_price_frame(path, "AAPL")

    assert read_calls == 1


def test_load_symbol_price_frame_reloads_after_invalid_file_is_fixed(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    path = tmp_path / "AAPL.csv"
    path.write_text(
        "timestamp,open,high,low,close\n"
        "2026-04-01,1,2,0.5,1.5\n",
        encoding="utf-8",
    )
    prod._SYMBOL_PRICE_FRAME_CACHE.clear()

    read_calls = 0
    real_read_csv = prod.pd.read_csv

    def counting_read_csv(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(prod.pd, "read_csv", counting_read_csv)

    with pytest.raises(ValueError, match="missing required columns: volume"):
        prod._load_symbol_price_frame(path, "AAPL")

    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01,1,2,0.5,1.5,100\n",
        encoding="utf-8",
    )
    stat_result = path.stat()
    os.utime(path, ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1))

    repaired = prod._load_symbol_price_frame(path, "AAPL")

    assert read_calls == 2
    assert list(repaired.columns) == ["open", "high", "low", "close", "volume"]
    assert list(repaired.index.strftime("%Y-%m-%d")) == ["2026-04-01"]


def test_load_symbol_price_frame_serializes_concurrent_cache_misses(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    path = tmp_path / "AAPL.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01,1,2,0.5,1.5,100\n",
        encoding="utf-8",
    )
    prod._SYMBOL_PRICE_FRAME_CACHE.clear()

    first_read_entered = threading.Event()
    second_worker_started = threading.Event()
    release_first_read = threading.Event()
    read_calls = 0
    real_read_csv = prod.pd.read_csv
    results: list[object] = []
    errors: list[BaseException] = []

    def gated_read_csv(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        if read_calls == 1:
            first_read_entered.set()
            assert second_worker_started.wait(timeout=5)
            assert release_first_read.wait(timeout=5)
        return real_read_csv(*args, **kwargs)

    def worker(*, mark_started: threading.Event | None = None) -> None:
        try:
            if mark_started is not None:
                mark_started.set()
            results.append(prod._load_symbol_price_frame(path, "AAPL"))
        except BaseException as exc:  # pragma: no cover - assertions should keep this empty
            errors.append(exc)

    monkeypatch.setattr(prod.pd, "read_csv", gated_read_csv)

    first_thread = threading.Thread(target=worker)
    second_thread = threading.Thread(target=worker, kwargs={"mark_started": second_worker_started})
    first_thread.start()
    assert first_read_entered.wait(timeout=5)
    second_thread.start()
    release_first_read.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not errors
    assert read_calls == 1
    assert len(results) == 2


def test_load_symbol_price_frame_allows_parallel_reads_for_different_files(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    first_path = tmp_path / "AAPL.csv"
    second_path = tmp_path / "MSFT.csv"
    csv_text = (
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01,1,2,0.5,1.5,100\n"
    )
    first_path.write_text(csv_text, encoding="utf-8")
    second_path.write_text(csv_text, encoding="utf-8")
    prod._SYMBOL_PRICE_FRAME_CACHE.clear()

    first_read_entered = threading.Event()
    second_read_entered = threading.Event()
    release_reads = threading.Event()
    errors: list[BaseException] = []
    real_read_csv = prod.pd.read_csv

    def gated_read_csv(path_arg, *args, **kwargs):
        resolved = Path(path_arg).resolve()
        if resolved == first_path.resolve():
            first_read_entered.set()
            assert release_reads.wait(timeout=5)
        elif resolved == second_path.resolve():
            second_read_entered.set()
            assert release_reads.wait(timeout=5)
        return real_read_csv(path_arg, *args, **kwargs)

    def worker(path_arg: Path, symbol: str) -> None:
        try:
            prod._load_symbol_price_frame(path_arg, symbol)
        except BaseException as exc:  # pragma: no cover - assertions should keep this empty
            errors.append(exc)

    monkeypatch.setattr(prod.pd, "read_csv", gated_read_csv)

    first_thread = threading.Thread(target=worker, args=(first_path, "AAPL"))
    second_thread = threading.Thread(target=worker, args=(second_path, "MSFT"))
    first_thread.start()
    assert first_read_entered.wait(timeout=5)
    second_thread.start()
    assert second_read_entered.wait(timeout=5)
    release_reads.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not errors


def test_symbol_price_frame_path_lock_registry_drops_unused_locks(tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    path_lock = prod._get_symbol_price_frame_path_lock(tmp_path / "AAPL.csv")

    assert len(prod._SYMBOL_PRICE_FRAME_PATH_LOCKS) == 1

    del path_lock
    gc.collect()

    assert len(prod._SYMBOL_PRICE_FRAME_PATH_LOCKS) == 0


def test_run_inference_encodes_observation_once(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_bin = tmp_path / "sample.bin"
    _write_mktd(data_bin, ["AAPL", "BTCUSD"])
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    policy_instances: list[object] = []
    vec_init_kwargs: dict[str, object] = {}

    class FakeTradingPolicy:
        def __init__(self, obs_size: int, num_actions: int, hidden_size: int):
            self.obs_size = obs_size
            self.num_actions = num_actions
            self.hidden_size = hidden_size
            self.encoder_calls = 0
            policy_instances.append(self)

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def to(self, _device):
            return self

        def encoder(self, obs_tensor):
            self.encoder_calls += 1
            return obs_tensor + 1.0

        def actor(self, encoded_obs):
            assert torch.all(encoded_obs > 0)
            return torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        def critic(self, encoded_obs):
            assert torch.all(encoded_obs > 0)
            return torch.tensor([[0.25]], dtype=torch.float32)

    reset_calls: list[int] = []
    close_calls: list[object] = []

    fake_train = types.ModuleType("pufferlib_market.train")
    fake_train.TradingPolicy = FakeTradingPolicy
    fake_binding = types.ModuleType("pufferlib_market.binding")

    def fake_shared(*, data_path: str):
        assert data_path == str(data_bin.resolve())

    def fake_vec_init(obs_buf, act_buf, rew_buf, term_buf, trunc_buf, num_envs, seed, **kwargs):
        assert num_envs == 1
        assert seed == 42
        obs_buf[:] = 0.0
        vec_init_kwargs.update(kwargs)
        return "vec-handle"

    def fake_vec_reset(handle, seed):
        reset_calls.append(seed)
        assert handle == "vec-handle"

    def fake_vec_close(handle):
        close_calls.append(handle)

    fake_binding.shared = fake_shared
    fake_binding.vec_init = fake_vec_init
    fake_binding.vec_reset = fake_vec_reset
    fake_binding.vec_close = fake_vec_close

    monkeypatch.setitem(sys.modules, "pufferlib_market.train", fake_train)
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    pufferlib_market_pkg = sys.modules.get("pufferlib_market")
    if pufferlib_market_pkg is not None:
        monkeypatch.setattr(pufferlib_market_pkg, "train", fake_train, raising=False)
        monkeypatch.setattr(pufferlib_market_pkg, "binding", fake_binding, raising=False)
    monkeypatch.setattr(prod.torch, "load", lambda *args, **kwargs: {"model": {}})
    monkeypatch.setattr(prod.torch.cuda, "is_available", lambda: False)

    signal = prod.run_inference(
        str(checkpoint),
        str(data_bin),
        hidden_size=32,
        max_episode_steps=17,
        fill_slippage_bps=7.5,
        fee_rate=0.001,
        max_leverage=1.25,
        periods_per_year=252.0,
        action_max_offset_bps=4.5,
    )

    assert signal["direction"] == "LONG"
    assert signal["symbol"] == "AAPL"
    assert reset_calls == [42]
    assert close_calls == ["vec-handle"]
    assert len(policy_instances) == 1
    assert policy_instances[0].encoder_calls == 1
    assert policy_instances[0].obs_size == 2 * 16 + prod.MIXED_ENV_PORTFOLIO_FEATURES + 2
    assert vec_init_kwargs["max_steps"] == 17
    assert vec_init_kwargs["fill_slippage_bps"] == 7.5
    assert vec_init_kwargs["fee_rate"] == 0.001
    assert vec_init_kwargs["max_leverage"] == 1.25
    assert vec_init_kwargs["periods_per_year"] == 252.0
    assert vec_init_kwargs["action_max_offset_bps"] == 4.5


def test_run_inference_uses_header_features_per_symbol_for_obs_size(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_bin = tmp_path / "sample.bin"
    _write_mktd(data_bin, ["AAPL", "BTCUSD"], features_per_sym=20)
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    policy_instances: list[object] = []

    class FakeTradingPolicy:
        def __init__(self, obs_size: int, num_actions: int, hidden_size: int):
            self.obs_size = obs_size
            self.num_actions = num_actions
            self.hidden_size = hidden_size
            policy_instances.append(self)

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def to(self, _device):
            return self

        def encoder(self, obs_tensor):
            return obs_tensor

        def actor(self, _encoded_obs):
            return torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        def critic(self, _encoded_obs):
            return torch.tensor([[0.0]], dtype=torch.float32)

    fake_train = types.ModuleType("pufferlib_market.train")
    fake_train.TradingPolicy = FakeTradingPolicy
    fake_binding = types.ModuleType("pufferlib_market.binding")
    fake_binding.shared = lambda **_kwargs: None
    fake_binding.vec_init = lambda *args, **kwargs: "vec-handle"
    fake_binding.vec_reset = lambda handle, seed: (handle, seed)
    fake_binding.vec_close = lambda handle: handle

    monkeypatch.setitem(sys.modules, "pufferlib_market.train", fake_train)
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    pufferlib_market_pkg = sys.modules.get("pufferlib_market")
    if pufferlib_market_pkg is not None:
        monkeypatch.setattr(pufferlib_market_pkg, "train", fake_train, raising=False)
        monkeypatch.setattr(pufferlib_market_pkg, "binding", fake_binding, raising=False)
    monkeypatch.setattr(prod.torch, "load", lambda *args, **kwargs: {"model": {}})
    monkeypatch.setattr(prod.torch.cuda, "is_available", lambda: False)

    prod.run_inference(str(checkpoint), str(data_bin), hidden_size=32)

    assert len(policy_instances) == 1
    assert policy_instances[0].obs_size == 2 * 20 + prod.MIXED_ENV_PORTFOLIO_FEATURES + 2


def test_main_once_reuses_validated_symbol_frames(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01,1,2,0.5,1.5,100\n"
        "2026-04-02,2,3,1.5,2.5,150\n",
        encoding="utf-8",
    )
    prod._SYMBOL_PRICE_FRAME_CACHE.clear()

    read_calls = 0
    real_read_csv = prod.pd.read_csv

    def counting_read_csv(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return real_read_csv(*args, **kwargs)

    def fake_run_inference(*_args, **_kwargs):
        return {
            "direction": "FLAT",
            "symbol": None,
            "confidence": 1.0,
            "value": 0.0,
            "action": 0,
            "timestamp": "2026-04-02T00:00:00+00:00",
            "all_probs": [1.0],
            "sym_names": ["AAPL"],
        }

    monkeypatch.setattr(prod.pd, "read_csv", counting_read_csv)
    monkeypatch.setattr(prod, "run_inference", fake_run_inference)

    rc = prod.main(
        [
            "--once",
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
        ]
    )

    assert rc == 0
    assert read_calls == 1


def test_run_inference_closes_vec_handle_when_inference_raises(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_bin = tmp_path / "sample.bin"
    _write_mktd(data_bin, ["AAPL", "BTCUSD"])
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    class FakeTradingPolicy:
        def __init__(self, _obs_size: int, _num_actions: int, _hidden_size: int):
            pass

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def to(self, _device):
            return self

        def encoder(self, obs_tensor):
            return obs_tensor

        def actor(self, _encoded_obs):
            raise RuntimeError("boom")

        def critic(self, _encoded_obs):
            raise AssertionError("critic should not run after actor failure")

    close_calls: list[object] = []

    fake_train = types.ModuleType("pufferlib_market.train")
    fake_train.TradingPolicy = FakeTradingPolicy
    fake_binding = types.ModuleType("pufferlib_market.binding")
    fake_binding.shared = lambda *, data_path: data_path
    fake_binding.vec_init = lambda *args, **kwargs: "vec-handle"
    fake_binding.vec_reset = lambda handle, seed: (handle, seed)
    fake_binding.vec_close = lambda handle: close_calls.append(handle)

    monkeypatch.setitem(sys.modules, "pufferlib_market.train", fake_train)
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    pufferlib_market_pkg = sys.modules.get("pufferlib_market")
    if pufferlib_market_pkg is not None:
        monkeypatch.setattr(pufferlib_market_pkg, "train", fake_train, raising=False)
        monkeypatch.setattr(pufferlib_market_pkg, "binding", fake_binding, raising=False)
    monkeypatch.setattr(prod.torch, "load", lambda *args, **kwargs: {"model": {}})
    monkeypatch.setattr(prod.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="boom"):
        prod.run_inference(str(checkpoint), str(data_bin), hidden_size=32)

    assert close_calls == ["vec-handle"]


def test_run_inference_uses_safe_checkpoint_loading_by_default(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_bin = tmp_path / "sample.bin"
    _write_mktd(data_bin, ["AAPL", "BTCUSD"])
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    load_calls: list[dict[str, object]] = []

    class FakeTradingPolicy:
        def __init__(self, _obs_size: int, _num_actions: int, _hidden_size: int):
            pass

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def to(self, _device):
            return self

        def encoder(self, obs_tensor):
            return obs_tensor

        def actor(self, _encoded_obs):
            return torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        def critic(self, _encoded_obs):
            return torch.tensor([[0.25]], dtype=torch.float32)

    fake_train = types.ModuleType("pufferlib_market.train")
    fake_train.TradingPolicy = FakeTradingPolicy
    fake_binding = types.ModuleType("pufferlib_market.binding")
    fake_binding.shared = lambda *, data_path: data_path
    fake_binding.vec_init = lambda *args, **kwargs: "vec-handle"
    fake_binding.vec_reset = lambda handle, seed: (handle, seed)
    fake_binding.vec_close = lambda handle: handle

    monkeypatch.setitem(sys.modules, "pufferlib_market.train", fake_train)
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    monkeypatch.setattr(prod.torch.cuda, "is_available", lambda: False)

    def fake_load(*args, **kwargs):
        load_calls.append(kwargs)
        return {"model": {}}

    monkeypatch.setattr(prod.torch, "load", fake_load)

    signal = prod.run_inference(str(checkpoint), str(data_bin), hidden_size=32)

    assert signal["direction"] == "LONG"
    assert load_calls == [{"map_location": torch.device("cpu"), "weights_only": True}]


def test_run_inference_supports_explicit_unsafe_checkpoint_loading(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_bin = tmp_path / "sample.bin"
    _write_mktd(data_bin, ["AAPL", "BTCUSD"])
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    load_calls: list[dict[str, object]] = []

    class FakeTradingPolicy:
        def __init__(self, _obs_size: int, _num_actions: int, _hidden_size: int):
            pass

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def to(self, _device):
            return self

        def encoder(self, obs_tensor):
            return obs_tensor

        def actor(self, _encoded_obs):
            return torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        def critic(self, _encoded_obs):
            return torch.tensor([[0.25]], dtype=torch.float32)

    fake_train = types.ModuleType("pufferlib_market.train")
    fake_train.TradingPolicy = FakeTradingPolicy
    fake_binding = types.ModuleType("pufferlib_market.binding")
    fake_binding.shared = lambda *, data_path: data_path
    fake_binding.vec_init = lambda *args, **kwargs: "vec-handle"
    fake_binding.vec_reset = lambda handle, seed: (handle, seed)
    fake_binding.vec_close = lambda handle: handle

    monkeypatch.setitem(sys.modules, "pufferlib_market.train", fake_train)
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    monkeypatch.setattr(prod.torch.cuda, "is_available", lambda: False)

    def fake_load(*args, **kwargs):
        load_calls.append(kwargs)
        return {"model": {}}

    monkeypatch.setattr(prod.torch, "load", fake_load)

    signal = prod.run_inference(
        str(checkpoint),
        str(data_bin),
        hidden_size=32,
        allow_unsafe_checkpoint_loading=True,
    )

    assert signal["direction"] == "LONG"
    assert load_calls == [{"map_location": torch.device("cpu"), "weights_only": False}]


def test_run_inference_reuses_cached_policy_for_unchanged_checkpoint(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_bin = tmp_path / "sample.bin"
    _write_mktd(data_bin, ["AAPL", "BTCUSD"])
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    load_calls: list[dict[str, object]] = []
    policy_init_calls: list[tuple[int, int, int]] = []

    class FakeTradingPolicy:
        def __init__(self, obs_size: int, num_actions: int, hidden_size: int):
            policy_init_calls.append((obs_size, num_actions, hidden_size))

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def to(self, _device):
            return self

        def encoder(self, obs_tensor):
            return obs_tensor

        def actor(self, _encoded_obs):
            return torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        def critic(self, _encoded_obs):
            return torch.tensor([[0.25]], dtype=torch.float32)

    fake_train = types.ModuleType("pufferlib_market.train")
    fake_train.TradingPolicy = FakeTradingPolicy
    fake_binding = types.ModuleType("pufferlib_market.binding")
    fake_binding.shared = lambda *, data_path: data_path
    fake_binding.vec_init = lambda *args, **kwargs: "vec-handle"
    fake_binding.vec_reset = lambda handle, seed: (handle, seed)
    fake_binding.vec_close = lambda handle: handle

    monkeypatch.setitem(sys.modules, "pufferlib_market.train", fake_train)
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    monkeypatch.setattr(prod.torch.cuda, "is_available", lambda: False)

    def fake_load(*args, **kwargs):
        load_calls.append(kwargs)
        return {"model": {}}

    monkeypatch.setattr(prod.torch, "load", fake_load)
    prod._CHECKPOINT_PAYLOAD_CACHE.clear()
    prod._INFERENCE_POLICY_CACHE.clear()
    try:
        first_signal = prod.run_inference(str(checkpoint), str(data_bin), hidden_size=32)
        second_signal = prod.run_inference(str(checkpoint), str(data_bin), hidden_size=32)
    finally:
        prod._CHECKPOINT_PAYLOAD_CACHE.clear()
        prod._INFERENCE_POLICY_CACHE.clear()

    assert first_signal["direction"] == "LONG"
    assert second_signal["direction"] == "LONG"
    assert load_calls == [{"map_location": torch.device("cpu"), "weights_only": True}]
    assert policy_init_calls == [(39, 5, 32)]


def test_run_inference_invalidates_cached_policy_when_checkpoint_changes(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_bin = tmp_path / "sample.bin"
    _write_mktd(data_bin, ["AAPL", "BTCUSD"])
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    load_calls: list[dict[str, object]] = []
    policy_init_calls: list[tuple[int, int, int]] = []

    class FakeTradingPolicy:
        def __init__(self, obs_size: int, num_actions: int, hidden_size: int):
            policy_init_calls.append((obs_size, num_actions, hidden_size))

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def to(self, _device):
            return self

        def encoder(self, obs_tensor):
            return obs_tensor

        def actor(self, _encoded_obs):
            return torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        def critic(self, _encoded_obs):
            return torch.tensor([[0.25]], dtype=torch.float32)

    fake_train = types.ModuleType("pufferlib_market.train")
    fake_train.TradingPolicy = FakeTradingPolicy
    fake_binding = types.ModuleType("pufferlib_market.binding")
    fake_binding.shared = lambda *, data_path: data_path
    fake_binding.vec_init = lambda *args, **kwargs: "vec-handle"
    fake_binding.vec_reset = lambda handle, seed: (handle, seed)
    fake_binding.vec_close = lambda handle: handle

    monkeypatch.setitem(sys.modules, "pufferlib_market.train", fake_train)
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    monkeypatch.setattr(prod.torch.cuda, "is_available", lambda: False)

    def fake_load(*args, **kwargs):
        load_calls.append(kwargs)
        return {"model": {}}

    monkeypatch.setattr(prod.torch, "load", fake_load)
    prod._CHECKPOINT_PAYLOAD_CACHE.clear()
    prod._INFERENCE_POLICY_CACHE.clear()
    try:
        first_signal = prod.run_inference(str(checkpoint), str(data_bin), hidden_size=32)
        checkpoint.write_bytes(checkpoint.read_bytes() + b"!")
        second_signal = prod.run_inference(str(checkpoint), str(data_bin), hidden_size=32)
    finally:
        prod._CHECKPOINT_PAYLOAD_CACHE.clear()
        prod._INFERENCE_POLICY_CACHE.clear()

    assert first_signal["direction"] == "LONG"
    assert second_signal["direction"] == "LONG"
    assert load_calls == [
        {"map_location": torch.device("cpu"), "weights_only": True},
        {"map_location": torch.device("cpu"), "weights_only": True},
    ]
    assert policy_init_calls == [(39, 5, 32), (39, 5, 32)]


def test_run_inference_policy_cache_evicts_older_checkpoints_when_capacity_is_exceeded(
    monkeypatch, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    data_bin = tmp_path / "sample.bin"
    _write_mktd(data_bin, ["AAPL", "BTCUSD"])
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    _write_checkpoint(checkpoint_a)
    _write_checkpoint(checkpoint_b)

    load_calls: list[dict[str, object]] = []
    policy_init_calls: list[tuple[int, int, int]] = []

    class FakeTradingPolicy:
        def __init__(self, obs_size: int, num_actions: int, hidden_size: int):
            policy_init_calls.append((obs_size, num_actions, hidden_size))

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def to(self, _device):
            return self

        def encoder(self, obs_tensor):
            return obs_tensor

        def actor(self, _encoded_obs):
            return torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        def critic(self, _encoded_obs):
            return torch.tensor([[0.25]], dtype=torch.float32)

    fake_train = types.ModuleType("pufferlib_market.train")
    fake_train.TradingPolicy = FakeTradingPolicy
    fake_binding = types.ModuleType("pufferlib_market.binding")
    fake_binding.shared = lambda *, data_path: data_path
    fake_binding.vec_init = lambda *args, **kwargs: "vec-handle"
    fake_binding.vec_reset = lambda handle, seed: (handle, seed)
    fake_binding.vec_close = lambda handle: handle

    monkeypatch.setitem(sys.modules, "pufferlib_market.train", fake_train)
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    monkeypatch.setattr(prod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(prod, "_CHECKPOINT_PAYLOAD_CACHE_MAX_ENTRIES", 1)
    monkeypatch.setattr(prod, "_INFERENCE_POLICY_CACHE_MAX_ENTRIES", 1)

    def fake_load(*args, **kwargs):
        load_calls.append(kwargs)
        return {"model": {}}

    monkeypatch.setattr(prod.torch, "load", fake_load)
    prod._CHECKPOINT_PAYLOAD_CACHE.clear()
    prod._INFERENCE_POLICY_CACHE.clear()
    try:
        prod.run_inference(str(checkpoint_a), str(data_bin), hidden_size=32)
        prod.run_inference(str(checkpoint_b), str(data_bin), hidden_size=32)
        prod.run_inference(str(checkpoint_a), str(data_bin), hidden_size=32)
    finally:
        prod._CHECKPOINT_PAYLOAD_CACHE.clear()
        prod._INFERENCE_POLICY_CACHE.clear()

    assert load_calls == [
        {"map_location": torch.device("cpu"), "weights_only": True},
        {"map_location": torch.device("cpu"), "weights_only": True},
        {"map_location": torch.device("cpu"), "weights_only": True},
    ]
    assert policy_init_calls == [(39, 5, 32), (39, 5, 32), (39, 5, 32)]


def test_load_checkpoint_payload_rejects_unsafe_checkpoint_without_opt_in(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    def fake_load(*args, **kwargs):
        raise RuntimeError("Weights only load failed due to unsupported global")

    monkeypatch.setattr(prod.torch, "load", fake_load)

    with pytest.raises(
        ValueError,
        match=r"safe checkpoint loading failed: Weights only load failed due to unsupported global.*--allow-unsafe-checkpoint-loading",
    ):
        prod._load_checkpoint_payload(checkpoint, torch.device("cpu"))


def test_load_checkpoint_payload_reuses_cached_payload_for_unchanged_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    load_calls: list[dict[str, object]] = []

    def fake_load(*args, **kwargs):
        load_calls.append(kwargs)
        return {"model": {}}

    monkeypatch.setattr(prod.torch, "load", fake_load)
    prod._CHECKPOINT_PAYLOAD_CACHE.clear()
    try:
        first_payload = prod._load_checkpoint_payload(checkpoint, torch.device("cpu"))
        second_payload = prod._load_checkpoint_payload(checkpoint, torch.device("cpu"))
    finally:
        prod._CHECKPOINT_PAYLOAD_CACHE.clear()

    assert first_payload is second_payload
    assert load_calls == [{"map_location": torch.device("cpu"), "weights_only": True}]


def test_load_checkpoint_payload_evicts_older_entries_when_capacity_is_exceeded(
    monkeypatch, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    _write_checkpoint(checkpoint_a)
    _write_checkpoint(checkpoint_b)

    load_calls: list[dict[str, object]] = []

    def fake_load(*args, **kwargs):
        load_calls.append(kwargs)
        return {"model": {}}

    monkeypatch.setattr(prod.torch, "load", fake_load)
    monkeypatch.setattr(prod, "_CHECKPOINT_PAYLOAD_CACHE_MAX_ENTRIES", 1)
    prod._CHECKPOINT_PAYLOAD_CACHE.clear()
    try:
        prod._load_checkpoint_payload(checkpoint_a, torch.device("cpu"))
        prod._load_checkpoint_payload(checkpoint_b, torch.device("cpu"))
        prod._load_checkpoint_payload(checkpoint_a, torch.device("cpu"))
    finally:
        prod._CHECKPOINT_PAYLOAD_CACHE.clear()

    assert load_calls == [
        {"map_location": torch.device("cpu"), "weights_only": True},
        {"map_location": torch.device("cpu"), "weights_only": True},
        {"map_location": torch.device("cpu"), "weights_only": True},
    ]


def test_main_print_config_reports_normalized_symbols_and_missing_setup(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    rc = prod.main(
        [
            "--print-config",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--data-dir",
            str(tmp_path / "missing-data"),
            "--symbols",
            "aapl",
            "AAPL",
            "",
            "btcusd",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["symbols"] == ["AAPL", "BTCUSD"]
    assert payload["removed_duplicate_symbols"] == ["AAPL"]
    assert payload["ignored_symbol_inputs"] == [""]
    assert payload["symbol_source"] == "cli"
    assert payload["symbols_file"] is None
    assert payload["configured_run_mode"] == "config_only"
    assert payload["suggested_run_mode"] == "once"
    assert payload["stock_symbol_count"] == 1
    assert payload["crypto_symbol_count"] == 1
    assert payload["usable_symbols"] == []
    assert payload["usable_symbol_count"] == 0
    assert payload["latest_local_data_date"] is None
    assert payload["oldest_local_data_date"] is None
    assert payload["stale_symbol_data"] == {}
    assert payload["stale_symbol_count"] == 0
    assert payload["symbol_details"] == {
        "AAPL": {
            "asset_class": "stock",
            "local_data_date": None,
            "reason": "missing local CSV data",
            "status": "missing",
        },
        "BTCUSD": {
            "asset_class": "crypto",
            "local_data_date": None,
            "reason": "missing local CSV data",
            "status": "missing",
        },
    }
    assert payload["daemon_schedule_utc"] == "00:05 UTC"
    assert payload["summary"].startswith("not ready: config_only config for 2 symbols")
    assert "usable local data for 0/2 symbols" in payload["summary"]
    assert "--check-config" in payload["check_command_preview"]
    assert "--once" in payload["run_command_preview"]
    assert "--dry-run" not in payload["run_command_preview"]
    assert "--once" in payload["safe_command_preview"]
    assert "--dry-run" in payload["safe_command_preview"]
    assert any(step.startswith("Set --checkpoint to a trained mixed-daily policy file:") for step in payload["next_steps"])
    assert any(step.startswith("Set --data-dir to a directory containing per-symbol OHLCV CSV files:") for step in payload["next_steps"])
    assert any(step.startswith("Re-check setup after fixes: ") for step in payload["next_steps"])
    assert payload["ready"] is False
    assert any("checkpoint not found" in error for error in payload["errors"])
    assert any("data_dir not found" in error for error in payload["errors"])


def test_main_once_returns_actionable_preflight_error_when_config_is_not_ready(
    capsys, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    rc = prod.main(
        [
            "--once",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--data-dir",
            str(tmp_path / "missing-data"),
            "--symbols",
            "AAPL",
            "BTCUSD",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "Configuration not ready:" in captured.err
    assert "not ready: once config for 2 symbols" in captured.err
    assert f"checkpoint not found: {tmp_path / 'missing.pt'}" in captured.err
    assert f"data_dir not found: {tmp_path / 'missing-data'}" in captured.err
    assert "Check config: python -u trade_mixed_daily_prod.py --check-config" in captured.err
    assert "Next steps:" in captured.err
    assert "Re-check setup after fixes:" in captured.err


def test_main_once_reports_local_data_health_when_symbol_data_is_incomplete(
    capsys, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01,1,1,1,1,100\n"
        "2026-04-03,1,1,1,1,100\n",
        encoding="utf-8",
    )
    (data_dir / "BTCUSD.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01,1,1,1,1,100\n"
        "2026-04-02,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--once",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
            "BTCUSD",
            "ETHUSD",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "Configuration not ready:" in captured.err
    assert "Local data health:" in captured.err
    assert "- usable symbols: 2/3" in captured.err
    assert "- latest local data date: 2026-04-03" in captured.err
    assert "- stale symbols: BTCUSD (2026-04-02)" in captured.err
    assert "- missing symbols: ETHUSD" in captured.err
    assert "Check config: python -u trade_mixed_daily_prod.py --check-config" in captured.err
    assert "Next steps:" in captured.err
    assert "- Refresh stale local CSVs for: BTCUSD" in captured.err


def test_main_print_config_reports_unsafe_checkpoint_loading_opt_in(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--print-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
            "--allow-unsafe-checkpoint-loading",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["allow_unsafe_checkpoint_loading"] is True
    assert "--allow-unsafe-checkpoint-loading" in payload["run_command_preview"]
    assert "--allow-unsafe-checkpoint-loading" in payload["safe_command_preview"]
    assert any("unsafe checkpoint loading enabled" in warning for warning in payload["warnings"])


def test_main_print_config_reports_duplicate_symbol_once(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    rc = prod.main(
        [
            "--print-config",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--data-dir",
            str(tmp_path / "missing-data"),
            "--symbols",
            "aapl",
            "AAPL",
            "aapl",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["symbols"] == ["AAPL"]
    assert payload["removed_duplicate_symbols"] == ["AAPL"]


def test_main_print_config_reports_unsafe_symbol_input(capsys) -> None:
    import trade_mixed_daily_prod as prod

    rc = prod.main(
        [
            "--print-config",
            "--symbols",
            "../AAPL",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["symbols"] == []
    assert any("Unsupported symbol: ../AAPL" in error for error in payload["errors"])
    assert "--symbols" not in payload["run_command_preview"]


def test_main_print_config_allows_standard_share_class_symbols(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, symbol_count=2)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "BRK.B.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )
    (data_dir / "BF-B.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--print-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "brk.b",
            "bf-b",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["symbols"] == ["BRK.B", "BF-B"]


def test_main_check_config_succeeds_with_existing_checkpoint_and_data(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "aapl",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ready"] is True
    assert payload["symbols"] == ["AAPL"]
    assert payload["usable_symbols"] == ["AAPL"]
    assert payload["usable_symbol_count"] == 1
    assert payload["checkpoint_arch"] == "mlp"
    assert payload["checkpoint_hidden_size"] == 1024
    assert payload["checkpoint_obs_size"] == 22
    assert payload["checkpoint_num_actions"] == 3
    assert payload["checkpoint_matches_runtime"] is True
    assert payload["checkpoint_inspection_error"] is None
    assert payload["expected_obs_size"] == 22
    assert payload["expected_num_actions"] == 3
    assert payload["missing_symbol_data"] == []
    assert payload["invalid_symbol_data"] == {}
    assert payload["latest_local_data_date"] == "2026-04-01"
    assert payload["oldest_local_data_date"] == "2026-04-01"
    assert payload["stale_symbol_data"] == {}
    assert payload["stale_symbol_count"] == 0
    assert payload["symbol_details"] == {
        "AAPL": {
            "asset_class": "stock",
            "local_data_date": "2026-04-01",
            "reason": None,
            "status": "usable",
        }
    }
    assert "--check-config" in payload["check_command_preview"]
    assert payload["next_steps"] == [
        f"Dry run first: {payload['safe_command_preview']}",
        f"Then run live: {payload['run_command_preview']}",
    ]


def test_main_check_config_text_emits_human_ready_summary_to_stderr(
    capsys, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--check-config",
            "--check-config-text",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["ready"] is True
    assert captured.err != ""
    assert "Configuration ready:" in captured.err
    assert "Suggested commands:" in captured.err
    assert f"- dry run: {payload['safe_command_preview']}" in captured.err
    assert f"- one-off run: {payload['run_command_preview']}" in captured.err
    assert "Additional next steps:" not in captured.err


def test_main_check_config_reports_stale_symbol_data(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, symbol_count=2)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01,1,1,1,1,100\n"
        "2026-04-03,1,1,1,1,100\n",
        encoding="utf-8",
    )
    (data_dir / "BTCUSD.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-04-01,1,1,1,1,100\n"
        "2026-04-02,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "aapl",
            "btcusd",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["usable_symbols"] == ["AAPL", "BTCUSD"]
    assert payload["usable_symbol_count"] == 2
    assert payload["latest_local_data_date"] == "2026-04-03"
    assert payload["oldest_local_data_date"] == "2026-04-02"
    assert payload["stale_symbol_data"] == {"BTCUSD": "2026-04-02"}
    assert payload["stale_symbol_count"] == 1
    assert payload["next_steps"] == [
        "Refresh stale local CSVs for: BTCUSD",
        f"Dry run first: {payload['safe_command_preview']}",
        f"Then run live: {payload['run_command_preview']}",
    ]
    assert payload["symbol_details"] == {
        "AAPL": {
            "asset_class": "stock",
            "local_data_date": "2026-04-03",
            "reason": None,
            "status": "usable",
        },
        "BTCUSD": {
            "asset_class": "crypto",
            "local_data_date": "2026-04-02",
            "reason": "local CSV data lags freshest symbol date 2026-04-03",
            "status": "stale",
        },
    }
    assert "usable local data for 2/2 symbols through 2026-04-03 (1 stale)" in payload["summary"]
    assert any("local CSV data for 1 symbols lags freshest date 2026-04-03" in warning for warning in payload["warnings"])


def test_main_check_config_text_emits_human_failure_summary_to_stderr(
    capsys, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    rc = prod.main(
        [
            "--check-config",
            "--check-config-text",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--data-dir",
            str(tmp_path / "missing-data"),
            "--symbols",
            "AAPL",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["ready"] is False
    assert "Configuration not ready:" in captured.err
    assert f"checkpoint not found: {tmp_path / 'missing.pt'}" in captured.err
    assert f"data_dir not found: {tmp_path / 'missing-data'}" in captured.err
    assert "Check config: python -u trade_mixed_daily_prod.py --check-config" in captured.err
    assert "Next steps:" in captured.err


def test_main_check_config_rejects_checkpoint_shape_mismatch(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, symbol_count=1)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )
    (data_dir / "BTCUSD.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
            "BTCUSD",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["checkpoint_arch"] == "mlp"
    assert payload["checkpoint_obs_size"] == 22
    assert payload["checkpoint_num_actions"] == 3
    assert payload["expected_obs_size"] == 39
    assert payload["expected_num_actions"] == 5
    assert payload["checkpoint_matches_runtime"] is False
    assert any("checkpoint obs_size=22 does not match current runtime obs_size=39" in error for error in payload["errors"])
    assert any("checkpoint num_actions=3 does not match current runtime num_actions=5" in error for error in payload["errors"])
    assert any(
        step.startswith("Use a checkpoint trained for this symbol set")
        and "checkpoint: 1 symbols, runtime: 2" in step
        for step in payload["next_steps"]
    )


def test_main_once_reports_checkpoint_compatibility_details_on_shape_mismatch(
    capsys, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, symbol_count=1)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )
    (data_dir / "BTCUSD.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--once",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
            "BTCUSD",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "Configuration not ready:" in captured.err
    assert "Checkpoint compatibility:" in captured.err
    assert "- architecture: mlp" in captured.err
    assert "- hidden size: 1024" in captured.err
    assert "- obs size: checkpoint 22, runtime 39" in captured.err
    assert "- action count: checkpoint 3, runtime 5" in captured.err
    assert "Check config: python -u trade_mixed_daily_prod.py --check-config" in captured.err
    assert "Use a checkpoint trained for this symbol set" in captured.err
    assert "checkpoint: 1 symbols, runtime: 2" in captured.err


def test_format_runtime_preflight_failure_reports_checkpoint_inspection_error(tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    config = prod.MixedDailyRuntimeConfig(
        checkpoint=str(tmp_path / "broken.pt"),
        checkpoint_exists=True,
        checkpoint_arch=None,
        checkpoint_hidden_size=None,
        checkpoint_obs_size=None,
        checkpoint_num_actions=None,
        checkpoint_matches_runtime=None,
        checkpoint_inspection_error="Checkpoint load failed: bad checkpoint payload",
        expected_obs_size=22,
        expected_num_actions=3,
        data_dir=str(data_dir),
        data_dir_exists=True,
        symbols_file=None,
        symbol_source="cli",
        symbols=["AAPL"],
        symbol_count=1,
        stock_symbol_count=1,
        crypto_symbol_count=0,
        usable_symbols=["AAPL"],
        usable_symbol_count=1,
        missing_symbol_data=[],
        invalid_symbol_data={},
        latest_local_data_date="2026-04-01",
        oldest_local_data_date="2026-04-01",
        stale_symbol_data={},
        stale_symbol_count=0,
        symbol_details={
            "AAPL": prod.MixedDailySymbolDetail(
                asset_class="stock",
                status="usable",
                local_data_date="2026-04-01",
                reason=None,
            )
        },
        removed_duplicate_symbols=[],
        ignored_symbol_inputs=[],
        hidden_size=256,
        requested_hidden_size=256,
        allocation_pct=10.0,
        lookback_days=45,
        warmup_buffer_days=12,
        daemon_hour_utc=0,
        daemon_minute_utc=5,
        max_episode_steps=123,
        fill_slippage_bps=7.5,
        fee_rate=0.001,
        max_leverage=1.25,
        periods_per_year=252.0,
        action_max_offset_bps=4.5,
        allow_unsafe_checkpoint_loading=False,
        once=True,
        daemon=False,
        dry_run=False,
        check_config=False,
        print_config=False,
        configured_run_mode="once",
        suggested_run_mode="once",
        daemon_schedule_utc="00:05 UTC",
        summary="not ready: once config for 1 symbol",
        check_command_preview="python -u trade_mixed_daily_prod.py --check-config",
        run_command_preview="python -u trade_mixed_daily_prod.py --once",
        safe_command_preview="python -u trade_mixed_daily_prod.py --once --dry-run",
        next_steps=["Re-check setup after fixes: python -u trade_mixed_daily_prod.py --check-config"],
        ready=False,
        errors=["Checkpoint load failed: bad checkpoint payload"],
        warnings=[],
    )

    rendered = prod._format_runtime_preflight_failure(config)

    assert "Configuration not ready:" in rendered
    assert "Checkpoint compatibility:" in rendered
    assert "- inspection error: Checkpoint load failed: bad checkpoint payload" in rendered
    assert "Local data health:" in rendered
    assert "Check config: python -u trade_mixed_daily_prod.py --check-config" in rendered


def test_main_check_config_rejects_unsupported_checkpoint_architecture(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, arch="resmlp")
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["checkpoint_arch"] == "resmlp"
    assert payload["checkpoint_matches_runtime"] is False
    assert any("checkpoint architecture 'resmlp' is not supported" in error for error in payload["errors"])
    assert any(
        step.startswith("Use a supported mixed-daily checkpoint architecture")
        and "'mlp'" in step
        and "'resmlp'" in step
        for step in payload["next_steps"]
    )


def test_main_check_config_suggests_replacing_unreadable_checkpoint(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "broken.pt"
    checkpoint.write_text("not a torch checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["checkpoint_inspection_error"] is not None
    assert any(
        step == f"Replace or repair the checkpoint file: {checkpoint}"
        for step in payload["next_steps"]
    )


def test_main_check_config_reports_invalid_symbol_data(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close\n2026-04-01,1,1,1,1\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "aapl",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["usable_symbols"] == []
    assert payload["usable_symbol_count"] == 0
    assert payload["missing_symbol_data"] == []
    assert "AAPL" in payload["invalid_symbol_data"]
    assert payload["latest_local_data_date"] is None
    assert payload["oldest_local_data_date"] is None
    assert payload["stale_symbol_data"] == {}
    assert payload["stale_symbol_count"] == 0
    assert payload["symbol_details"] == {
        "AAPL": {
            "asset_class": "stock",
            "local_data_date": None,
            "reason": payload["invalid_symbol_data"]["AAPL"],
            "status": "invalid",
        }
    }
    assert "missing required columns: volume" in payload["invalid_symbol_data"]["AAPL"]
    assert any("found unreadable local CSV data for 1 symbols" in warning for warning in payload["warnings"])
    assert any("no usable local CSV data found for any requested symbol" in error for error in payload["errors"])


def test_main_print_config_reports_explicit_runtime_overrides(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--print-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "aapl",
            "--lookback-days",
            "45",
            "--warmup-buffer-days",
            "12",
            "--daemon-hour-utc",
            "2",
            "--daemon-minute-utc",
            "17",
            "--max-episode-steps",
            "123",
            "--fill-slippage-bps",
            "7.5",
            "--fee-rate",
            "0.001",
            "--max-leverage",
            "1.25",
            "--periods-per-year",
            "252.0",
            "--action-max-offset-bps",
            "4.5",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["lookback_days"] == 45
    assert payload["warmup_buffer_days"] == 12
    assert payload["daemon_hour_utc"] == 2
    assert payload["daemon_minute_utc"] == 17
    assert payload["daemon_schedule_utc"] == "02:17 UTC"
    assert payload["max_episode_steps"] == 123
    assert payload["fill_slippage_bps"] == 7.5
    assert payload["fee_rate"] == 0.001
    assert payload["max_leverage"] == 1.25
    assert payload["periods_per_year"] == 252.0
    assert payload["action_max_offset_bps"] == 4.5


def test_main_print_config_uses_checkpoint_hidden_size_for_runtime(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, hidden_size=256)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--print-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
            "--hidden-size",
            "512",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["requested_hidden_size"] == 512
    assert payload["hidden_size"] == 256
    assert payload["checkpoint_hidden_size"] == 256
    assert any("checkpoint hidden_size=256 overrides requested hidden_size=512" in warning for warning in payload["warnings"])
    assert "--hidden-size 256" in payload["check_command_preview"]
    assert "--hidden-size 256" in payload["run_command_preview"]
    assert "--hidden-size 256" in payload["safe_command_preview"]
    assert "--hidden-size 512" not in payload["run_command_preview"]


def test_main_print_config_includes_daemon_preview_commands(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--print-config",
            "--daemon",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "aapl",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["configured_run_mode"] == "daemon"
    assert payload["suggested_run_mode"] == "daemon"
    assert "--daemon" in payload["run_command_preview"]
    assert "--check-config" not in payload["run_command_preview"]
    assert "--once" in payload["safe_command_preview"]
    assert "--dry-run" in payload["safe_command_preview"]


def test_main_print_config_shell_quotes_preview_paths(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "dir with spaces" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data with spaces"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    rc = prod.main(
        [
            "--print-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "aapl",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert f"--checkpoint '{checkpoint}'" in payload["run_command_preview"]
    assert f"--data-dir '{data_dir}'" in payload["run_command_preview"]


def test_main_print_config_loads_symbols_from_file(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, symbol_count=2)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )
    (data_dir / "BTCUSD.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("aapl, btcusd\n# comment\nAAPL\n", encoding="utf-8")

    rc = prod.main(
        [
            "--print-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols-file",
            str(symbols_file),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["symbol_source"] == "file"
    assert payload["symbols_file"] == str(symbols_file)
    assert payload["symbols"] == ["AAPL", "BTCUSD"]
    assert payload["removed_duplicate_symbols"] == ["AAPL"]
    assert "--symbols-file" in payload["run_command_preview"]
    assert str(symbols_file) in payload["run_command_preview"]
    assert "--symbols-file" in payload["safe_command_preview"]


def test_main_check_config_reports_missing_symbols_file(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rc = prod.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols-file",
            str(tmp_path / "missing-symbols.txt"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["symbol_source"] == "file"
    assert payload["symbols"] == []
    assert any("symbols_file could not be read" in error for error in payload["errors"])
    assert any("at least one non-empty symbol is required" in error for error in payload["errors"])


def test_main_check_config_reports_empty_symbols_file(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("# comment only\n \n", encoding="utf-8")

    rc = prod.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols-file",
            str(symbols_file),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["symbol_source"] == "file"
    assert any(f"No valid symbols found in {symbols_file}" in error for error in payload["errors"])


def test_main_check_config_rejects_oversized_symbols_file(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("A" * (prod.MAX_SYMBOLS_FILE_BYTES + 1), encoding="utf-8")

    rc = prod.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols-file",
            str(symbols_file),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["symbol_source"] == "file"
    assert payload["symbols"] == []
    assert any(
        f"symbols_file exceeds {prod.MAX_SYMBOLS_FILE_BYTES} bytes" in error
        for error in payload["errors"]
    )
    assert any("at least one non-empty symbol is required" in error for error in payload["errors"])


def test_main_check_config_rejects_too_many_symbols_from_cli(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    symbols = [f"S{i:03d}" for i in range(prod.MAX_MIXED_SYMBOL_COUNT + 1)]

    rc = prod.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            *symbols,
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["symbol_source"] == "cli"
    assert payload["symbols"] == []
    assert any(
        "too many symbols after normalization" in error
        and f"{prod.MAX_MIXED_SYMBOL_COUNT + 1} > {prod.MAX_MIXED_SYMBOL_COUNT}" in error
        for error in payload["errors"]
    )
    assert any("at least one non-empty symbol is required" in error for error in payload["errors"])


def test_main_print_config_warns_when_symbols_file_overrides_inline_symbols(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\n", encoding="utf-8")

    rc = prod.main(
        [
            "--print-config",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols-file",
            str(symbols_file),
            "--symbols",
            "MSFT",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["symbols"] == ["AAPL"]
    assert any("ignoring inline symbols because --symbols-file was provided" in warning for warning in payload["warnings"])


def test_export_live_binary_raises_clear_error_for_unusable_local_data(tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close\n2026-04-01,1,1,1,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No usable local CSV data found for requested symbols"):
        prod.export_live_binary(["AAPL"], str(data_dir), str(tmp_path / "out.bin"))


def test_export_live_binary_round_trips_through_read_mktd_context(tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume",
                "2026-04-01,10,11,9,10.5,100",
                "2026-04-02,11,12,10,11.5,110",
                "2026-04-03,12,13,11,12.5,120",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.bin"

    prod.export_live_binary(["AAPL"], str(data_dir), str(output), lookback_days=2, warmup_buffer_days=0)
    context = prod._read_mktd_context(output)

    assert context.num_symbols == 1
    assert context.num_timesteps == 3
    assert context.features_per_sym > 0
    assert context.symbol_names == ["AAPL"]
    np.testing.assert_allclose(
        context.last_prices[0],
        np.array([12.0, 13.0, 11.0, 12.5, 120.0], dtype=np.float32),
    )


def test_main_dry_run_threads_explicit_runtime_tuning(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, hidden_size=256)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    def fake_export(symbols, data_dir_arg, output, lookback_days=0, warmup_buffer_days=0):
        calls["export"] = {
            "symbols": symbols,
            "data_dir": data_dir_arg,
            "output": output,
            "lookback_days": lookback_days,
            "warmup_buffer_days": warmup_buffer_days,
        }
        return (str(tmp_path / "fake.bin"), len(symbols), 1)

    def fake_run(
        checkpoint_arg,
        data_bin,
        hidden_size=0,
        *,
        max_episode_steps=0,
        fill_slippage_bps=0.0,
        fee_rate=0.0,
        max_leverage=0.0,
        periods_per_year=0.0,
        action_max_offset_bps=0.0,
        allow_unsafe_checkpoint_loading=False,
    ):
        calls["run"] = {
            "checkpoint": checkpoint_arg,
            "data_bin": data_bin,
            "hidden_size": hidden_size,
            "max_episode_steps": max_episode_steps,
            "fill_slippage_bps": fill_slippage_bps,
            "fee_rate": fee_rate,
            "max_leverage": max_leverage,
            "periods_per_year": periods_per_year,
            "action_max_offset_bps": action_max_offset_bps,
            "allow_unsafe_checkpoint_loading": allow_unsafe_checkpoint_loading,
        }
        return {
            "direction": "FLAT",
            "symbol": None,
            "confidence": 1.0,
            "value": 0.0,
            "action": 0,
            "timestamp": "2026-04-02T00:00:00+00:00",
            "all_probs": [1.0],
            "sym_names": ["AAPL"],
        }

    monkeypatch.setattr(prod, "export_live_binary", fake_export)
    monkeypatch.setattr(prod, "run_inference", fake_run)

    rc = prod.main(
        [
            "--once",
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "aapl",
            "--lookback-days",
            "45",
            "--warmup-buffer-days",
            "12",
            "--hidden-size",
            "256",
            "--max-episode-steps",
            "123",
            "--fill-slippage-bps",
            "7.5",
            "--fee-rate",
            "0.001",
            "--max-leverage",
            "1.25",
            "--periods-per-year",
            "252.0",
            "--action-max-offset-bps",
            "4.5",
        ]
    )

    assert rc == 0
    assert calls["export"] == {
        "symbols": ["AAPL"],
        "data_dir": str(data_dir),
        "output": calls["export"]["output"],
        "lookback_days": 45,
        "warmup_buffer_days": 12,
    }
    assert Path(calls["export"]["output"]).parent == Path(tempfile.gettempdir())
    assert Path(calls["export"]["output"]).name.startswith(prod.TEMP_BIN_PREFIX)
    assert Path(calls["export"]["output"]).suffix == ".bin"
    assert calls["run"] == {
        "checkpoint": str(checkpoint),
        "data_bin": str(tmp_path / "fake.bin"),
        "hidden_size": 256,
        "max_episode_steps": 123,
        "fill_slippage_bps": 7.5,
        "fee_rate": 0.001,
        "max_leverage": 1.25,
        "periods_per_year": 252.0,
        "action_max_offset_bps": 4.5,
        "allow_unsafe_checkpoint_loading": False,
    }


def test_run_export_and_inference_cycle_threads_config_and_cleans_up(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    temp_path = tmp_path / "temp.bin"

    config = prod.MixedDailyRuntimeConfig(
        checkpoint=str(tmp_path / "best.pt"),
        checkpoint_exists=True,
        checkpoint_arch="mlp",
        checkpoint_hidden_size=256,
        checkpoint_obs_size=22,
        checkpoint_num_actions=3,
        checkpoint_matches_runtime=True,
        checkpoint_inspection_error=None,
        expected_obs_size=22,
        expected_num_actions=3,
        data_dir=str(data_dir),
        data_dir_exists=True,
        symbols_file=None,
        symbol_source="cli",
        symbols=["AAPL"],
        symbol_count=1,
        stock_symbol_count=1,
        crypto_symbol_count=0,
        usable_symbols=["AAPL"],
        usable_symbol_count=1,
        missing_symbol_data=[],
        invalid_symbol_data={},
        latest_local_data_date="2026-04-01",
        oldest_local_data_date="2026-04-01",
        stale_symbol_data={},
        stale_symbol_count=0,
        symbol_details={
            "AAPL": prod.MixedDailySymbolDetail(
                asset_class="stock",
                status="usable",
                local_data_date="2026-04-01",
                reason=None,
            )
        },
        removed_duplicate_symbols=[],
        ignored_symbol_inputs=[],
        hidden_size=256,
        requested_hidden_size=256,
        allocation_pct=10.0,
        lookback_days=45,
        warmup_buffer_days=12,
        daemon_hour_utc=0,
        daemon_minute_utc=5,
        max_episode_steps=123,
        fill_slippage_bps=7.5,
        fee_rate=0.001,
        max_leverage=1.25,
        periods_per_year=252.0,
        action_max_offset_bps=4.5,
        allow_unsafe_checkpoint_loading=False,
        once=True,
        daemon=False,
        dry_run=True,
        check_config=False,
        print_config=False,
        configured_run_mode="dry_run",
        suggested_run_mode="dry_run",
        daemon_schedule_utc="00:05 UTC",
        summary="ready",
        check_command_preview="python -u trade_mixed_daily_prod.py --check-config",
        run_command_preview="python -u trade_mixed_daily_prod.py --once",
        safe_command_preview="python -u trade_mixed_daily_prod.py --once --dry-run",
        next_steps=[],
        ready=True,
        errors=[],
        warnings=[],
    )

    calls: dict[str, object] = {}

    monkeypatch.setattr(prod, "_make_temp_data_bin_path", lambda: temp_path)

    def fake_export(symbols, data_dir_arg, output, lookback_days=0, warmup_buffer_days=0):
        Path(output).write_bytes(b"test")
        calls["export"] = {
            "symbols": symbols,
            "data_dir": data_dir_arg,
            "output": output,
            "lookback_days": lookback_days,
            "warmup_buffer_days": warmup_buffer_days,
        }
        return (output, len(symbols), 1)

    def fake_run(
        checkpoint_arg,
        data_bin,
        hidden_size=0,
        *,
        max_episode_steps=0,
        fill_slippage_bps=0.0,
        fee_rate=0.0,
        max_leverage=0.0,
        periods_per_year=0.0,
        action_max_offset_bps=0.0,
        allow_unsafe_checkpoint_loading=False,
    ):
        calls["run"] = {
            "checkpoint": checkpoint_arg,
            "data_bin": data_bin,
            "hidden_size": hidden_size,
            "max_episode_steps": max_episode_steps,
            "fill_slippage_bps": fill_slippage_bps,
            "fee_rate": fee_rate,
            "max_leverage": max_leverage,
            "periods_per_year": periods_per_year,
            "action_max_offset_bps": action_max_offset_bps,
            "allow_unsafe_checkpoint_loading": allow_unsafe_checkpoint_loading,
        }
        return {
            "direction": "FLAT",
            "symbol": None,
            "confidence": 1.0,
            "value": 0.0,
            "action": 0,
            "timestamp": "2026-04-02T00:00:00+00:00",
            "all_probs": [1.0],
            "sym_names": ["AAPL"],
        }

    monkeypatch.setattr(prod, "export_live_binary", fake_export)
    monkeypatch.setattr(prod, "run_inference", fake_run)

    signal = prod._run_export_and_inference_cycle(config)

    assert signal["direction"] == "FLAT"
    assert calls["export"] == {
        "symbols": ["AAPL"],
        "data_dir": str(data_dir),
        "output": str(temp_path),
        "lookback_days": 45,
        "warmup_buffer_days": 12,
    }
    assert calls["run"] == {
        "checkpoint": str(tmp_path / "best.pt"),
        "data_bin": str(temp_path),
        "hidden_size": 256,
        "max_episode_steps": 123,
        "fill_slippage_bps": 7.5,
        "fee_rate": 0.001,
        "max_leverage": 1.25,
        "periods_per_year": 252.0,
        "action_max_offset_bps": 4.5,
        "allow_unsafe_checkpoint_loading": False,
    }
    assert temp_path.exists() is False


def test_run_export_and_inference_cycle_cleans_up_temp_bin_after_export_failure(
    monkeypatch, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    temp_path = tmp_path / "temp.bin"

    config = prod.MixedDailyRuntimeConfig(
        checkpoint=str(tmp_path / "best.pt"),
        checkpoint_exists=True,
        checkpoint_arch="mlp",
        checkpoint_hidden_size=256,
        checkpoint_obs_size=22,
        checkpoint_num_actions=3,
        checkpoint_matches_runtime=True,
        checkpoint_inspection_error=None,
        expected_obs_size=22,
        expected_num_actions=3,
        data_dir=str(data_dir),
        data_dir_exists=True,
        symbols_file=None,
        symbol_source="cli",
        symbols=["AAPL"],
        symbol_count=1,
        stock_symbol_count=1,
        crypto_symbol_count=0,
        usable_symbols=["AAPL"],
        usable_symbol_count=1,
        missing_symbol_data=[],
        invalid_symbol_data={},
        latest_local_data_date="2026-04-01",
        oldest_local_data_date="2026-04-01",
        stale_symbol_data={},
        stale_symbol_count=0,
        symbol_details={
            "AAPL": prod.MixedDailySymbolDetail(
                asset_class="stock",
                status="usable",
                local_data_date="2026-04-01",
                reason=None,
            )
        },
        removed_duplicate_symbols=[],
        ignored_symbol_inputs=[],
        hidden_size=256,
        requested_hidden_size=256,
        allocation_pct=10.0,
        lookback_days=45,
        warmup_buffer_days=12,
        daemon_hour_utc=0,
        daemon_minute_utc=5,
        max_episode_steps=123,
        fill_slippage_bps=7.5,
        fee_rate=0.001,
        max_leverage=1.25,
        periods_per_year=252.0,
        action_max_offset_bps=4.5,
        allow_unsafe_checkpoint_loading=False,
        once=True,
        daemon=False,
        dry_run=True,
        check_config=False,
        print_config=False,
        configured_run_mode="dry_run",
        suggested_run_mode="dry_run",
        daemon_schedule_utc="00:05 UTC",
        summary="ready",
        check_command_preview="python -u trade_mixed_daily_prod.py --check-config",
        run_command_preview="python -u trade_mixed_daily_prod.py --once",
        safe_command_preview="python -u trade_mixed_daily_prod.py --once --dry-run",
        next_steps=[],
        ready=True,
        errors=[],
        warnings=[],
    )

    monkeypatch.setattr(prod, "_make_temp_data_bin_path", lambda: temp_path)

    def fake_export(symbols, data_dir_arg, output, lookback_days=0, warmup_buffer_days=0):
        del symbols, data_dir_arg, lookback_days, warmup_buffer_days
        Path(output).write_bytes(b"test")
        raise ValueError("broken export")

    monkeypatch.setattr(prod, "export_live_binary", fake_export)

    with pytest.raises(ValueError, match="broken export"):
        prod._run_export_and_inference_cycle(config)

    assert temp_path.exists() is False


def test_run_export_and_inference_cycle_preserves_export_error_when_cleanup_fails(
    monkeypatch, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    temp_path = tmp_path / "temp.bin"

    config = prod.MixedDailyRuntimeConfig(
        checkpoint=str(tmp_path / "best.pt"),
        checkpoint_exists=True,
        checkpoint_arch="mlp",
        checkpoint_hidden_size=256,
        checkpoint_obs_size=22,
        checkpoint_num_actions=3,
        checkpoint_matches_runtime=True,
        checkpoint_inspection_error=None,
        expected_obs_size=22,
        expected_num_actions=3,
        data_dir=str(data_dir),
        data_dir_exists=True,
        symbols_file=None,
        symbol_source="cli",
        symbols=["AAPL"],
        symbol_count=1,
        stock_symbol_count=1,
        crypto_symbol_count=0,
        usable_symbols=["AAPL"],
        usable_symbol_count=1,
        missing_symbol_data=[],
        invalid_symbol_data={},
        latest_local_data_date="2026-04-01",
        oldest_local_data_date="2026-04-01",
        stale_symbol_data={},
        stale_symbol_count=0,
        symbol_details={
            "AAPL": prod.MixedDailySymbolDetail(
                asset_class="stock",
                status="usable",
                local_data_date="2026-04-01",
                reason=None,
            )
        },
        removed_duplicate_symbols=[],
        ignored_symbol_inputs=[],
        hidden_size=256,
        requested_hidden_size=256,
        allocation_pct=10.0,
        lookback_days=45,
        warmup_buffer_days=12,
        daemon_hour_utc=0,
        daemon_minute_utc=5,
        max_episode_steps=123,
        fill_slippage_bps=7.5,
        fee_rate=0.001,
        max_leverage=1.25,
        periods_per_year=252.0,
        action_max_offset_bps=4.5,
        allow_unsafe_checkpoint_loading=False,
        once=True,
        daemon=False,
        dry_run=True,
        check_config=False,
        print_config=False,
        configured_run_mode="dry_run",
        suggested_run_mode="dry_run",
        daemon_schedule_utc="00:05 UTC",
        summary="ready",
        check_command_preview="python -u trade_mixed_daily_prod.py --check-config",
        run_command_preview="python -u trade_mixed_daily_prod.py --once",
        safe_command_preview="python -u trade_mixed_daily_prod.py --once --dry-run",
        next_steps=[],
        ready=True,
        errors=[],
        warnings=[],
    )

    monkeypatch.setattr(prod, "_make_temp_data_bin_path", lambda: temp_path)

    def fake_export(symbols, data_dir_arg, output, lookback_days=0, warmup_buffer_days=0):
        del symbols, data_dir_arg, lookback_days, warmup_buffer_days
        Path(output).write_bytes(b"test")
        raise ValueError("broken export")

    monkeypatch.setattr(prod, "export_live_binary", fake_export)
    monkeypatch.setattr(prod, "_cleanup_temp_data_bin", lambda _path: (_ for _ in ()).throw(OSError("busy file")))

    with pytest.raises(ValueError, match="broken export") as excinfo:
        prod._run_export_and_inference_cycle(config)

    assert any(
        "Temporary data bin cleanup failed" in note and "busy file" in note
        for note in getattr(excinfo.value, "__notes__", [])
    )


def test_run_export_and_inference_cycle_warns_when_cleanup_fails_after_success(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import trade_mixed_daily_prod as prod

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    temp_path = tmp_path / "temp.bin"

    config = prod.MixedDailyRuntimeConfig(
        checkpoint=str(tmp_path / "best.pt"),
        checkpoint_exists=True,
        checkpoint_arch="mlp",
        checkpoint_hidden_size=256,
        checkpoint_obs_size=22,
        checkpoint_num_actions=3,
        checkpoint_matches_runtime=True,
        checkpoint_inspection_error=None,
        expected_obs_size=22,
        expected_num_actions=3,
        data_dir=str(data_dir),
        data_dir_exists=True,
        symbols_file=None,
        symbol_source="cli",
        symbols=["AAPL"],
        symbol_count=1,
        stock_symbol_count=1,
        crypto_symbol_count=0,
        usable_symbols=["AAPL"],
        usable_symbol_count=1,
        missing_symbol_data=[],
        invalid_symbol_data={},
        latest_local_data_date="2026-04-01",
        oldest_local_data_date="2026-04-01",
        stale_symbol_data={},
        stale_symbol_count=0,
        symbol_details={
            "AAPL": prod.MixedDailySymbolDetail(
                asset_class="stock",
                status="usable",
                local_data_date="2026-04-01",
                reason=None,
            )
        },
        removed_duplicate_symbols=[],
        ignored_symbol_inputs=[],
        hidden_size=256,
        requested_hidden_size=256,
        allocation_pct=10.0,
        lookback_days=45,
        warmup_buffer_days=12,
        daemon_hour_utc=0,
        daemon_minute_utc=5,
        max_episode_steps=123,
        fill_slippage_bps=7.5,
        fee_rate=0.001,
        max_leverage=1.25,
        periods_per_year=252.0,
        action_max_offset_bps=4.5,
        allow_unsafe_checkpoint_loading=False,
        once=True,
        daemon=False,
        dry_run=True,
        check_config=False,
        print_config=False,
        configured_run_mode="dry_run",
        suggested_run_mode="dry_run",
        daemon_schedule_utc="00:05 UTC",
        summary="ready",
        check_command_preview="python -u trade_mixed_daily_prod.py --check-config",
        run_command_preview="python -u trade_mixed_daily_prod.py --once",
        safe_command_preview="python -u trade_mixed_daily_prod.py --once --dry-run",
        next_steps=[],
        ready=True,
        errors=[],
        warnings=[],
    )

    monkeypatch.setattr(prod, "_make_temp_data_bin_path", lambda: temp_path)

    def fake_export(symbols, data_dir_arg, output, lookback_days=0, warmup_buffer_days=0):
        del symbols, data_dir_arg, lookback_days, warmup_buffer_days
        Path(output).write_bytes(b"test")
        return (output, 1, 1)

    monkeypatch.setattr(prod, "export_live_binary", fake_export)
    monkeypatch.setattr(
        prod,
        "run_inference",
        lambda *args, **kwargs: {
            "direction": "FLAT",
            "symbol": None,
            "confidence": 1.0,
            "value": 0.0,
            "action": 0,
            "timestamp": "2026-04-02T00:00:00+00:00",
            "all_probs": [1.0],
            "sym_names": ["AAPL"],
        },
    )
    monkeypatch.setattr(prod, "_cleanup_temp_data_bin", lambda _path: (_ for _ in ()).throw(OSError("busy file")))

    signal = prod._run_export_and_inference_cycle(config)

    captured = capsys.readouterr()
    assert signal["direction"] == "FLAT"
    assert "WARNING: Temporary data bin cleanup failed" in captured.err
    assert "busy file" in captured.err


def test_main_dry_run_cleans_up_temp_bin_after_success(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    export_outputs: list[Path] = []

    def fake_export(symbols, data_dir_arg, output, lookback_days=0, warmup_buffer_days=0):
        del symbols, data_dir_arg, lookback_days, warmup_buffer_days
        output_path = Path(output)
        output_path.write_bytes(b"test")
        export_outputs.append(output_path)
        return (str(output_path), 1, 1)

    monkeypatch.setattr(prod, "export_live_binary", fake_export)
    monkeypatch.setattr(
        prod,
        "run_inference",
        lambda *args, **kwargs: {
            "direction": "FLAT",
            "symbol": None,
            "confidence": 1.0,
            "value": 0.0,
            "action": 0,
            "timestamp": "2026-04-02T00:00:00+00:00",
            "all_probs": [1.0],
            "sym_names": ["AAPL"],
        },
    )

    rc = prod.main(
        [
            "--once",
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
        ]
    )

    assert rc == 0
    assert len(export_outputs) == 1
    assert export_outputs[0].exists() is False


def test_main_dry_run_cleans_up_temp_bin_after_inference_error(monkeypatch, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    export_outputs: list[Path] = []

    def fake_export(symbols, data_dir_arg, output, lookback_days=0, warmup_buffer_days=0):
        del symbols, data_dir_arg, lookback_days, warmup_buffer_days
        output_path = Path(output)
        output_path.write_bytes(b"test")
        export_outputs.append(output_path)
        return (str(output_path), 1, 1)

    monkeypatch.setattr(prod, "export_live_binary", fake_export)

    def fail_run(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(prod, "run_inference", fail_run)

    rc = prod.main(
        [
            "--once",
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
        ]
    )

    assert rc == 1
    assert len(export_outputs) == 1
    assert export_outputs[0].exists() is False


def test_main_dry_run_emits_runtime_context_event(monkeypatch, capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(prod, "export_live_binary", lambda *args, **kwargs: (str(tmp_path / "fake.bin"), 1, 1))
    monkeypatch.setattr(
        prod,
        "run_inference",
        lambda *args, **kwargs: {
            "direction": "FLAT",
            "symbol": None,
            "confidence": 1.0,
            "value": 0.0,
            "action": 0,
            "timestamp": "2026-04-02T00:00:00+00:00",
            "all_probs": [1.0],
            "sym_names": ["AAPL"],
        },
    )

    rc = prod.main(
        [
            "--once",
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
        ]
    )

    payloads = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert rc == 0
    assert payloads[0]["event"] == "mixed_daily_runtime"
    assert payloads[0]["mode"] == "dry_run"
    assert payloads[0]["stage"] == "startup"
    assert payloads[0]["symbol_count"] == 1
    assert payloads[0]["usable_symbol_count"] == 1
    assert payloads[0]["allow_unsafe_checkpoint_loading"] is False
    assert payloads[0]["checkpoint"] == str(checkpoint)
    assert payloads[0]["checkpoint_arch"] == "mlp"
    assert payloads[0]["checkpoint_hidden_size"] == 1024
    assert payloads[0]["checkpoint_obs_size"] == 22
    assert payloads[0]["checkpoint_num_actions"] == 3
    assert payloads[0]["checkpoint_matches_runtime"] is True
    assert payloads[0]["expected_obs_size"] == 22
    assert payloads[0]["expected_num_actions"] == 3
    assert payloads[0]["data_dir"] == str(data_dir)


def test_main_once_emits_signal_and_trade_execution_events(monkeypatch, capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, symbol_count=2)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for symbol in ("AAPL", "BTCUSD"):
        (data_dir / f"{symbol}.csv").write_text(
            "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        prod,
        "_run_export_and_inference_cycle",
        lambda _config: {
            "direction": "SHORT",
            "symbol": "BTCUSD",
            "confidence": 0.8,
            "value": 1.25,
            "action": 4,
            "timestamp": "2026-04-03T00:00:00+00:00",
            "all_probs": [0.01, 0.6, 0.05, 0.1, 0.8],
            "sym_names": ["AAPL", "BTCUSD"],
        },
    )

    results = iter(
        [
            prod.TradeExecutionResult(
                submitted=False,
                status="skipped",
                reason="crypto_short_without_position",
                symbol="BTCUSD",
                alpaca_symbol="BTC/USD",
                direction="SHORT",
                confidence=0.8,
                is_crypto=True,
                trade_value=900.0,
                price=50_000.0,
                qty=0.018,
            ),
            prod.TradeExecutionResult(
                submitted=True,
                status="submitted",
                reason="order_submitted",
                symbol="AAPL",
                alpaca_symbol="AAPL",
                direction="LONG",
                confidence=0.6,
                is_crypto=False,
                trade_value=900.0,
                price=150.0,
                qty=6.0,
                limit_price=149.5,
                order_id="ord-2",
                order_status="accepted",
            ),
        ]
    )
    monkeypatch.setattr(prod, "execute_signal_result", lambda *_args, **_kwargs: next(results))

    rc = prod.main(
        [
            "--once",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
            "BTCUSD",
        ]
    )

    payloads = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    signal_payload = next(payload for payload in payloads if payload["event"] == "mixed_daily_signal")
    trade_payloads = [payload for payload in payloads if payload["event"] == "mixed_daily_trade_execution"]

    assert rc == 0
    assert signal_payload["signal_direction"] == "SHORT"
    assert signal_payload["signal_symbol"] == "BTCUSD"
    assert signal_payload["signal_action"] == 4
    assert len(trade_payloads) == 2
    assert trade_payloads[0]["attempt_kind"] == "primary"
    assert trade_payloads[0]["attempt_index"] == 0
    assert trade_payloads[0]["execution_submitted"] is False
    assert trade_payloads[0]["execution_reason"] == "crypto_short_without_position"
    assert trade_payloads[1]["attempt_kind"] == "alternative"
    assert trade_payloads[1]["attempt_index"] == 1
    assert trade_payloads[1]["signal_symbol"] == "AAPL"
    assert trade_payloads[1]["execution_submitted"] is True
    assert trade_payloads[1]["broker_order_id"] == "ord-2"


def test_main_once_returns_clean_error_after_trade_execution_failure(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, symbol_count=1)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        prod,
        "_run_export_and_inference_cycle",
        lambda _config: {
            "direction": "LONG",
            "symbol": "AAPL",
            "confidence": 0.8,
            "value": 1.25,
            "action": 1,
            "timestamp": "2026-04-03T00:00:00+00:00",
            "all_probs": [0.2, 0.8, 0.0],
            "sym_names": ["AAPL"],
        },
    )
    monkeypatch.setattr(
        prod,
        "execute_signal_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broker boom")),
    )

    rc = prod.main(
        [
            "--once",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
        ]
    )

    captured = capsys.readouterr()
    payloads = [
        json.loads(line)
        for line in captured.out.splitlines()
        if line.startswith("{")
    ]

    assert rc == 1
    assert payloads[0]["event"] == "mixed_daily_runtime"
    assert payloads[1]["event"] == "mixed_daily_signal"
    assert payloads[2]["event"] == "mixed_daily_runtime_error"
    assert payloads[2]["stage"] == "trade_execution"
    assert payloads[2]["error_type"] == "RuntimeError"
    assert payloads[2]["error"] == "broker boom"
    assert "Mixed daily run failed during trade execution." in captured.err
    assert "Error: RuntimeError: broker boom" in captured.err


def test_main_dry_run_emits_runtime_error_event(monkeypatch, capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    def fail_export(*args, **kwargs):
        raise ValueError("broken export")

    monkeypatch.setattr(prod, "export_live_binary", fail_export)

    rc = prod.main(
        [
            "--once",
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
        ]
    )

    captured = capsys.readouterr()
    payloads = [
        json.loads(line)
        for line in captured.out.splitlines()
        if line.startswith("{")
    ]
    assert rc == 1
    assert payloads[0]["event"] == "mixed_daily_runtime"
    assert payloads[1]["event"] == "mixed_daily_runtime_error"
    assert payloads[1]["mode"] == "dry_run"
    assert payloads[1]["stage"] == "export_or_inference"
    assert payloads[1]["error_type"] == "ValueError"
    assert payloads[1]["error"] == "broken export"
    assert "Mixed daily run failed during export or inference." in captured.err
    assert "Error: ValueError: broken export" in captured.err
    assert "Check config: python -u trade_mixed_daily_prod.py --check-config" in captured.err


def test_main_dry_run_reports_cleanup_notes_in_runtime_error(monkeypatch, capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-01,1,1,1,1,100\n",
        encoding="utf-8",
    )

    def fail_export(*args, **kwargs):
        output = Path(kwargs.get("output") or args[2])
        output.write_bytes(b"test")
        raise ValueError("broken export")

    monkeypatch.setattr(prod, "export_live_binary", fail_export)
    monkeypatch.setattr(prod, "_cleanup_temp_data_bin", lambda _path: (_ for _ in ()).throw(OSError("busy file")))

    rc = prod.main(
        [
            "--once",
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL",
        ]
    )

    captured = capsys.readouterr()
    payloads = [
        json.loads(line)
        for line in captured.out.splitlines()
        if line.startswith("{")
    ]
    assert rc == 1
    assert payloads[1]["event"] == "mixed_daily_runtime_error"
    assert payloads[1]["stage"] == "export_or_inference"
    assert payloads[1]["error"] == "broken export"
    assert len(payloads[1]["error_notes"]) == 1
    assert "Temporary data bin cleanup failed for " in payloads[1]["error_notes"][0]
    assert "busy file" in payloads[1]["error_notes"][0]
    assert "Additional context:" in captured.err
    assert "Temporary data bin cleanup failed" in captured.err
    assert "busy file" in captured.err


def test_main_print_config_rejects_invalid_runtime_tuning(capsys, tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    rc = prod.main(
        [
            "--print-config",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--data-dir",
            str(tmp_path / "missing-data"),
            "--symbols",
            "aapl",
            "--lookback-days",
            "0",
            "--warmup-buffer-days",
            "-1",
            "--daemon-hour-utc",
            "24",
            "--daemon-minute-utc",
            "60",
            "--max-episode-steps",
            "0",
            "--fill-slippage-bps",
            "-0.5",
            "--fee-rate",
            "-0.1",
            "--max-leverage",
            "0",
            "--periods-per-year",
            "0",
            "--action-max-offset-bps",
            "-1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert any("lookback_days must be > 0" in error for error in payload["errors"])
    assert any("warmup_buffer_days must be >= 0" in error for error in payload["errors"])
    assert any("daemon_hour_utc must be between 0 and 23" in error for error in payload["errors"])
    assert any("daemon_minute_utc must be between 0 and 59" in error for error in payload["errors"])
    assert any("max_episode_steps must be > 0" in error for error in payload["errors"])
    assert any("fill_slippage_bps must be >= 0" in error for error in payload["errors"])
    assert any("fee_rate must be >= 0" in error for error in payload["errors"])
    assert any("max_leverage must be > 0" in error for error in payload["errors"])
    assert any("periods_per_year must be > 0" in error for error in payload["errors"])
    assert any("action_max_offset_bps must be >= 0" in error for error in payload["errors"])


def test_resolve_symbol_data_path_rejects_unsafe_symbol_input(tmp_path: Path) -> None:
    import trade_mixed_daily_prod as prod

    with pytest.raises(ValueError, match=r"Unsupported symbol: \.\./AAPL"):
        prod._resolve_symbol_data_path(tmp_path, "../AAPL")


def test_execute_signal_rejects_unsafe_symbol_before_alpaca_import() -> None:
    import trade_mixed_daily_prod as prod

    with pytest.raises(ValueError, match=r"Unsupported symbol: \.\./AAPL"):
        prod.execute_signal(
            {"symbol": "../AAPL", "direction": "LONG", "confidence": 0.9},
            allocation_pct=10.0,
        )
