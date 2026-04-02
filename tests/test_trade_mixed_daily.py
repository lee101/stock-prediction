from __future__ import annotations

import argparse
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from trade_mixed_daily import _resolve_symbols


def _write_mktd(path: Path, symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_symbols = len(symbols)
    num_timesteps = 2
    features_per_sym = 16
    price_features = 5

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
