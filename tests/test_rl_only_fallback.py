"""Tests for RL-only fallback in hybrid trading when Gemini is unavailable."""
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-trading-agent-binance"))

from hybrid_prompt import AllocationPlan, SymbolContext


@dataclass
class FakeRLSignal:
    action: int = 1
    action_name: str = "LONG_BTC"
    target_symbol: Optional[str] = "BTCUSD"
    direction: str = "long"
    logits: list = field(default_factory=lambda: [0.1, 2.0, 0.5, 0.3, 0.2])
    value: float = 0.5


@dataclass
class FakeRLGen:
    symbols: tuple = ("BTCUSD", "ETHUSD")
    num_symbols: int = 2
    action_names: list = field(default_factory=lambda: ["FLAT", "LONG_BTC", "LONG_ETH"])


def _make_ctx(symbol: str, price: float) -> SymbolContext:
    idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    klines = pd.DataFrame({
        "open": price, "high": price * 1.01, "low": price * 0.99,
        "close": price, "volume": 1000.0,
    }, index=idx)
    return SymbolContext(symbol=symbol, price=price, klines=klines)


from trade_binance_live import _rl_signal_to_allocation_plan


def test_flat_signal_returns_none():
    sig = FakeRLSignal(action=0, action_name="FLAT", target_symbol=None, direction="flat")
    gen = FakeRLGen()
    ctxs = [_make_ctx("BTCUSD", 85000), _make_ctx("ETHUSD", 2000)]
    result = _rl_signal_to_allocation_plan(sig, ctxs, gen, 1.0)
    assert result is None


def test_long_signal_creates_plan():
    sig = FakeRLSignal()
    gen = FakeRLGen()
    ctxs = [_make_ctx("BTCUSD", 85000), _make_ctx("ETHUSD", 2000)]
    result = _rl_signal_to_allocation_plan(sig, ctxs, gen, 1.0)
    assert result is not None
    assert "BTCUSD" in result.allocations
    assert result.allocations["BTCUSD"] >= 15.0
    assert result.allocations["BTCUSD"] <= 50.0
    assert result.entry_prices["BTCUSD"] < 85000
    assert result.exit_prices["BTCUSD"] > 85000
    assert "rl_only_fallback" in result.reasoning


def test_unknown_symbol_returns_none():
    sig = FakeRLSignal(target_symbol="FOOUSD")
    gen = FakeRLGen()
    ctxs = [_make_ctx("BTCUSD", 85000)]
    result = _rl_signal_to_allocation_plan(sig, ctxs, gen, 1.0)
    assert result is None


def test_alloc_pct_bounds():
    logits_high_confidence = [0.0, 10.0, 0.0, 0.0, 0.0]
    sig = FakeRLSignal(logits=logits_high_confidence)
    gen = FakeRLGen()
    ctxs = [_make_ctx("BTCUSD", 85000)]
    result = _rl_signal_to_allocation_plan(sig, ctxs, gen, 1.0)
    assert result is not None
    assert result.allocations["BTCUSD"] <= 50.0

    logits_low_confidence = [5.0, 0.1, 0.0, 0.0, 0.0]
    sig2 = FakeRLSignal(logits=logits_low_confidence)
    result2 = _rl_signal_to_allocation_plan(sig2, ctxs, gen, 1.0)
    assert result2 is not None
    assert result2.allocations["BTCUSD"] >= 15.0


def test_entry_exit_prices():
    price = 85000.0
    sig = FakeRLSignal()
    gen = FakeRLGen()
    ctxs = [_make_ctx("BTCUSD", price)]
    result = _rl_signal_to_allocation_plan(sig, ctxs, gen, 1.0)
    assert result is not None
    assert abs(result.entry_prices["BTCUSD"] - price * 0.999) < 1.0
    assert abs(result.exit_prices["BTCUSD"] - price * 1.008) < 1.0
