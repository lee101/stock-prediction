from __future__ import annotations

import sys
import types
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict

import pytest

from fal_marketsimulator import runner


class _Context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _TorchStub:
    def __init__(self) -> None:
        self.cuda = types.SimpleNamespace(is_available=lambda: False)

    def inference_mode(self) -> _Context:
        return _Context()

    def no_grad(self) -> _Context:
        return _Context()


class _DummyController:
    def __init__(self, initial_cash: float) -> None:
        self._time = datetime(2024, 1, 1, 9, 30)
        self._minutes_per_step = 5
        self._cash = float(initial_cash)
        self._equity = self._cash + 250.0
        self.positions: Dict[str, int] = {"AAPL": 1}

    def advance_steps(self, steps: int = 1) -> datetime:
        steps = max(1, int(steps))
        self._time += timedelta(minutes=self._minutes_per_step * steps)
        return self._time

    def current_time(self) -> datetime:
        return self._time

    def summary(self) -> Dict[str, Any]:
        return {
            "cash": self._cash,
            "equity": self._equity,
            "positions": self.positions,
        }


def test_setup_training_imports_registers_modules(monkeypatch):
    original_torch = runner.torch
    original_np = runner.np
    original_pd = getattr(runner, "pd", None)

    called = {}

    def fake_setup_src_imports(torch_module, numpy_module, pandas_module=None):
        called["torch"] = torch_module
        called["numpy"] = numpy_module
        called["pandas"] = pandas_module

    monkeypatch.setattr(runner, "setup_src_imports", fake_setup_src_imports)

    torch_stub = types.SimpleNamespace(marker="torch")
    numpy_stub = types.SimpleNamespace(marker="numpy")
    pandas_stub = types.SimpleNamespace(marker="pandas")

    try:
        runner.setup_training_imports(torch_stub, numpy_stub, pandas_module=pandas_stub)
    finally:
        runner.torch = original_torch
        runner.np = original_np
        runner.pd = original_pd

    assert called["torch"] is torch_stub
    assert called["numpy"] is numpy_stub
    assert called["pandas"] is pandas_stub


def test_simulate_trading_with_stubbed_environment(monkeypatch):
    torch_stub = _TorchStub()
    numpy_stub = types.SimpleNamespace()
    monkeypatch.setattr(runner, "torch", torch_stub, raising=False)
    monkeypatch.setattr(runner, "np", numpy_stub, raising=False)

    trade_module = types.SimpleNamespace()
    trade_module.logged = []
    trade_module.manage_calls = []
    trade_module.released = False
    call_counter = {"count": 0}

    def analyze_symbols(symbols):
        call_counter["count"] += 1
        ordered = OrderedDict()
        for idx, symbol in enumerate(symbols):
            ordered[symbol] = {
                "avg_return": 0.05 * (idx + 1),
                "expected_profit": 10.0 * (call_counter["count"] + idx),
                "predicted_return": 0.02 * (idx + 1),
            }
        return ordered

    def log_trading_plan(picks, label):
        trade_module.logged.append((label, picks))

    def manage_positions(current, previous, analyzed):
        trade_module.manage_calls.append({"current": current, "previous": previous, "analyzed": analyzed})

    def release_model_resources():
        trade_module.released = True

    trade_module.analyze_symbols = analyze_symbols
    trade_module.log_trading_plan = log_trading_plan
    trade_module.manage_positions = manage_positions
    trade_module.release_model_resources = release_model_resources

    monkeypatch.setitem(sys.modules, "trade_stock_e2e", trade_module)

    def fake_activate_simulation(**kwargs):
        controller = _DummyController(kwargs["initial_cash"])

        class _ControllerCtx:
            def __enter__(self_inner):
                return controller

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _ControllerCtx()

    monkeypatch.setattr("marketsimulator.environment.activate_simulation", fake_activate_simulation, raising=False)

    result = runner.simulate_trading(
        symbols=["AAPL", "MSFT", "GOOG"],
        steps=3,
        step_size=2,
        initial_cash=1_000.0,
        top_k=2,
        kronos_only=False,
        compact_logs=True,
    )

    assert len(result["timeline"]) == 3
    assert all(entry["picked"] for entry in result["timeline"])
    assert result["summary"]["cash"] == pytest.approx(1_000.0)
    assert result["summary"]["equity"] == pytest.approx(1_250.0)
    assert trade_module.released is True
    assert len(trade_module.manage_calls) == 3
    assert trade_module.manage_calls[0]["previous"] == {}
    assert trade_module.manage_calls[1]["previous"] == trade_module.manage_calls[0]["current"]
    assert any(label == "SIM-STEP-1" for label, _ in trade_module.logged)
