from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from types import ModuleType, SimpleNamespace

import pytest
import fal_marketsimulator.runner as runner


@pytest.fixture(autouse=True)
def _inject_modules():
    original_torch = runner.torch
    original_numpy = runner.np
    original_pandas = runner.pd
    previous_sys_modules = {name: sys.modules.get(name) for name in ("torch", "numpy", "pandas")}

    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
    numpy_stub = ModuleType("numpy")
    numpy_stub.isscalar = lambda value: not hasattr(value, "__len__")
    numpy_stub.bool_ = bool
    pandas_stub = ModuleType("pandas")
    runner.setup_training_imports(torch_stub, numpy_stub, pandas_stub)
    try:
        yield
    finally:
        runner.torch = original_torch
        runner.np = original_numpy
        runner.pd = original_pandas
        for name, module in previous_sys_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_simulate_trading_returns_summary(monkeypatch):
    class _Controller:
        def __init__(self):
            self._step = 0

        def current_time(self):
            return datetime(2024, 1, 1, 12, 0, 0)

        def advance_steps(self, step):
            self._step += step

        def summary(self):
            return {"cash": 10123.45, "equity": 11000.0, "positions": {"AAPL": 5}}

    @contextmanager
    def fake_activate_simulation(*args, **kwargs):
        yield _Controller()

    trade_module = ModuleType("trade_stock_e2e")

    def analyze_symbols(symbols):
        return {symbol: {"avg_return": 0.02 * (idx + 1), "confidence": 0.5} for idx, symbol in enumerate(symbols)}

    def log_trading_plan(current, name):
        logged.append((name, sorted(current)))

    def manage_positions(current, previous, analyzed):
        managed.append((list(current.keys()), len(analyzed)))

    def release_model_resources():
        released.append(True)

    trade_module.analyze_symbols = analyze_symbols
    trade_module.log_trading_plan = log_trading_plan
    trade_module.manage_positions = manage_positions
    trade_module.release_model_resources = release_model_resources

    logged = []
    managed = []
    released = []

    monkeypatch.setitem(sys.modules, "trade_stock_e2e", trade_module)

    env_module = ModuleType("marketsimulator.environment")
    env_module.activate_simulation = fake_activate_simulation
    monkeypatch.setitem(sys.modules, "marketsimulator.environment", env_module)

    monkeypatch.setenv("MARKETSIM_SIM_ANALYSIS_CHUNK", "1")

    result = runner.simulate_trading(
        symbols=["AAPL", "MSFT"],
        steps=2,
        step_size=1,
        initial_cash=10_000.0,
        top_k=1,
        kronos_only=False,
        compact_logs=False,
    )

    assert result["summary"]["cash"] == pytest.approx(10123.45)
    assert result["summary"]["positions"]["AAPL"] == 5
    assert len(result["timeline"]) == 2
    assert logged and managed and released
