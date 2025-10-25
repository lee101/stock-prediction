from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime
from types import ModuleType, SimpleNamespace
from typing import Dict, Iterable

import pytest
from fal_marketsimulator import runner as fal_runner
from falmarket.app import MarketSimulatorApp
from src.runtime_imports import _reset_for_tests


def _build_torch_stub() -> ModuleType:
    torch_stub = ModuleType("torch")
    torch_stub.__version__ = "0.0-test"

    @contextmanager
    def _ctx():
        yield

    torch_stub.inference_mode = lambda *args, **kwargs: _ctx()
    torch_stub.no_grad = lambda *args, **kwargs: _ctx()
    torch_stub.autocast = lambda *args, **kwargs: _ctx()
    torch_stub.compile = lambda module, **kwargs: module  # pragma: no cover - not exercised
    torch_stub.tensor = lambda data, **kwargs: data  # type: ignore[assignment]
    torch_stub.zeros = lambda *args, **kwargs: 0
    torch_stub.ones_like = lambda tensor, **kwargs: tensor
    torch_stub.zeros_like = lambda tensor, **kwargs: tensor
    torch_stub.full = lambda *args, **kwargs: 0
    torch_stub.float = object()
    cuda_ns = SimpleNamespace(
        is_available=lambda: False,
        amp=SimpleNamespace(autocast=lambda **_: nullcontext()),
        empty_cache=lambda: None,
        get_device_name=lambda idx: f"cuda:{idx}",
        current_device=lambda: 0,
    )
    torch_stub.cuda = cuda_ns  # type: ignore[assignment]
    backends_ns = SimpleNamespace(cuda=SimpleNamespace(enable_flash_sdp=lambda *args, **kwargs: None))
    torch_stub.backends = backends_ns  # type: ignore[assignment]
    return torch_stub


def _build_numpy_stub() -> ModuleType:
    numpy_stub = ModuleType("numpy")
    numpy_stub.asarray = lambda data, **kwargs: list(data)
    numpy_stub.quantile = lambda data, qs, axis=0: [0.1, 0.5, 0.9]
    numpy_stub.float64 = float
    numpy_stub.sort = lambda matrix, axis=0: matrix
    numpy_stub.median = lambda matrix, axis=0: matrix[0]
    numpy_stub.mean = lambda matrix, axis=0, dtype=None: matrix[0]
    numpy_stub.std = lambda matrix, axis=0, dtype=None: matrix[0]
    numpy_stub.clip = lambda array, a_min, a_max: array
    numpy_stub.array = lambda data, **kwargs: list(data)
    numpy_stub.bool_ = bool
    return numpy_stub


def _build_pandas_stub() -> ModuleType:
    pandas_stub = ModuleType("pandas")
    pandas_stub.DataFrame = dict  # minimal placeholder
    pandas_stub.Series = dict
    pandas_stub.Index = list
    pandas_stub.to_datetime = lambda values, **kwargs: values
    return pandas_stub


def _register_trade_module() -> None:
    trade_module = ModuleType("trade_stock_e2e")

    def analyze_symbols(symbols: Iterable[str]) -> Dict[str, Dict[str, float]]:
        return {symbol: {"avg_return": 0.1, "confidence": 0.5} for symbol in symbols}

    def log_trading_plan(current, name):
        pass

    def manage_positions(current, previous, analyzed):
        pass

    def release_model_resources():
        pass

    trade_module.analyze_symbols = analyze_symbols  # type: ignore[attr-defined]
    trade_module.log_trading_plan = log_trading_plan  # type: ignore[attr-defined]
    trade_module.manage_positions = manage_positions  # type: ignore[attr-defined]
    trade_module.release_model_resources = release_model_resources  # type: ignore[attr-defined]
    sys.modules["trade_stock_e2e"] = trade_module


def _register_environment_module() -> None:
    env_module = ModuleType("marketsimulator.environment")

    class _Controller:
        def __init__(self):
            self._step = 0

        def current_time(self):
            return datetime(2025, 1, 1, 0, 0, 0)

        def advance_steps(self, step):
            self._step += step

        def summary(self):
            return {"cash": 100_500.0, "equity": 110_000.0}

    @contextmanager
    def activate_simulation(*args, **kwargs):
        yield _Controller()

    env_module.activate_simulation = activate_simulation  # type: ignore[attr-defined]
    sys.modules["marketsimulator.environment"] = env_module


@pytest.fixture(autouse=True)
def _cleanup_modules():
    preserved = {name: mod for name, mod in sys.modules.items()}
    try:
        yield
    finally:
        to_delete = set(sys.modules) - set(preserved)
        for name in to_delete:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        _reset_for_tests()


def test_simulate_trading_only_uses_allowed_packages(monkeypatch):
    for heavy in ("torch", "numpy", "pandas"):
        sys.modules.pop(heavy, None)

    torch_stub = _build_torch_stub()
    numpy_stub = _build_numpy_stub()
    pandas_stub = _build_pandas_stub()

    fal_runner.setup_training_imports(torch_stub, numpy_stub, pandas_stub)
    _register_trade_module()
    _register_environment_module()

    repo_modules_before = {
        name for name, mod in sys.modules.items() if getattr(mod, "__file__", "") and "code/stock" in mod.__file__
    }
    result = fal_runner.simulate_trading(
        symbols=["AAPL", "MSFT"],
        steps=2,
        step_size=1,
        initial_cash=100_000.0,
        top_k=1,
        kronos_only=False,
        compact_logs=True,
    )

    assert result["summary"]["cash"] == pytest.approx(100_500.0)
    assert len(result["timeline"]) == 2

    repo_modules_after = {
        name for name, mod in sys.modules.items() if getattr(mod, "__file__", "") and "code/stock" in mod.__file__
    }
    new_modules = repo_modules_after - repo_modules_before

    allowed = set(MarketSimulatorApp.local_python_modules) | {
        "falmarket",
        "fal_marketsimulator",
        "faltrain",
        "marketsimulator",
        "trade_stock_e2e",
        "trade_stock_e2e_trained",
        "src",
        "stock",
        "utils",
        "traininglib",
        "rlinference",
        "training",
        "gymrl",
        "analysis",
        "analysis_runner_funcs",
        "tests",
    }

    disallowed = []
    for module_name in new_modules:
        root = module_name.split(".")[0]
        if root not in allowed:
            disallowed.append(module_name)

    assert not disallowed, f"Modules outside local_python_modules imported: {disallowed}"

    assert sys.modules["torch"] is torch_stub
    assert sys.modules["numpy"] is numpy_stub
    assert sys.modules["pandas"] is pandas_stub
