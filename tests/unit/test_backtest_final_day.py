import importlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


def _load_backtest_module():
    module = importlib.import_module("backtest_test3_inline")
    if getattr(module, "evaluate_strategy", None) is not None:
        return module

    import_error = getattr(module, "__import_error__", None)
    module_path = Path(__file__).resolve().parents[2] / "backtest_test3_inline.py"
    spec = importlib.util.spec_from_file_location("backtest_test3_inline_real", module_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"backtest_test3_inline unavailable: {import_error!r}")

    real_module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    try:
        loader.exec_module(real_module)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - skip when optional deps missing
        reason = import_error or exc
        pytest.skip(f"backtest_test3_inline unavailable: {reason!r}")

    sys.modules["backtest_test3_inline_real"] = real_module
    return real_module


def _series(values):
    return pd.Series(values, index=pd.RangeIndex(len(values), name="t"))


def test_final_day_resets_to_flat_when_guard_cancels_trade(monkeypatch):
    bt3 = _load_backtest_module()
    monkeypatch.setattr(bt3, "SPREAD", 1.0, raising=False)
    actual_returns = _series([0.1, -0.1, -0.1, -0.1])
    raw_signals = torch.tensor([1.0, 1.0, -1.0, -1.0])

    evaluation = bt3.evaluate_strategy(raw_signals, actual_returns, trading_fee=0.0, trading_days_per_year=252)
    final_day = bt3._final_day_return_from_series(evaluation.returns)

    assert final_day == pytest.approx(0.0, abs=1e-9)


def test_final_day_matches_realized_return_without_guard(monkeypatch):
    bt3 = _load_backtest_module()
    monkeypatch.setattr(bt3, "SPREAD", 1.0, raising=False)
    actual_returns = _series([0.05, 0.02, 0.03])
    raw_signals = torch.ones(len(actual_returns))

    evaluation = bt3.evaluate_strategy(raw_signals, actual_returns, trading_fee=0.0, trading_days_per_year=252)
    final_day = bt3._final_day_return_from_series(evaluation.returns)

    assert final_day == pytest.approx(actual_returns.iloc[-1], abs=1e-9)
