from __future__ import annotations

import importlib

import pytest

from src.risk_state import ProbeState


@pytest.fixture(scope="module")
def trade_module():
    module = importlib.import_module("trade_stock_e2e")
    return module


def test_forecast_plus_sim_nonpositive_detects_negative_sum(trade_module):
    data = {
        "strategy": "simple",
        "strategy_candidate_forecasted_pnl": {"simple": 0.05},
        "recent_return_sum": -0.2,
    }
    result = trade_module._forecast_plus_sim_nonpositive(data)
    assert result == pytest.approx((0.05, -0.2))


def test_collect_forced_reasons_includes_last_two_losses(monkeypatch, trade_module):
    monkeypatch.setattr(trade_module, "DISABLE_RECENT_PNL_PROBE", False)
    monkeypatch.setattr(trade_module, "_recent_trade_pnl_pcts", lambda *args, **kwargs: [-0.05, -0.03])
    data = {
        "side": "buy",
        "strategy": "simple",
        "strategy_candidate_forecasted_pnl": {"simple": 1.0},
        "recent_return_sum": 0.1,
    }
    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    reasons = trade_module._collect_forced_probe_reasons("AAPL", data, probe_state)
    assert any("recent_pnl_pct_sum" in reason for reason in reasons)


def test_apply_forced_probe_annotations_sets_trade_mode(monkeypatch, trade_module):
    monkeypatch.setattr(trade_module, "DISABLE_RECENT_PNL_PROBE", False)
    monkeypatch.setattr(trade_module, "_recent_trade_pnl_pcts", lambda *args, **kwargs: [])
    monkeypatch.setattr(trade_module, "_recent_trade_pnls", lambda *args, **kwargs: [-2.0, -2.5])
    data = {
        "side": "buy",
        "strategy": "simple",
        "strategy_candidate_forecasted_pnl": {"simple": 0.4},
        "recent_return_sum": 0.05,
        "trade_mode": "normal",
    }
    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    trade_module._apply_forced_probe_annotations({"AAPL": data}, probe_state)
    assert data["forced_probe"] is True
    assert data["trade_mode"] == "probe"
    assert data.get("pending_probe") is True


def test_collect_forced_reasons_includes_global(monkeypatch, trade_module):
    monkeypatch.setattr(trade_module, "_recent_trade_pnl_pcts", lambda *args, **kwargs: [])
    monkeypatch.setattr(trade_module, "_recent_trade_pnls", lambda *args, **kwargs: [])
    data = {
        "side": "sell",
        "strategy": "simple",
        "strategy_candidate_forecasted_pnl": {"simple": 0.5},
        "recent_return_sum": 0.2,
    }
    probe_state = ProbeState(force_probe=True, reason="global-loss", probe_date=None, state={})
    reasons = trade_module._collect_forced_probe_reasons("MSFT", data, probe_state)
    assert any(reason.startswith("global_loss") for reason in reasons)
