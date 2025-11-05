import math
from typing import Dict

import pytest

from src import trade_stock_gate_utils as gate_utils


def test_coerce_positive_int_handles_invalid_inputs():
    assert gate_utils.coerce_positive_int(None, 5) == 5
    assert gate_utils.coerce_positive_int("7", 5) == 7
    assert gate_utils.coerce_positive_int("-2", 3) == 3
    assert gate_utils.coerce_positive_int("not-an-int", 2) == 2


def test_should_skip_closed_equity_respects_env(monkeypatch):
    monkeypatch.delenv("MARKETSIM_SKIP_CLOSED_EQUITY", raising=False)
    assert gate_utils.should_skip_closed_equity() is True
    monkeypatch.setenv("MARKETSIM_SKIP_CLOSED_EQUITY", "0")
    assert gate_utils.should_skip_closed_equity() is False
    monkeypatch.setenv("MARKETSIM_SKIP_CLOSED_EQUITY", "true")
    assert gate_utils.should_skip_closed_equity() is True


def test_get_trend_stat_uses_summary(monkeypatch):
    summary: Dict[str, Dict[str, float]] = {"AAPL": {"pnl": 1.25}}
    monkeypatch.setattr(gate_utils, "_load_trend_summary", lambda: summary)
    assert gate_utils.get_trend_stat("AAPL", "pnl") == pytest.approx(1.25)
    assert gate_utils.get_trend_stat("AAPL", "missing") is None
    assert gate_utils.get_trend_stat("MSFT", "pnl") is None


def test_is_tradeable_checks_spread(monkeypatch):
    monkeypatch.setattr(gate_utils, "DISABLE_TRADE_GATES", False, raising=False)
    monkeypatch.setattr(gate_utils, "compute_spread_bps", lambda bid, ask: math.inf)
    tradeable, reason = gate_utils.is_tradeable("AAPL", None, None)
    assert tradeable is False
    assert "Missing bid/ask" in reason

    monkeypatch.setattr(gate_utils, "compute_spread_bps", lambda bid, ask: 12.5)
    tradeable, reason = gate_utils.is_tradeable("AAPL", 100.0, 100.1)
    assert tradeable is True
    assert "Spread 12.5bps" in reason


def test_pass_edge_threshold_accounts_for_costs(monkeypatch):
    monkeypatch.setattr(gate_utils, "DISABLE_TRADE_GATES", False, raising=False)
    monkeypatch.setattr(gate_utils, "is_kronos_only_mode", lambda: False)
    monkeypatch.setattr(gate_utils, "expected_cost_bps", lambda symbol: 20.0)

    ok, reason = gate_utils.pass_edge_threshold("AAPL", 0.002)  # 20bps move
    assert ok is False
    assert "need 30.0bps" in reason

    ok, reason = gate_utils.pass_edge_threshold("AAPL", 0.005)  # 50bps move
    assert ok is True
    assert "≥ need" in reason


def test_resolve_signal_sign_respects_kronos_mode(monkeypatch):
    monkeypatch.setattr(gate_utils, "CONSENSUS_MIN_MOVE_PCT", 0.01, raising=False)
    monkeypatch.setattr(gate_utils, "is_kronos_only_mode", lambda: False)

    assert gate_utils.resolve_signal_sign(0.0) == 0
    assert gate_utils.resolve_signal_sign(0.02) == 1
    assert gate_utils.resolve_signal_sign(-0.03) == -1

    monkeypatch.setattr(gate_utils, "is_kronos_only_mode", lambda: True)
    # Threshold quarters to 0.0025 with Kronos mode
    assert gate_utils.resolve_signal_sign(0.003) == 1
