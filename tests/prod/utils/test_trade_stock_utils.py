from __future__ import annotations

import math

import pytest

from src.trade_stock_utils import (
    agree_direction,
    coerce_optional_float,
    compute_spread_bps,
    edge_threshold_bps,
    evaluate_strategy_entry_gate,
    expected_cost_bps,
    kelly_lite,
    parse_float_list,
    resolve_spread_cap,
    should_rebalance,
)


def test_coerce_optional_float_basic_cases():
    assert coerce_optional_float(None) is None
    assert coerce_optional_float(" 1.50 ") == pytest.approx(1.5)
    assert coerce_optional_float(7) == pytest.approx(7.0)
    assert coerce_optional_float(float("nan")) is None
    assert coerce_optional_float("nan") is None


def test_parse_float_list_from_string_and_iterable():
    text = "[1.0, 2, 'nan', '3.5']"
    assert parse_float_list(text) == [1.0, 2.0, 3.5]
    assert parse_float_list([1, "4.0", None]) == [1.0, 4.0]
    assert parse_float_list(None) is None
    assert parse_float_list("[]") is None
    assert parse_float_list("invalid") is None


def test_compute_spread_bps_and_resolve_cap():
    assert compute_spread_bps(99.5, 100.5) == pytest.approx(100.0)
    assert math.isinf(compute_spread_bps(None, 100.0))
    assert resolve_spread_cap("BTCUSD") == 35
    assert resolve_spread_cap("AAPL") == 8
    assert resolve_spread_cap("RANDOM") == 25


def test_expected_cost_and_edge_threshold():
    assert expected_cost_bps("BTCUSD") == pytest.approx(20.0)
    assert expected_cost_bps("META") == pytest.approx(31.0)
    assert edge_threshold_bps("AAPL") == pytest.approx(16.0)
    assert edge_threshold_bps("ETHUSD") == pytest.approx(40.0)


def test_agree_direction_and_kelly_lite():
    assert agree_direction(1, 1, 0, 1) is True
    assert agree_direction(1, -1) is False
    assert kelly_lite(0.02, 0.1) == pytest.approx(0.15)
    assert kelly_lite(-0.01, 0.1) == 0.0
    assert kelly_lite(0.02, 0.0) == 0.0
    assert kelly_lite(1.0, 0.5, cap=0.1) == pytest.approx(0.1)


def test_should_rebalance_decisions():
    assert should_rebalance("buy", "sell", 10.0, 9.0) is True
    assert should_rebalance("buy", "buy", 10.0, 10.1, eps=0.05) is False
    assert should_rebalance(None, "buy", 0.0, 5.0) is True
    assert should_rebalance("sell", "sell", 8.0, 5.0, eps=0.1) is True


def test_evaluate_strategy_entry_gate_passes_when_metrics_strong():
    ok, reason = evaluate_strategy_entry_gate(
        "AAPL",
        {
            "avg_return": 0.02,
            "sharpe": 0.9,
            "turnover": 1.2,
            "max_drawdown": -0.05,
        },
        fallback_used=False,
        sample_size=200,
    )
    assert ok is True
    assert reason == "ok"


def test_evaluate_strategy_entry_gate_rejects_fallback_and_low_edge():
    ok, reason = evaluate_strategy_entry_gate(
        "AAPL",
        {"avg_return": 0.0005, "sharpe": 0.6, "turnover": 1.0, "max_drawdown": -0.02},
        fallback_used=False,
        sample_size=200,
    )
    assert ok is False
    assert "edge" in reason

    ok_fallback, reason_fallback = evaluate_strategy_entry_gate(
        "AAPL",
        {"avg_return": 0.02, "sharpe": 1.0, "turnover": 0.5, "max_drawdown": -0.01},
        fallback_used=True,
        sample_size=200,
    )
    assert ok_fallback is False
    assert reason_fallback == "fallback_metrics"


def test_evaluate_strategy_entry_gate_accepts_liquid_crypto_with_smaller_sample():
    ok, reason = evaluate_strategy_entry_gate(
        "UNIUSD",
        {"avg_return": 0.015, "sharpe": 1.5, "turnover": 1.2, "max_drawdown": -0.04},
        fallback_used=False,
        sample_size=70,
    )
    assert ok is True
    assert reason == "ok"
