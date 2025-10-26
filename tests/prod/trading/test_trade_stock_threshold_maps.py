from __future__ import annotations

import pytest

import trade_stock_e2e as trade_module


@pytest.fixture(autouse=True)
def reset_threshold_caches(monkeypatch):
    monkeypatch.setattr(trade_module, "_THRESHOLD_MAP_CACHE", {}, raising=False)
    monkeypatch.setattr(trade_module, "_DRAW_CAPS_CACHE", None, raising=False)
    monkeypatch.setattr(trade_module, "_DRAW_RESUME_CACHE", None, raising=False)
    monkeypatch.delenv("TEST_THRESHOLD_ENV", raising=False)
    monkeypatch.delenv("MARKETSIM_KELLY_DRAWDOWN_CAP_MAP", raising=False)
    monkeypatch.delenv("MARKETSIM_KELLY_DRAWDOWN_CAP", raising=False)
    monkeypatch.delenv("MARKETSIM_DRAWDOWN_RESUME_MAP", raising=False)
    monkeypatch.delenv("MARKETSIM_DRAWDOWN_RESUME", raising=False)
    monkeypatch.delenv("MARKETSIM_DRAWDOWN_RESUME_FACTOR", raising=False)
    yield


def test_parse_threshold_map_supports_symbol_and_strategy_specific_entries(monkeypatch):
    monkeypatch.setenv(
        "TEST_THRESHOLD_ENV",
        "AAPL@maxdiff:1.2, AAPL:0.9, maxdiff:0.5, fallback:0.2, @:0.1, invalid-entry, :0.7",
    )

    parsed = trade_module._parse_threshold_map("TEST_THRESHOLD_ENV")

    assert parsed[("aapl", "maxdiff")] == pytest.approx(1.2)
    assert parsed[("aapl", None)] == pytest.approx(0.9)
    assert parsed[(None, "maxdiff")] == pytest.approx(0.5)
    assert parsed[(None, "fallback")] == pytest.approx(0.2)
    assert parsed[(None, None)] == pytest.approx(0.1)
    assert len(parsed) == 5  # invalid entries ignored


def test_lookup_threshold_applies_precedence(monkeypatch):
    monkeypatch.setenv(
        "TEST_THRESHOLD_ENV",
        "SPY@maxdiff:0.7, SPY:0.5, maxdiff:0.3, @:0.1",
    )

    primary = trade_module._lookup_threshold("TEST_THRESHOLD_ENV", "SPY", "maxdiff")
    symbol_only = trade_module._lookup_threshold("TEST_THRESHOLD_ENV", "SPY", "probe")
    strategy_only = trade_module._lookup_threshold("TEST_THRESHOLD_ENV", "QQQ", "maxdiff")
    default_value = trade_module._lookup_threshold("TEST_THRESHOLD_ENV", "QQQ", "probe")

    assert primary == pytest.approx(0.7)
    assert symbol_only == pytest.approx(0.5)
    assert strategy_only == pytest.approx(0.3)
    assert default_value == pytest.approx(0.1)


def test_drawdown_cap_map_and_fallback(monkeypatch):
    monkeypatch.setenv(
        "MARKETSIM_KELLY_DRAWDOWN_CAP_MAP",
        "SPY@maxdiff:0.35, SPY:0.3, maxdiff:0.25",
    )
    monkeypatch.setenv("MARKETSIM_KELLY_DRAWDOWN_CAP", "0.8")

    cap_primary = trade_module._drawdown_cap_for("maxdiff", "SPY")
    cap_symbol = trade_module._drawdown_cap_for("probe", "SPY")
    cap_strategy = trade_module._drawdown_cap_for("maxdiff", "QQQ")
    cap_default = trade_module._drawdown_cap_for("probe", "QQQ")

    assert cap_primary == pytest.approx(0.35)
    assert cap_symbol == pytest.approx(0.3)
    assert cap_strategy == pytest.approx(0.25)
    assert cap_default == pytest.approx(0.8)


def test_drawdown_resume_map_and_factor(monkeypatch):
    monkeypatch.setenv(
        "MARKETSIM_DRAWDOWN_RESUME_MAP",
        "SPY@maxdiff:0.2, SPY:0.15, maxdiff:0.12",
    )
    monkeypatch.setenv("MARKETSIM_DRAWDOWN_RESUME_FACTOR", "0.6")

    resume_primary = trade_module._drawdown_resume_for("maxdiff", cap=0.3, symbol="SPY")
    resume_symbol = trade_module._drawdown_resume_for("probe", cap=0.3, symbol="SPY")
    resume_strategy = trade_module._drawdown_resume_for("maxdiff", cap=0.3, symbol="QQQ")
    resume_factor = trade_module._drawdown_resume_for("probe", cap=0.5, symbol="QQQ")

    assert resume_primary == pytest.approx(0.2)
    assert resume_symbol == pytest.approx(0.15)
    assert resume_strategy == pytest.approx(0.12)
    assert resume_factor == pytest.approx(0.3)  # factor 0.6 * cap 0.5
