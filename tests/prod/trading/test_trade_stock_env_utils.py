from __future__ import annotations

import pytest

import src.trade_stock_env_utils as env_utils


@pytest.fixture(autouse=True)
def reset_env_utils_state(monkeypatch):
    monkeypatch.setattr(env_utils, "_THRESHOLD_MAP_CACHE", {}, raising=False)
    monkeypatch.setattr(env_utils, "_SYMBOL_MAX_ENTRIES_CACHE", None, raising=False)
    monkeypatch.setattr(env_utils, "_SYMBOL_FORCE_PROBE_CACHE", None, raising=False)
    monkeypatch.setattr(env_utils, "_SYMBOL_RUN_ENTRY_COUNTS", {}, raising=False)
    monkeypatch.setattr(env_utils, "_SYMBOL_RUN_ENTRY_ID", None, raising=False)
    monkeypatch.delenv("MARKETSIM_SYMBOL_MAX_ENTRIES_MAP", raising=False)
    monkeypatch.delenv("MARKETSIM_SYMBOL_FORCE_PROBE_MAP", raising=False)
    yield


def test_symbol_max_entries_per_run_precedence(monkeypatch):
    monkeypatch.setenv(
        "MARKETSIM_SYMBOL_MAX_ENTRIES_MAP",
        "AAPL@maxdiff:1, AAPL:3, @maxdiff:5, @:7",
    )

    primary_limit, primary_key = env_utils._symbol_max_entries_per_run("AAPL", "maxdiff")
    symbol_limit, symbol_key = env_utils._symbol_max_entries_per_run("AAPL", "probe")
    strategy_limit, strategy_key = env_utils._symbol_max_entries_per_run("QQQ", "maxdiff")
    default_limit, default_key = env_utils._symbol_max_entries_per_run("QQQ", "probe")

    assert primary_limit == 1
    assert primary_key == ("aapl", "maxdiff")
    assert symbol_limit == 3
    assert symbol_key == ("aapl", None)
    assert strategy_limit == 5
    assert strategy_key == (None, "maxdiff")
    assert default_limit == 7
    assert default_key == (None, None)


def test_entry_counter_snapshot_includes_aggregated_information(monkeypatch):
    monkeypatch.setenv(
        "MARKETSIM_SYMBOL_MAX_ENTRIES_MAP",
        "AAPL@maxdiff:1, AAPL:3, @:4",
    )

    env_utils.reset_symbol_entry_counters("run-123")
    env_utils._increment_symbol_entry("AAPL", "maxdiff")
    env_utils._increment_symbol_entry("AAPL", "maxdiff")
    env_utils._increment_symbol_entry("AAPL", None)
    env_utils._increment_symbol_entry("MSFT", None)

    snapshot = env_utils.get_entry_counter_snapshot()

    per_key = snapshot["per_key"]
    assert per_key["AAPL@maxdiff"]["entries"] == 2
    assert per_key["AAPL@maxdiff"]["entry_limit"] == pytest.approx(1.0)
    assert per_key["AAPL@maxdiff"]["approx_trade_limit"] == pytest.approx(2.0)
    assert per_key["AAPL@maxdiff"]["resolved_limit_key"] == "aapl@maxdiff"

    assert per_key["AAPL"]["entries"] == 1
    assert per_key["AAPL"]["entry_limit"] == pytest.approx(3.0)
    assert per_key["AAPL"]["approx_trade_limit"] == pytest.approx(6.0)

    assert per_key["MSFT"]["entries"] == 1
    assert per_key["MSFT"]["entry_limit"] == pytest.approx(4.0)

    per_symbol = snapshot["per_symbol"]
    assert per_symbol["AAPL"]["entries"] == 3
    assert per_symbol["AAPL"]["entry_limit"] == pytest.approx(1.0)
    assert per_symbol["AAPL"]["approx_trade_limit"] == pytest.approx(2.0)
    assert per_symbol["MSFT"]["entries"] == 1
    assert per_symbol["MSFT"]["entry_limit"] == pytest.approx(4.0)


def test_symbol_force_probe_truthy_map(monkeypatch):
    monkeypatch.setenv(
        "MARKETSIM_SYMBOL_FORCE_PROBE_MAP",
        "AAPL:yes, MSFT:no, TSLA",
    )

    assert env_utils._symbol_force_probe("AAPL") is True
    assert env_utils._symbol_force_probe("TSLA") is True
    assert env_utils._symbol_force_probe("MSFT") is False
    assert env_utils._symbol_force_probe("AMZN") is False
