"""Tests for multi-service symbol conflict prevention.

Verifies that:
  1. service_config.json can be loaded and parsed.
  2. Symbol sets are disjoint across all services.
  3. load_service_symbols returns the correct (crypto, stocks) tuples.
  4. warn_position_conflicts logs correctly when a conflict exists.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from unified_orchestrator.symbol_lock import (
    assert_no_overlaps,
    find_symbol_overlaps,
    get_all_symbols_by_service,
    load_config,
    load_service_symbols,
    warn_position_conflicts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    """Write a minimal valid service_config.json and return its path."""
    cfg = {
        "unified-orchestrator": {
            "crypto_symbols": ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"],
            "stock_symbols": [],
        },
        "daily-rl-trader": {
            "crypto_symbols": [],
            "stock_symbols": ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA",
                              "SPY", "QQQ", "PLTR", "JPM", "V", "AMZN"],
        },
        "alpaca-hourly-trader": {
            "crypto_symbols": [],
            "stock_symbols": [],
        },
        "trade-unified-hourly-meta": {
            "crypto_symbols": [],
            "stock_symbols": ["DBX", "TRIP", "MTCH", "NYT", "NET",
                              "BKNG", "EBAY", "EXPE", "ITUB", "BTG", "ABEV"],
        },
    }
    p = tmp_path / "service_config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture()
def overlapping_config_path(tmp_path: Path) -> Path:
    """Config with deliberate overlaps (BTCUSD, NVDA)."""
    cfg = {
        "service-a": {
            "crypto_symbols": ["BTCUSD", "ETHUSD"],
            "stock_symbols": ["NVDA", "PLTR"],
        },
        "service-b": {
            "crypto_symbols": ["BTCUSD"],   # overlap!
            "stock_symbols": ["NVDA", "MSFT"],  # overlap!
        },
    }
    p = tmp_path / "service_config_overlap.json"
    p.write_text(json.dumps(cfg))
    return p


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_returns_dict(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert isinstance(cfg, dict)
    assert "unified-orchestrator" in cfg


def test_load_config_missing_file(tmp_path: Path) -> None:
    result = load_config(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_config_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{")
    result = load_config(bad)
    assert result == {}


# ---------------------------------------------------------------------------
# load_service_symbols
# ---------------------------------------------------------------------------

def test_load_service_symbols_orchestrator(config_path: Path) -> None:
    crypto, stocks = load_service_symbols("unified-orchestrator", config_path)
    assert set(crypto) == {"BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"}
    assert stocks == []


def test_load_service_symbols_daily_rl(config_path: Path) -> None:
    crypto, stocks = load_service_symbols("daily-rl-trader", config_path)
    assert crypto == []
    assert set(stocks) == {"AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA",
                           "SPY", "QQQ", "PLTR", "JPM", "V", "AMZN"}


def test_load_service_symbols_alpaca_hourly(config_path: Path) -> None:
    crypto, stocks = load_service_symbols("alpaca-hourly-trader", config_path)
    assert crypto == []
    assert stocks == []


def test_load_service_symbols_meta(config_path: Path) -> None:
    crypto, stocks = load_service_symbols("trade-unified-hourly-meta", config_path)
    assert crypto == []
    assert "DBX" in stocks
    assert "NET" in stocks
    # Confirm none of the daily RL stocks leak in
    assert "AAPL" not in stocks
    assert "GOOG" not in stocks
    assert "NVDA" not in stocks
    assert "META" not in stocks
    assert "MSFT" not in stocks
    assert "PLTR" not in stocks
    assert "TSLA" not in stocks


def test_load_service_symbols_unknown_service(config_path: Path) -> None:
    crypto, stocks = load_service_symbols("nonexistent-service", config_path)
    assert crypto == []
    assert stocks == []


def test_load_service_symbols_uppercase(config_path: Path) -> None:
    """Symbols are always uppercased regardless of config case."""
    # Patch the config with lowercase symbols
    low_cfg = {
        "svc": {"crypto_symbols": ["btcusd"], "stock_symbols": ["nvda"]},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(low_cfg, fh)
        p = Path(fh.name)
    crypto, stocks = load_service_symbols("svc", p)
    assert crypto == ["BTCUSD"]
    assert stocks == ["NVDA"]
    p.unlink()


# ---------------------------------------------------------------------------
# Disjoint (no overlap) checks
# ---------------------------------------------------------------------------

def test_symbol_sets_are_disjoint(config_path: Path) -> None:
    """The canonical service_config.json must have fully disjoint symbol sets."""
    overlaps = find_symbol_overlaps(config_path)
    assert overlaps == {}, f"Symbol overlaps found: {overlaps}"


def test_assert_no_overlaps_passes_for_valid_config(config_path: Path) -> None:
    assert_no_overlaps(config_path)  # must not raise


def test_find_overlaps_detects_conflicts(overlapping_config_path: Path) -> None:
    overlaps = find_symbol_overlaps(overlapping_config_path)
    assert "BTCUSD" in overlaps
    assert "NVDA" in overlaps
    assert set(overlaps["BTCUSD"]) == {"service-a", "service-b"}
    assert set(overlaps["NVDA"]) == {"service-a", "service-b"}
    # PLTR and MSFT should not appear (only in one service each)
    assert "PLTR" not in overlaps
    assert "MSFT" not in overlaps


def test_assert_no_overlaps_raises_on_conflicts(overlapping_config_path: Path) -> None:
    with pytest.raises(ValueError, match="Symbol ownership conflict"):
        assert_no_overlaps(overlapping_config_path)


# ---------------------------------------------------------------------------
# Production config is internally consistent
# ---------------------------------------------------------------------------

def test_production_config_exists() -> None:
    prod_config = Path(__file__).resolve().parent.parent / "unified_orchestrator" / "service_config.json"
    assert prod_config.exists(), f"service_config.json missing at {prod_config}"


def test_production_config_no_overlaps() -> None:
    prod_config = Path(__file__).resolve().parent.parent / "unified_orchestrator" / "service_config.json"
    assert_no_overlaps(prod_config)


def test_production_config_orchestrator_is_crypto_only() -> None:
    prod_config = Path(__file__).resolve().parent.parent / "unified_orchestrator" / "service_config.json"
    crypto, stocks = load_service_symbols("unified-orchestrator", prod_config)
    assert {"BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"} <= set(crypto)
    assert stocks == []


def test_production_config_daily_rl_owns_stocks12_live_universe() -> None:
    prod_config = Path(__file__).resolve().parent.parent / "unified_orchestrator" / "service_config.json"
    _, daily_stocks = load_service_symbols("daily-rl-trader", prod_config)
    assert set(daily_stocks) == {"AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA",
                                 "SPY", "QQQ", "PLTR", "JPM", "V", "AMZN"}


def test_production_config_meta_excludes_daily_rl_stocks() -> None:
    """trade-unified-hourly-meta must not overlap the live daily RL stock universe."""
    prod_config = Path(__file__).resolve().parent.parent / "unified_orchestrator" / "service_config.json"
    _, meta_stocks = load_service_symbols("trade-unified-hourly-meta", prod_config)
    daily_stocks = {"AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "SPY", "QQQ", "PLTR", "JPM", "V", "AMZN"}
    conflicts = daily_stocks & set(meta_stocks)
    assert not conflicts, f"Meta service must not trade daily RL stocks: {conflicts}"


# ---------------------------------------------------------------------------
# warn_position_conflicts
# ---------------------------------------------------------------------------

def _make_position(symbol: str, qty: str = "1.0", market_value: str = "1000.0"):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    pos.market_value = market_value
    return pos


def test_warn_position_conflicts_no_conflict(config_path: Path, caplog) -> None:
    """No warnings when positions belong to this service."""
    api = MagicMock()
    api.get_all_positions.return_value = [
        _make_position("SOLUSD"),
        _make_position("BTCUSD"),
    ]
    with patch("unified_orchestrator.symbol_lock.logger") as mock_log:
        warn_position_conflicts("unified-orchestrator", api, config_path)
        # No WARNING calls expected
        warning_calls = [c for c in mock_log.warning.call_args_list
                         if "CONFLICT" in str(c)]
        assert warning_calls == []


def test_warn_position_conflicts_detects_overlap(config_path: Path) -> None:
    """Warns when orchestrator holds a position that belongs to another service."""
    api = MagicMock()
    # AAPL belongs to daily-rl-trader, not unified-orchestrator
    api.get_all_positions.return_value = [_make_position("AAPL")]
    with patch("unified_orchestrator.symbol_lock.logger") as mock_log:
        warn_position_conflicts("unified-orchestrator", api, config_path)
        warning_msgs = [str(c) for c in mock_log.warning.call_args_list]
        assert any("CONFLICT" in m and "AAPL" in m for m in warning_msgs), (
            f"Expected CONFLICT warning for AAPL, got: {warning_msgs}"
        )


def test_warn_position_conflicts_api_error(config_path: Path) -> None:
    """Gracefully handles API errors without raising."""
    api = MagicMock()
    api.get_all_positions.side_effect = RuntimeError("network error")
    # Should not raise
    warn_position_conflicts("unified-orchestrator", api, config_path)


def test_warn_position_conflicts_uses_list_positions_fallback(config_path: Path) -> None:
    """Falls back to list_positions() when get_all_positions() is absent."""
    api = MagicMock(spec=["list_positions"])
    api.list_positions.return_value = [_make_position("NET")]
    with patch("unified_orchestrator.symbol_lock.logger") as mock_log:
        warn_position_conflicts("unified-orchestrator", api, config_path)
        warning_msgs = [str(c) for c in mock_log.warning.call_args_list]
        assert any("CONFLICT" in m and "NET" in m for m in warning_msgs)


# ---------------------------------------------------------------------------
# get_all_symbols_by_service
# ---------------------------------------------------------------------------

def test_get_all_symbols_by_service(config_path: Path) -> None:
    by_svc = get_all_symbols_by_service(config_path)
    assert "unified-orchestrator" in by_svc
    assert "daily-rl-trader" in by_svc
    assert "alpaca-hourly-trader" in by_svc
    assert "trade-unified-hourly-meta" in by_svc

    orch = by_svc["unified-orchestrator"]
    assert "SOLUSD" in orch
    assert "BTCUSD" in orch
    assert "NVDA" not in orch

    daily = by_svc["daily-rl-trader"]
    assert "NVDA" in daily
    assert "AAPL" in daily

    hourly = by_svc["alpaca-hourly-trader"]
    assert hourly == set()
