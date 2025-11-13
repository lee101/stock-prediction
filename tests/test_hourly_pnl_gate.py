"""Tests for hourly PnL gate functionality."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from jsonshelve import FlatShelf

from src.hourly_pnl_gate import (
    get_pnl_blocking_report,
    get_recent_trade_pnl,
    should_block_trade_by_pnl,
)


@pytest.fixture
def temp_store():
    """Create a temporary trade history store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "trade_history_test.json"
        store = FlatShelf(str(store_path))

        def loader() -> Optional[FlatShelf]:
            return store

        yield loader, store


def test_no_trade_history_allows_trading(temp_store):
    """When there's no trade history, trading should be allowed (probe scenario)."""
    loader, _ = temp_store

    should_block, reason = should_block_trade_by_pnl(
        loader, "BTCUSD", "buy", max_trades=2
    )

    assert not should_block
    assert reason is None


def test_positive_pnl_allows_trading(temp_store):
    """When recent trades are profitable, trading should be allowed."""
    loader, store = temp_store

    # Add two profitable trades (key must match symbol case)
    store["BTCUSD|buy"] = [
        {"symbol": "BTCUSD", "side": "buy", "pnl": 10.0, "closed_at": "2025-01-01T10:00:00Z"},
        {"symbol": "BTCUSD", "side": "buy", "pnl": 15.0, "closed_at": "2025-01-01T12:00:00Z"},
    ]

    should_block, reason = should_block_trade_by_pnl(
        loader, "BTCUSD", "buy", max_trades=2
    )

    assert not should_block
    assert reason is None


def test_negative_pnl_blocks_trading(temp_store):
    """When recent trades have negative PnL, trading should be blocked."""
    loader, store = temp_store

    # Add two losing trades (key must match symbol case)
    store["BTCUSD|buy"] = [
        {"symbol": "BTCUSD", "side": "buy", "pnl": -10.0, "closed_at": "2025-01-01T10:00:00Z"},
        {"symbol": "BTCUSD", "side": "buy", "pnl": -5.0, "closed_at": "2025-01-01T12:00:00Z"},
    ]

    should_block, reason = should_block_trade_by_pnl(
        loader, "BTCUSD", "buy", max_trades=2
    )

    assert should_block
    assert "negative PnL: -15.00" in reason


def test_mixed_pnl_negative_sum_blocks(temp_store):
    """When recent trades sum to negative despite some wins, should block."""
    loader, store = temp_store

    # One win, one bigger loss (key must match symbol case)
    store["AAPL|buy"] = [
        {"symbol": "AAPL", "side": "buy", "pnl": 5.0, "closed_at": "2025-01-01T10:00:00Z"},
        {"symbol": "AAPL", "side": "buy", "pnl": -20.0, "closed_at": "2025-01-01T12:00:00Z"},
    ]

    should_block, reason = should_block_trade_by_pnl(
        loader, "AAPL", "buy", max_trades=2
    )

    assert should_block
    assert "negative PnL: -15.00" in reason


def test_single_negative_trade_blocks(temp_store):
    """When there's only one trade and it's negative, should block."""
    loader, store = temp_store

    store["ETHUSD|sell"] = [
        {"symbol": "ETHUSD", "side": "sell", "pnl": -10.0, "closed_at": "2025-01-01T10:00:00Z"},
    ]

    should_block, reason = should_block_trade_by_pnl(
        loader, "ETHUSD", "sell", max_trades=2
    )

    assert should_block
    assert "1 trade" in reason
    assert "negative PnL: -10.00" in reason


def test_only_considers_recent_trades(temp_store):
    """Should only look at the most recent N trades."""
    loader, store = temp_store

    # Old trades were losing, but recent ones are winning (key must match symbol case)
    store["TSLA|buy"] = [
        {"symbol": "TSLA", "side": "buy", "pnl": -50.0, "closed_at": "2025-01-01T08:00:00Z"},
        {"symbol": "TSLA", "side": "buy", "pnl": -30.0, "closed_at": "2025-01-01T09:00:00Z"},
        {"symbol": "TSLA", "side": "buy", "pnl": 20.0, "closed_at": "2025-01-01T10:00:00Z"},
        {"symbol": "TSLA", "side": "buy", "pnl": 25.0, "closed_at": "2025-01-01T11:00:00Z"},
    ]

    should_block, reason = should_block_trade_by_pnl(
        loader, "TSLA", "buy", max_trades=2
    )

    # Should only look at last 2 trades, which sum to +45
    assert not should_block
    assert reason is None


def test_strategy_specific_blocking(temp_store):
    """Should support strategy-specific PnL tracking."""
    loader, store = temp_store

    # maxdiff strategy had losses (key must match symbol case)
    store["BTCUSD|buy|maxdiff"] = [
        {"symbol": "BTCUSD", "side": "buy", "pnl": -10.0, "closed_at": "2025-01-01T10:00:00Z"},
    ]

    # highlow strategy was profitable
    store["BTCUSD|buy|highlow"] = [
        {"symbol": "BTCUSD", "side": "buy", "pnl": 20.0, "closed_at": "2025-01-01T10:00:00Z"},
    ]

    # maxdiff should be blocked
    should_block_maxdiff, _ = should_block_trade_by_pnl(
        loader, "BTCUSD", "buy", strategy="maxdiff", max_trades=2
    )
    assert should_block_maxdiff

    # highlow should be allowed
    should_block_highlow, _ = should_block_trade_by_pnl(
        loader, "BTCUSD", "buy", strategy="highlow", max_trades=2
    )
    assert not should_block_highlow


def test_get_recent_trade_pnl(temp_store):
    """Test the get_recent_trade_pnl helper function."""
    loader, store = temp_store

    store["NVDA|buy"] = [
        {"symbol": "NVDA", "side": "buy", "pnl": 10.0, "closed_at": "2025-01-01T10:00:00Z"},
        {"symbol": "NVDA", "side": "buy", "pnl": -5.0, "closed_at": "2025-01-01T11:00:00Z"},
        {"symbol": "NVDA", "side": "buy", "pnl": 15.0, "closed_at": "2025-01-01T12:00:00Z"},
    ]

    trades, total_pnl = get_recent_trade_pnl(
        loader, "NVDA", "buy", max_trades=2
    )

    assert len(trades) == 2
    assert total_pnl == 10.0  # -5 + 15


def test_pnl_blocking_report(temp_store):
    """Test generating a report of blocked symbols."""
    loader, store = temp_store

    # Set up some history (key must match symbol case used in query)
    store["BTCUSD|buy"] = [
        {"symbol": "BTCUSD", "side": "buy", "pnl": -10.0, "closed_at": "2025-01-01T10:00:00Z"},
    ]
    store["ETHUSD|buy"] = [
        {"symbol": "ETHUSD", "side": "buy", "pnl": 20.0, "closed_at": "2025-01-01T10:00:00Z"},
    ]

    report = get_pnl_blocking_report(
        loader,
        ["BTCUSD", "ETHUSD"],
        max_trades=2,
    )

    assert report["blocked_count"] >= 1
    assert report["allowed_count"] >= 1
    # BTCUSD buy should be blocked
    assert any("BTCUSD" in key and "buy" in key for key in report["blocked"])


def test_state_file_isolation():
    """Verify that hourly and daily state files are separate."""
    from stock.state import get_state_file

    # Save current env
    old_suffix = os.environ.get("TRADE_STATE_SUFFIX")

    try:
        # Daily bot (no suffix)
        os.environ.pop("TRADE_STATE_SUFFIX", None)
        daily_file = get_state_file("trade_history")
        assert "trade_history.json" == daily_file.name

        # Hourly bot (with suffix)
        os.environ["TRADE_STATE_SUFFIX"] = "hourly"
        hourly_file = get_state_file("trade_history")
        assert "trade_history_hourly.json" == hourly_file.name

        # Verify they're different
        assert daily_file != hourly_file

    finally:
        # Restore
        if old_suffix:
            os.environ["TRADE_STATE_SUFFIX"] = old_suffix
        else:
            os.environ.pop("TRADE_STATE_SUFFIX", None)
