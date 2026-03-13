"""Tests for time-aware prompts and stock awareness in crypto prompts."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from unified_orchestrator.state import UnifiedPortfolioSnapshot, Position
from unified_orchestrator.prompt_builder import (
    _build_time_context,
    build_portfolio_context,
    build_unified_prompt,
)


def _make_snapshot(**overrides) -> UnifiedPortfolioSnapshot:
    defaults = dict(
        alpaca_cash=10_000.0,
        alpaca_buying_power=20_000.0,
        alpaca_positions={},
        regime="CRYPTO_ONLY",
        minutes_to_open=None,
        minutes_to_close=None,
    )
    defaults.update(overrides)
    return UnifiedPortfolioSnapshot(**defaults)


def test_time_context_crypto_only():
    """Time context should show day + ET time during CRYPTO_ONLY."""
    snapshot = _make_snapshot(regime="CRYPTO_ONLY")

    # Mock to a known time: Wednesday 2:30 AM ET
    mock_dt = datetime(2026, 3, 11, 2, 30, tzinfo=ZoneInfo("America/New_York"))
    with patch("unified_orchestrator.prompt_builder.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _build_time_context(snapshot)

    assert "Wednesday" in result
    assert "02:30 ET" in result
    assert "Stock market closed" in result


def test_time_context_near_market_open():
    """Time context should show countdown when market opens soon."""
    snapshot = _make_snapshot(regime="CRYPTO_ONLY", minutes_to_open=45)

    mock_dt = datetime(2026, 3, 11, 8, 45, tzinfo=ZoneInfo("America/New_York"))
    with patch("unified_orchestrator.prompt_builder.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _build_time_context(snapshot)

    assert "opens in 45 min" in result


def test_time_context_stock_hours():
    """During stock hours, show time until close."""
    snapshot = _make_snapshot(regime="STOCK_HOURS", minutes_to_close=120)

    mock_dt = datetime(2026, 3, 11, 14, 0, tzinfo=ZoneInfo("America/New_York"))
    with patch("unified_orchestrator.prompt_builder.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _build_time_context(snapshot)

    assert "closes in 120 min" in result


def test_time_context_pre_market():
    """Pre-market shows time to open."""
    snapshot = _make_snapshot(regime="PRE_MARKET", minutes_to_open=15)

    mock_dt = datetime(2026, 3, 11, 9, 15, tzinfo=ZoneInfo("America/New_York"))
    with patch("unified_orchestrator.prompt_builder.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _build_time_context(snapshot)

    assert "opens in 15 min" in result


def test_stock_awareness_in_crypto_prompt():
    """Crypto prompt should show stock positions for cross-asset awareness."""
    snapshot = _make_snapshot(
        regime="CRYPTO_ONLY",
        alpaca_positions={
            "NVDA": Position(
                symbol="NVDA", qty=50.0, avg_price=142.0,
                current_price=145.0, unrealized_pnl=150.0, broker="alpaca",
            ),
        },
    )

    ctx = build_portfolio_context(snapshot)
    # Stock positions should be visible even during crypto hours
    assert "NVDA" in ctx


def test_time_context_in_unified_prompt():
    """build_unified_prompt should include time context."""
    snapshot = _make_snapshot(regime="CRYPTO_ONLY")

    history = [
        {"timestamp": "2026-03-11T02:00", "open": 69000, "high": 69500,
         "low": 68500, "close": 69200, "volume": 100}
    ] * 12

    mock_dt = datetime(2026, 3, 11, 2, 30, tzinfo=ZoneInfo("America/New_York"))
    with patch("unified_orchestrator.prompt_builder.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        prompt = build_unified_prompt(
            symbol="BTCUSD",
            history_rows=history,
            current_price=69200.0,
            snapshot=snapshot,
            asset_class="crypto",
        )

    assert "TIME:" in prompt
    assert "Wednesday" in prompt


def test_5_symbol_feature_building():
    """Verify compute_hourly_features produces correct shape for 5-symbol stacking."""
    import numpy as np
    import pandas as pd
    from pufferlib_market.inference import compute_hourly_features

    # Create 96 hours of dummy OHLCV data
    n = 96
    df = pd.DataFrame({
        "open": np.random.uniform(60000, 70000, n),
        "high": np.random.uniform(70000, 72000, n),
        "low": np.random.uniform(58000, 60000, n),
        "close": np.random.uniform(60000, 70000, n),
        "volume": np.random.uniform(100, 1000, n),
    })

    features = compute_hourly_features(df)
    assert features.shape == (16,)
    assert features.dtype == np.float32

    # Stack 5 symbols
    all_features = np.zeros((5, 16), dtype=np.float32)
    for i in range(5):
        all_features[i] = compute_hourly_features(df)

    assert all_features.shape == (5, 16)
