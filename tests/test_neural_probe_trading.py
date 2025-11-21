"""
Tests for neural daily probe trading logic.

Covers edge cases for per-symbol trade history:
- No history → normal trading
- 1 negative trade → probe mode
- 1 positive trade → normal trading
- 2 trades with negative sum → probe mode
- 2 trades with positive sum → normal trading
- Mix of positive/negative trades
"""
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# We need to set up the state file before importing
@pytest.fixture(autouse=True)
def setup_temp_state(monkeypatch, tmp_path):
    """Setup temporary state directory for all tests."""
    monkeypatch.setenv("STATE_SUFFIX", "test")
    # Mock the state file paths to use temp directory
    with patch("neural_daily_trade_with_probe.TRADE_HISTORY_FILE", tmp_path / "trade_history_test.jsonl"):
        yield


def test_imports():
    """Test that we can import the module."""
    import neural_daily_trade_with_probe
    assert hasattr(neural_daily_trade_with_probe, "_should_use_probe_mode")
    assert hasattr(neural_daily_trade_with_probe, "record_trade_outcome")


def test_no_trade_history_normal_mode():
    """
    Case 1: No trade history
    Expected: Normal trading (no probe mode)
    """
    from neural_daily_trade_with_probe import _should_use_probe_mode
    from src.risk_state import ProbeState

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})

    # No trade history for SPY
    should_probe, reason = _should_use_probe_mode("SPY", "buy", probe_state)

    assert should_probe is False, "Should NOT probe with no history"
    assert reason is None


def test_single_negative_trade_probe_mode():
    """
    Case 2: 1 negative trade in history
    Expected: Probe mode
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record a single losing trade
    record_trade_outcome("AAPL", "buy", pnl=-50.0, pnl_pct=-0.05)  # -5%

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    should_probe, reason = _should_use_probe_mode("AAPL", "buy", probe_state)

    assert should_probe is True, "Should probe with single negative trade"
    assert "single_trade_negative" in reason
    assert "-0.05" in reason


def test_single_positive_trade_normal_mode():
    """
    Case 3: 1 positive trade in history
    Expected: Normal trading
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record a single winning trade
    record_trade_outcome("MSFT", "buy", pnl=100.0, pnl_pct=0.10)  # +10%

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    should_probe, reason = _should_use_probe_mode("MSFT", "buy", probe_state)

    assert should_probe is False, "Should NOT probe with single positive trade"
    assert reason is None


def test_two_negative_trades_probe_mode():
    """
    Case 4: 2 trades, both negative (sum < 0)
    Expected: Probe mode
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record two losing trades
    record_trade_outcome("NVDA", "buy", pnl=-30.0, pnl_pct=-0.03)  # -3%
    record_trade_outcome("NVDA", "buy", pnl=-20.0, pnl_pct=-0.02)  # -2%

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    should_probe, reason = _should_use_probe_mode("NVDA", "buy", probe_state)

    assert should_probe is True, "Should probe with 2 negative trades"
    assert "recent_pnl_sum" in reason
    assert "-0.05" in reason  # Sum is -5%


def test_two_positive_trades_normal_mode():
    """
    Case 5: 2 trades, both positive (sum > 0)
    Expected: Normal trading
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record two winning trades
    record_trade_outcome("TSLA", "buy", pnl=50.0, pnl_pct=0.05)   # +5%
    record_trade_outcome("TSLA", "buy", pnl=30.0, pnl_pct=0.03)   # +3%

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    should_probe, reason = _should_use_probe_mode("TSLA", "buy", probe_state)

    assert should_probe is False, "Should NOT probe with 2 positive trades"
    assert reason is None


def test_mixed_trades_positive_sum_normal_mode():
    """
    Case 6: 2 trades, mixed but sum > 0
    Example: +10% then -5% = +5% sum
    Expected: Normal trading
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record mixed trades with positive sum
    record_trade_outcome("QQQ", "buy", pnl=100.0, pnl_pct=0.10)   # +10%
    record_trade_outcome("QQQ", "buy", pnl=-50.0, pnl_pct=-0.05)  # -5%

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    should_probe, reason = _should_use_probe_mode("QQQ", "buy", probe_state)

    assert should_probe is False, "Should NOT probe when sum is positive"
    assert reason is None


def test_mixed_trades_negative_sum_probe_mode():
    """
    Case 7: 2 trades, mixed but sum < 0
    Example: +3% then -5% = -2% sum
    Expected: Probe mode
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record mixed trades with negative sum
    record_trade_outcome("SPY", "sell", pnl=30.0, pnl_pct=0.03)   # +3%
    record_trade_outcome("SPY", "sell", pnl=-50.0, pnl_pct=-0.05)  # -5%

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    should_probe, reason = _should_use_probe_mode("SPY", "sell", probe_state)

    assert should_probe is True, "Should probe when sum is negative"
    assert "recent_pnl_sum" in reason
    assert "-0.02" in reason  # Sum is -2%


def test_mixed_trades_zero_sum_probe_mode():
    """
    Case 8: 2 trades, sum exactly 0
    Example: +5% then -5% = 0% sum
    Expected: Probe mode (sum <= 0 triggers probe)
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record trades that sum to zero
    record_trade_outcome("GLD", "buy", pnl=50.0, pnl_pct=0.05)   # +5%
    record_trade_outcome("GLD", "buy", pnl=-50.0, pnl_pct=-0.05)  # -5%

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    should_probe, reason = _should_use_probe_mode("GLD", "buy", probe_state)

    assert should_probe is True, "Should probe when sum is exactly zero"
    assert "recent_pnl_sum" in reason
    assert "0.0000" in reason  # Sum is 0%


def test_single_zero_trade_probe_mode():
    """
    Case 9: 1 trade with exactly 0% PnL
    Expected: Probe mode (pnl <= 0 triggers probe)
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record a break-even trade
    record_trade_outcome("BTCUSD", "buy", pnl=0.0, pnl_pct=0.0)

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    should_probe, reason = _should_use_probe_mode("BTCUSD", "buy", probe_state)

    assert should_probe is True, "Should probe with zero PnL trade"
    assert "single_trade_negative" in reason


def test_global_probe_overrides_symbol_history():
    """
    Case 10: Global probe mode active
    Expected: Probe mode regardless of symbol history
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record positive trades
    record_trade_outcome("META", "buy", pnl=100.0, pnl_pct=0.10)
    record_trade_outcome("META", "buy", pnl=50.0, pnl_pct=0.05)

    # Global probe mode active
    probe_state = ProbeState(
        force_probe=True,
        reason="previous_day_loss",
        probe_date=None,
        state={}
    )
    should_probe, reason = _should_use_probe_mode("META", "buy", probe_state)

    assert should_probe is True, "Global probe should override symbol history"
    assert "global:" in reason
    assert "previous_day_loss" in reason


def test_different_sides_tracked_separately():
    """
    Case 11: Buy and sell sides tracked independently
    Expected: Buy side can be probe while sell is normal
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record losing buys
    record_trade_outcome("SHOP", "buy", pnl=-50.0, pnl_pct=-0.05)

    # Record winning sells
    record_trade_outcome("SHOP", "sell", pnl=100.0, pnl_pct=0.10)

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})

    # Buy side should probe
    should_probe_buy, reason_buy = _should_use_probe_mode("SHOP", "buy", probe_state)
    assert should_probe_buy is True

    # Sell side should NOT probe
    should_probe_sell, reason_sell = _should_use_probe_mode("SHOP", "sell", probe_state)
    assert should_probe_sell is False


def test_only_most_recent_two_trades_matter():
    """
    Case 12: With 3+ trades, only last 2 count
    Expected: Older trades don't affect probe decision
    """
    from neural_daily_trade_with_probe import (
        _should_use_probe_mode,
        record_trade_outcome,
    )
    from src.risk_state import ProbeState

    # Record 3 trades - oldest is very negative
    record_trade_outcome("INTC", "buy", pnl=-200.0, pnl_pct=-0.20)  # Old: -20%
    record_trade_outcome("INTC", "buy", pnl=50.0, pnl_pct=0.05)     # Recent: +5%
    record_trade_outcome("INTC", "buy", pnl=30.0, pnl_pct=0.03)     # Most recent: +3%

    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    should_probe, reason = _should_use_probe_mode("INTC", "buy", probe_state)

    # Last 2 trades sum to +8%, so should NOT probe
    assert should_probe is False, "Should only look at last 2 trades"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
