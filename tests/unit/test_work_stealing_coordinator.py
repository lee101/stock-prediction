"""Tests for work stealing coordinator."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest
from src.work_stealing_config import (
    WORK_STEALING_COOLDOWN_SECONDS,
    WORK_STEALING_FIGHT_THRESHOLD,
    WORK_STEALING_PROTECTION_PCT,
)
from src.work_stealing_coordinator import (
    WorkStealingCoordinator,
)


@pytest.fixture
def coordinator():
    """Create a fresh coordinator for each test."""
    return WorkStealingCoordinator()


@pytest.fixture
def mock_account():
    """Mock alpaca account with buying power."""
    account = Mock()
    account.buying_power = 10000.0
    return account


@pytest.fixture
def mock_empty_orders():
    """Mock empty orders list."""
    return []


class TestCapacityCheck:
    """Test capacity checking logic."""

    def test_can_open_order_with_capacity(self, coordinator, mock_account, mock_empty_orders):
        """Should allow order when capacity available."""
        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=mock_empty_orders):
                can_open = coordinator.can_open_order(
                    symbol="AAPL",
                    side="buy",
                    limit_price=150.0,
                    qty=10.0,
                )
                assert can_open is True

    def test_cannot_open_order_without_capacity(self, coordinator, mock_account):
        """Should block order when capacity exceeded."""
        # Mock existing order consuming all capacity
        existing_order = Mock()
        existing_order.qty = 100.0
        existing_order.limit_price = 200.0  # 20k notional

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[existing_order]):
                can_open = coordinator.can_open_order(
                    symbol="AAPL",
                    side="buy",
                    limit_price=150.0,
                    qty=10.0,
                )
                assert can_open is False


class TestProtection:
    """Test order protection logic."""

    def test_probe_trades_always_protected(self, coordinator):
        """Probe trades should never be stolen."""
        protected = coordinator.is_protected(
            symbol="AAPL",
            limit_price=150.0,
            current_price=150.0,
            mode="probe",
        )
        assert protected is True

    def test_close_to_execution_is_protected(self, coordinator):
        """Orders close to execution should be protected."""
        # Within protection tolerance
        limit = 100.0
        current = limit * (1 + WORK_STEALING_PROTECTION_PCT * 0.5)  # Half of protection

        protected = coordinator.is_protected(
            symbol="AAPL",
            limit_price=limit,
            current_price=current,
            mode="normal",
        )
        assert protected is True

    def test_far_from_execution_not_protected(self, coordinator):
        """Orders far from execution can be stolen."""
        # Well outside protection tolerance
        limit = 100.0
        current = limit * (1 + WORK_STEALING_PROTECTION_PCT * 2)

        protected = coordinator.is_protected(
            symbol="AAPL",
            limit_price=limit,
            current_price=current,
            mode="normal",
        )
        assert protected is False


class TestStealAttempt:
    """Test work steal attempt logic."""

    def test_steal_requires_close_price(self, coordinator):
        """Should not steal if price not close enough."""
        # Price far from limit
        result = coordinator.attempt_steal(
            symbol="MSFT",
            side="buy",
            limit_price=100.0,
            qty=10.0,
            current_price=110.0,  # 10% away, exceeds tolerance
            forecasted_pnl=5.0,
            mode="normal",
        )
        assert result is None

    def test_steal_requires_better_pnl(self, coordinator, mock_account):
        """Should not steal unless PnL significantly better."""
        # Mock existing order with similar PnL
        existing_order = Mock()
        existing_order.symbol = "AAPL"
        existing_order.qty = 10.0
        existing_order.limit_price = 150.0
        existing_order.side = "buy"
        existing_order.id = "order123"
        existing_order.current_price = 148.0

        # Mock forecast with PnL
        forecast_data = {
            "AAPL": {"avg_return": 4.0},
            "MSFT": {},
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[existing_order]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    # Try to steal with only marginally better PnL
                    result = coordinator.attempt_steal(
                        symbol="MSFT",
                        side="buy",
                        limit_price=200.0,
                        qty=10.0,
                        current_price=200.1,  # Within tolerance
                        forecasted_pnl=4.2,  # Only 5% better, need 10%+
                        mode="normal",
                    )
                    assert result is None

    def test_successful_steal(self, coordinator, mock_account):
        """Should successfully steal when conditions met."""
        # Mock existing order with poor PnL
        existing_order = Mock()
        existing_order.symbol = "AAPL"
        existing_order.qty = 10.0
        existing_order.limit_price = 150.0
        existing_order.side = "buy"
        existing_order.id = "order123"
        existing_order.current_price = 148.0

        # Mock forecast
        forecast_data = {
            "AAPL": {"avg_return": 2.0},
            "MSFT": {},
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[existing_order]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        # Try to steal with much better PnL
                        result = coordinator.attempt_steal(
                            symbol="MSFT",
                            side="buy",
                            limit_price=200.0,
                            qty=10.0,
                            current_price=200.1,  # Within tolerance
                            forecasted_pnl=5.0,  # 2.5x better
                            mode="normal",
                        )

                        # Should steal from AAPL
                        assert result == "AAPL"
                        mock_cancel.assert_called_once_with("order123")


class TestCooldown:
    """Test cooldown logic."""

    def test_cooldown_prevents_immediate_resteal(self, coordinator, mock_account):
        """Cannot steal from same symbol within cooldown."""
        existing_order = Mock()
        existing_order.symbol = "AAPL"
        existing_order.qty = 10.0
        existing_order.limit_price = 150.0
        existing_order.side = "buy"
        existing_order.id = "order123"
        existing_order.current_price = 148.0

        forecast_data = {"AAPL": {"avg_return": 1.0}}

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[existing_order]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order"):
                        # First steal succeeds
                        result1 = coordinator.attempt_steal(
                            symbol="MSFT",
                            side="buy",
                            limit_price=200.0,
                            qty=10.0,
                            current_price=200.1,
                            forecasted_pnl=5.0,
                            mode="normal",
                        )
                        assert result1 == "AAPL"

                        # Second steal immediately after should fail
                        result2 = coordinator.attempt_steal(
                            symbol="NVDA",
                            side="buy",
                            limit_price=500.0,
                            qty=10.0,
                            current_price=500.1,
                            forecasted_pnl=6.0,
                            mode="normal",
                        )
                        assert result2 is None  # Blocked by cooldown

    def test_cooldown_expires(self, coordinator, mock_account):
        """Cooldown should expire after timeout."""
        existing_order = Mock()
        existing_order.symbol = "AAPL"
        existing_order.qty = 10.0
        existing_order.limit_price = 150.0
        existing_order.side = "buy"
        existing_order.id = "order123"
        existing_order.current_price = 148.0

        forecast_data = {"AAPL": {"avg_return": 1.0}}

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[existing_order]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order"):
                        # First steal
                        coordinator.attempt_steal(
                            symbol="MSFT",
                            side="buy",
                            limit_price=200.0,
                            qty=10.0,
                            current_price=200.1,
                            forecasted_pnl=5.0,
                            mode="normal",
                        )

                        # Fast-forward time past cooldown
                        past_cooldown = datetime.now(timezone.utc) - timedelta(
                            seconds=WORK_STEALING_COOLDOWN_SECONDS + 10
                        )
                        coordinator._cooldown_tracker["AAPL"] = past_cooldown

                        # Now should be able to steal again
                        result = coordinator.attempt_steal(
                            symbol="NVDA",
                            side="buy",
                            limit_price=500.0,
                            qty=10.0,
                            current_price=500.1,
                            forecasted_pnl=6.0,
                            mode="normal",
                        )
                        assert result == "AAPL"


class TestFightingPrevention:
    """Test fighting detection and prevention."""

    def test_fighting_detected(self, coordinator, mock_account):
        """Should detect when two symbols fight."""
        # Simulate multiple steals between same pair
        now = datetime.now(timezone.utc)

        for i in range(WORK_STEALING_FIGHT_THRESHOLD - 1):
            coordinator._fight_tracker[("AAPL", "MSFT")] = [
                now - timedelta(seconds=i * 60) for i in range(WORK_STEALING_FIGHT_THRESHOLD - 1)
            ]

        # Next steal attempt should be blocked
        would_fight = coordinator._would_cause_fight("AAPL", "MSFT")
        assert would_fight is True

    def test_no_fighting_with_different_symbols(self, coordinator):
        """Different symbols should not trigger fighting."""
        would_fight = coordinator._would_cause_fight("AAPL", "NVDA")
        assert would_fight is False

    def test_old_steals_dont_count_as_fighting(self, coordinator):
        """Old steals outside window shouldn't count."""
        from src.work_stealing_config import WORK_STEALING_FIGHT_WINDOW_SECONDS

        # Steals from long ago
        old_time = datetime.now(timezone.utc) - timedelta(seconds=WORK_STEALING_FIGHT_WINDOW_SECONDS + 100)

        coordinator._fight_tracker[("AAPL", "MSFT")] = [old_time for _ in range(WORK_STEALING_FIGHT_THRESHOLD)]

        # Should not be considered fighting
        would_fight = coordinator._would_cause_fight("AAPL", "MSFT")
        assert would_fight is False


class TestStealCandidateSelection:
    """Test selection of steal candidates."""

    def test_selects_worst_pnl_order(self, coordinator, mock_account):
        """Should steal from worst PnL order."""
        # Multiple orders with different PnLs
        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 150.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 145.0

        order2 = Mock()
        order2.symbol = "MSFT"
        order2.qty = 10.0
        order2.limit_price = 200.0
        order2.side = "buy"
        order2.id = "order2"
        order2.current_price = 195.0

        forecast_data = {
            "AAPL": {"avg_return": 1.0},  # Worst
            "MSFT": {"avg_return": 3.0},  # Better
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1, order2]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    candidates = coordinator._get_steal_candidates()

                    # Should be sorted by PnL ascending (worst first)
                    assert len(candidates) > 0
                    assert candidates[0].symbol == "AAPL"

    def test_excludes_protected_orders(self, coordinator, mock_account):
        """Should exclude protected orders from candidates."""
        # Order close to execution (protected)
        protected_order = Mock()
        protected_order.symbol = "AAPL"
        protected_order.qty = 10.0
        protected_order.limit_price = 100.0
        protected_order.side = "buy"
        protected_order.id = "order1"
        protected_order.current_price = 100.2  # Within protection tolerance

        # Order far from execution (not protected)
        normal_order = Mock()
        normal_order.symbol = "MSFT"
        normal_order.qty = 10.0
        normal_order.limit_price = 100.0
        normal_order.side = "buy"
        normal_order.id = "order2"
        normal_order.current_price = 110.0  # Far from limit

        forecast_data = {
            "AAPL": {"avg_return": 1.0},
            "MSFT": {"avg_return": 2.0},
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[protected_order, normal_order]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    candidates = coordinator._get_steal_candidates()

                    # Should only include MSFT (not protected)
                    symbols = [c.symbol for c in candidates]
                    assert "MSFT" in symbols
                    assert "AAPL" not in symbols


class TestDryRunMode:
    """Test dry run mode."""

    def test_dry_run_doesnt_cancel_orders(self, coordinator, mock_account, monkeypatch):
        """Dry run should not actually cancel orders."""
        monkeypatch.setenv("WORK_STEALING_DRY_RUN", "1")

        # Reload module to pick up env
        import importlib

        import src.work_stealing_coordinator as coordinator_module

        importlib.reload(coordinator_module)

        dry_coordinator = coordinator_module.WorkStealingCoordinator()

        existing_order = Mock()
        existing_order.symbol = "AAPL"
        existing_order.qty = 10.0
        existing_order.limit_price = 150.0
        existing_order.side = "buy"
        existing_order.id = "order123"
        existing_order.current_price = 148.0

        forecast_data = {"AAPL": {"avg_return": 1.0}}

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[existing_order]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        result = dry_coordinator.attempt_steal(
                            symbol="MSFT",
                            side="buy",
                            limit_price=200.0,
                            qty=10.0,
                            current_price=200.1,
                            forecasted_pnl=5.0,
                            mode="normal",
                        )

                        # Should return symbol but not actually cancel
                        assert result == "AAPL"
                        mock_cancel.assert_not_called()
