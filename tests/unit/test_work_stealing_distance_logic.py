"""Tests for distance-based work stealing logic."""

from unittest.mock import Mock, patch

import pytest
from src.work_stealing_coordinator import WorkStealingCoordinator


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


class TestDistanceBasedStealing:
    """Test that work stealing prioritizes by distance, not PnL."""

    def test_steals_from_furthest_order_not_worst_pnl(self, coordinator, mock_account):
        """Should steal from furthest order even if it has better PnL."""
        # Order 1: Close to limit, low PnL (should be protected by distance)
        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 150.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 150.5  # 0.33% away (close)

        # Order 2: Far from limit, high PnL (should be stolen from)
        order2 = Mock()
        order2.symbol = "MSFT"
        order2.qty = 10.0
        order2.limit_price = 200.0
        order2.side = "buy"
        order2.id = "order2"
        order2.current_price = 185.0  # 7.5% away (far)

        forecast_data = {
            "AAPL": {"avg_return": 1.0},  # Low PnL
            "MSFT": {"avg_return": 5.0},  # High PnL but far from limit
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1, order2]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        # New order close to execution
                        result = coordinator.attempt_steal(
                            symbol="NVDA",
                            side="buy",
                            limit_price=500.0,
                            qty=5.0,
                            current_price=500.5,  # 0.1% away
                            forecasted_pnl=3.0,  # Medium PnL
                            mode="normal",
                        )

                        # Should steal from MSFT (furthest), not AAPL (worst PnL)
                        assert result == "MSFT"
                        mock_cancel.assert_called_once_with("order2")

    def test_pnl_used_as_tiebreaker_for_equal_distance(self, coordinator, mock_account):
        """When distances equal, worse PnL should be stolen from."""
        # Both orders at same distance
        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 100.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 95.0  # 5% away

        order2 = Mock()
        order2.symbol = "MSFT"
        order2.qty = 10.0
        order2.limit_price = 200.0
        order2.side = "buy"
        order2.id = "order2"
        order2.current_price = 190.0  # 5% away (same)

        forecast_data = {
            "AAPL": {"avg_return": 2.0},  # Better PnL
            "MSFT": {"avg_return": 1.0},  # Worse PnL
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1, order2]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        result = coordinator.attempt_steal(
                            symbol="NVDA",
                            side="buy",
                            limit_price=500.0,
                            qty=5.0,
                            current_price=500.1,
                            forecasted_pnl=3.0,
                            mode="normal",
                        )

                        # Same distance, so steal from worse PnL (MSFT)
                        assert result == "MSFT"
                        mock_cancel.assert_called_once_with("order2")

    def test_no_pnl_requirement_allows_any_steal(self, coordinator, mock_account):
        """Can steal even with worse PnL if order is furthest."""
        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 150.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 140.0  # 6.7% away

        forecast_data = {
            "AAPL": {"avg_return": 10.0},  # Great PnL!
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        # Terrible PnL but close to execution
                        result = coordinator.attempt_steal(
                            symbol="JUNK",
                            side="buy",
                            limit_price=10.0,
                            qty=100.0,
                            current_price=10.01,  # 0.1% away
                            forecasted_pnl=0.1,  # Terrible PnL
                            mode="normal",
                        )

                        # Should still steal because distance is what matters
                        assert result == "AAPL"
                        mock_cancel.assert_called_once()


class TestFightingResolution:
    """Test PnL-based fighting resolution."""

    def test_fighting_resolved_by_better_pnl(self, coordinator, mock_account):
        """Fighting allowed if new order has better PnL."""
        # Setup fighting history
        for i in range(2):  # 2 prior steals (need 3 total to fight)
            coordinator._record_steal(
                from_symbol="AAPL",
                to_symbol="MSFT",
                from_order_id=f"order{i}",
                to_forecasted_pnl=3.0,
                from_forecasted_pnl=2.0,
            )

        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 150.0
        order1.side = "buy"
        order1.id = "order_apple"
        order1.current_price = 140.0  # 6.7% away (furthest)

        forecast_data = {
            "AAPL": {"avg_return": 2.0},  # Lower PnL
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        # MSFT trying to steal again with better PnL
                        result = coordinator.attempt_steal(
                            symbol="MSFT",
                            side="buy",
                            limit_price=200.0,
                            qty=10.0,
                            current_price=200.1,
                            forecasted_pnl=4.0,  # Better than AAPL's 2.0
                            mode="normal",
                        )

                        # Should allow steal because PnL is better
                        assert result == "AAPL"
                        mock_cancel.assert_called_once()

    def test_fighting_blocked_by_worse_pnl(self, coordinator, mock_account):
        """Fighting blocked if new order has worse PnL."""
        # Setup fighting history
        for i in range(2):
            coordinator._record_steal(
                from_symbol="AAPL",
                to_symbol="MSFT",
                from_order_id=f"order{i}",
                to_forecasted_pnl=3.0,
                from_forecasted_pnl=2.0,
            )

        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 150.0
        order1.side = "buy"
        order1.id = "order_apple"
        order1.current_price = 140.0

        forecast_data = {
            "AAPL": {"avg_return": 5.0},  # Higher PnL
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    # MSFT trying to steal with worse PnL
                    result = coordinator.attempt_steal(
                        symbol="MSFT",
                        side="buy",
                        limit_price=200.0,
                        qty=10.0,
                        current_price=200.1,
                        forecasted_pnl=3.0,  # Worse than AAPL's 5.0
                        mode="normal",
                    )

                    # Should block steal because PnL is worse
                    assert result is None


class TestCandidateOrdering:
    """Test that candidates are correctly ordered by distance."""

    def test_candidates_sorted_by_distance_descending(self, coordinator, mock_account):
        """Candidates should be sorted furthest first."""
        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 100.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 99.0  # 1% away

        order2 = Mock()
        order2.symbol = "MSFT"
        order2.qty = 10.0
        order2.limit_price = 200.0
        order2.side = "buy"
        order2.id = "order2"
        order2.current_price = 190.0  # 5% away (furthest)

        order3 = Mock()
        order3.symbol = "NVDA"
        order3.qty = 10.0
        order3.limit_price = 500.0
        order3.side = "buy"
        order3.id = "order3"
        order3.current_price = 485.0  # 3% away

        forecast_data = {
            "AAPL": {"avg_return": 1.0},
            "MSFT": {"avg_return": 2.0},
            "NVDA": {"avg_return": 3.0},
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1, order2, order3]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    candidates = coordinator._get_steal_candidates()

                    # Should be ordered by distance descending
                    assert len(candidates) == 3
                    assert candidates[0].symbol == "MSFT"  # 5% furthest
                    assert candidates[1].symbol == "NVDA"  # 3%
                    assert candidates[2].symbol == "AAPL"  # 1% closest


class TestEdgeCases:
    """Test edge cases in distance-based stealing."""

    def test_zero_distance_not_stolen(self, coordinator, mock_account):
        """Orders at exact limit price should be protected."""
        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 100.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 100.0  # 0% away (at limit!)

        forecast_data = {"AAPL": {"avg_return": 1.0}}

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    candidates = coordinator._get_steal_candidates()

                    # Should be protected (within protection tolerance)
                    assert len(candidates) == 0

    def test_negative_pnl_can_be_stolen(self, coordinator, mock_account):
        """Orders with negative PnL can be stolen if furthest."""
        order1 = Mock()
        order1.symbol = "LOSER"
        order1.qty = 10.0
        order1.limit_price = 100.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 90.0  # 10% away

        forecast_data = {"LOSER": {"avg_return": -5.0}}  # Negative PnL

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        result = coordinator.attempt_steal(
                            symbol="WINNER",
                            side="buy",
                            limit_price=200.0,
                            qty=5.0,
                            current_price=200.1,
                            forecasted_pnl=0.5,  # Positive but small
                            mode="normal",
                        )

                        # Should steal from negative PnL order
                        assert result == "LOSER"

    def test_multiple_orders_same_distance_uses_pnl(self, coordinator, mock_account):
        """Multiple orders at same distance should use PnL tiebreaker."""
        # Three orders all 5% away
        orders = []
        for i, (symbol, pnl) in enumerate([("AAA", 3.0), ("BBB", 1.0), ("CCC", 2.0)]):
            order = Mock()
            order.symbol = symbol
            order.qty = 10.0
            order.limit_price = 100.0
            order.side = "buy"
            order.id = f"order{i}"
            order.current_price = 95.0  # All 5% away
            orders.append(order)

        forecast_data = {
            "AAA": {"avg_return": 3.0},
            "BBB": {"avg_return": 1.0},  # Worst PnL
            "CCC": {"avg_return": 2.0},
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=orders):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        result = coordinator.attempt_steal(
                            symbol="DDD",
                            side="buy",
                            limit_price=200.0,
                            qty=5.0,
                            current_price=200.1,
                            forecasted_pnl=5.0,
                            mode="normal",
                        )

                        # All same distance, should steal from worst PnL (BBB)
                        assert result == "BBB"
