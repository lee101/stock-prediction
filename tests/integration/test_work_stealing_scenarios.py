"""Integration tests for complex work stealing scenarios."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import pytz
from src.work_stealing_coordinator import get_coordinator


@pytest.fixture(autouse=True)
def reset_coordinator():
    """Reset coordinator state between tests."""
    coordinator = get_coordinator()
    coordinator._steal_history.clear()
    coordinator._cooldown_tracker.clear()
    coordinator._fight_tracker.clear()
    yield


class TestRapidPriceMovements:
    """Test work stealing under rapid price changes."""

    def test_orders_become_protected_as_price_approaches(self):
        """Orders should become protected as price gets close."""
        coordinator = get_coordinator()
        mock_account = Mock()
        mock_account.buying_power = 10000.0

        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 100.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 95.0  # 5% away - not protected

        forecast_data = {"AAPL": {"avg_return": 2.0}}

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    # First check - should be stealable
                    candidates_before = coordinator._get_steal_candidates()
                    assert len(candidates_before) == 1

                    # Price moves close to limit
                    order1.current_price = 100.3  # 0.3% away - protected!

                    # Second check - should be protected
                    candidates_after = coordinator._get_steal_candidates()
                    assert len(candidates_after) == 0  # Protected now

    def test_new_order_closer_than_existing_can_steal(self):
        """If new order gets closer than existing, can steal."""
        coordinator = get_coordinator()
        mock_account = Mock()
        mock_account.buying_power = 5000.0

        # Existing order moderately close
        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 100.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 98.0  # 2% away

        forecast_data = {"AAPL": {"avg_return": 3.0}}

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        # New order very close
                        result = coordinator.attempt_steal(
                            symbol="MSFT",
                            side="buy",
                            limit_price=200.0,
                            qty=5.0,
                            current_price=200.1,  # 0.05% away - very close!
                            forecasted_pnl=1.0,  # Even worse PnL
                            mode="normal",
                        )

                        # Should steal because much closer to execution
                        assert result == "AAPL"


class TestMixedCryptoStock:
    """Test work stealing with both crypto and stocks."""

    def test_crypto_out_of_hours_vs_stock_during_hours(self):
        """Crypto out-of-hours and stock should compete fairly."""
        coordinator = get_coordinator()
        mock_account = Mock()
        mock_account.buying_power = 5000.0

        # Stock order during market hours
        stock_order = Mock()
        stock_order.symbol = "AAPL"
        stock_order.qty = 10.0
        stock_order.limit_price = 150.0
        stock_order.side = "buy"
        stock_order.id = "stock1"
        stock_order.current_price = 145.0  # 3.3% away

        # Crypto order out of hours (weekend)
        crypto_order = Mock()
        crypto_order.symbol = "BTCUSD"
        crypto_order.qty = 0.1
        crypto_order.limit_price = 50000.0
        crypto_order.side = "buy"
        crypto_order.id = "crypto1"
        crypto_order.current_price = 48000.0  # 4% away (furthest)

        forecast_data = {
            "AAPL": {"avg_return": 2.0},
            "BTCUSD": {"avg_return": 4.0},  # Better PnL but further
        }

        est = pytz.timezone("US/Eastern")
        weekend_dt = est.localize(datetime(2025, 1, 18, 10, 0, 0))  # Saturday

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[stock_order, crypto_order]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        with patch("src.work_stealing_config.datetime") as mock_dt:
                            mock_dt.now.return_value = weekend_dt

                            # New entry close to execution
                            result = coordinator.attempt_steal(
                                symbol="ETHUSD",
                                side="buy",
                                limit_price=3000.0,
                                qty=1.0,
                                current_price=3005.0,  # 0.17% away
                                forecasted_pnl=3.0,
                                mode="normal",
                            )

                            # Should steal from furthest (BTCUSD)
                            assert result == "BTCUSD"

    def test_multiple_cryptos_compete_for_slots(self):
        """Multiple cryptos should compete based on distance."""
        coordinator = get_coordinator()
        mock_account = Mock()
        mock_account.buying_power = 5000.0

        # BTC far from limit
        btc_order = Mock()
        btc_order.symbol = "BTCUSD"
        btc_order.qty = 0.1
        btc_order.limit_price = 50000.0
        btc_order.side = "buy"
        btc_order.id = "btc1"
        btc_order.current_price = 48000.0  # 4% away

        # ETH closer
        eth_order = Mock()
        eth_order.symbol = "ETHUSD"
        eth_order.qty = 1.0
        eth_order.limit_price = 3000.0
        eth_order.side = "buy"
        eth_order.id = "eth1"
        eth_order.current_price = 2940.0  # 2% away

        forecast_data = {
            "BTCUSD": {"avg_return": 5.0},  # Best PnL but furthest
            "ETHUSD": {"avg_return": 3.0},
            "UNIUSD": {},
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[btc_order, eth_order]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        # UNI very close to execution
                        result = coordinator.attempt_steal(
                            symbol="UNIUSD",
                            side="buy",
                            limit_price=10.0,
                            qty=100.0,
                            current_price=10.01,  # 0.1% away
                            forecasted_pnl=2.0,  # Worst PnL
                            mode="normal",
                        )

                        # Should steal from BTC (furthest)
                        assert result == "BTCUSD"


class TestCapacityDynamics:
    """Test dynamic capacity changes."""

    def test_capacity_freed_by_partial_fill(self):
        """When order partially fills, capacity should increase."""
        coordinator = get_coordinator()
        mock_account = Mock()
        mock_account.buying_power = 10000.0

        # Initially no orders
        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[]):
                capacity_before = coordinator._get_available_capacity()
                assert capacity_before == 20000.0  # 2x leverage

                # After order placed (simulated)
                mock_order = Mock()
                mock_order.symbol = "AAPL"
                mock_order.qty = 50.0
                mock_order.limit_price = 200.0  # 10k notional
                mock_order.side = "buy"
                mock_order.id = "order1"
                mock_order.current_price = 195.0

                with patch("alpaca_wrapper.get_orders", return_value=[mock_order]):
                    capacity_after = coordinator._get_available_capacity()
                    assert capacity_after == 10000.0  # 20k - 10k = 10k left

    def test_multiple_steals_in_sequence(self):
        """Multiple steals should be possible as capacity frees up."""
        coordinator = get_coordinator()
        mock_account = Mock()
        mock_account.buying_power = 10000.0

        orders = []
        for i, symbol in enumerate(["AAPL", "MSFT", "NVDA"]):
            order = Mock()
            order.symbol = symbol
            order.qty = 10.0
            order.limit_price = 200.0
            order.side = "buy"
            order.id = f"order{i}"
            order.current_price = 190.0 - (i * 5)  # Varying distances
            orders.append(order)

        forecast_data = {
            "AAPL": {"avg_return": 1.0},
            "MSFT": {"avg_return": 2.0},
            "NVDA": {"avg_return": 3.0},
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                # First steal
                with patch("alpaca_wrapper.get_orders", return_value=orders):
                    with patch("alpaca_wrapper.cancel_order"):
                        result1 = coordinator.attempt_steal(
                            symbol="FIRST",
                            side="buy",
                            limit_price=100.0,
                            qty=10.0,
                            current_price=100.1,
                            forecasted_pnl=4.0,
                            mode="normal",
                        )
                        assert result1 is not None

                        # Remove stolen order
                        orders = [o for o in orders if o.id != result1]

                        # Clear cooldown to allow second steal
                        coordinator._cooldown_tracker.clear()

                        # Second steal
                        with patch("alpaca_wrapper.get_orders", return_value=orders):
                            result2 = coordinator.attempt_steal(
                                symbol="SECOND",
                                side="buy",
                                limit_price=150.0,
                                qty=10.0,
                                current_price=150.1,
                                forecasted_pnl=5.0,
                                mode="normal",
                            )
                            assert result2 is not None


class TestProtectionScenarios:
    """Test various protection scenarios."""

    def test_probe_trade_never_stolen_even_if_far(self):
        """Probe trades immune to stealing regardless of distance."""
        coordinator = get_coordinator()
        mock_account = Mock()
        mock_account.buying_power = 10000.0

        # Probe order very far from limit
        probe_order = Mock()
        probe_order.symbol = "AAPL"
        probe_order.qty = 1.0
        probe_order.limit_price = 150.0
        probe_order.side = "buy"
        probe_order.id = "probe1"
        probe_order.current_price = 100.0  # 33% away! (very far)

        forecast_data = {"AAPL": {"avg_return": 0.1}}

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[probe_order]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    candidates = coordinator._get_steal_candidates()

                    # Should be excluded (probe detected by small notional)
                    probe_candidates = [c for c in candidates if c.symbol == "AAPL" and c.mode == "probe"]
                    assert len(probe_candidates) == 0

    def test_recently_stolen_protected_by_cooldown(self):
        """Recently stolen symbols protected by cooldown."""
        coordinator = get_coordinator()
        mock_account = Mock()
        mock_account.buying_power = 10000.0

        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 150.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 140.0

        forecast_data = {"AAPL": {"avg_return": 2.0}}

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order"):
                        # First steal
                        result1 = coordinator.attempt_steal(
                            symbol="MSFT",
                            side="buy",
                            limit_price=200.0,
                            qty=10.0,
                            current_price=200.1,
                            forecasted_pnl=3.0,
                            mode="normal",
                        )
                        assert result1 == "AAPL"

                        # Try to steal immediately again (different symbol)
                        result2 = coordinator.attempt_steal(
                            symbol="NVDA",
                            side="buy",
                            limit_price=500.0,
                            qty=5.0,
                            current_price=500.1,
                            forecasted_pnl=4.0,
                            mode="normal",
                        )
                        # Should fail - AAPL on cooldown
                        assert result2 is None


class TestExtremeScenarios:
    """Test extreme or unusual scenarios."""

    def test_all_orders_very_close_none_stolen(self):
        """If all orders very close to execution, none should be stolen."""
        coordinator = get_coordinator()
        mock_account = Mock()
        mock_account.buying_power = 5000.0

        # All orders within protection tolerance
        orders = []
        for i, symbol in enumerate(["AAPL", "MSFT", "NVDA"]):
            order = Mock()
            order.symbol = symbol
            order.qty = 10.0
            order.limit_price = 100.0
            order.side = "buy"
            order.id = f"order{i}"
            order.current_price = 100.2  # 0.2% away (protected)
            orders.append(order)

        forecast_data = {
            "AAPL": {"avg_return": 1.0},
            "MSFT": {"avg_return": 2.0},
            "NVDA": {"avg_return": 3.0},
        }

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=orders):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    result = coordinator.attempt_steal(
                        symbol="GOOGL",
                        side="buy",
                        limit_price=150.0,
                        qty=10.0,
                        current_price=150.1,
                        forecasted_pnl=5.0,
                        mode="normal",
                    )

                    # Can't steal - all protected
                    assert result is None

    def test_single_order_far_away_gets_stolen(self):
        """Single order far from limit should be stolen."""
        coordinator = get_coordinator()
        mock_account = Mock()
        mock_account.buying_power = 5000.0

        order1 = Mock()
        order1.symbol = "AAPL"
        order1.qty = 10.0
        order1.limit_price = 100.0
        order1.side = "buy"
        order1.id = "order1"
        order1.current_price = 80.0  # 20% away!

        forecast_data = {"AAPL": {"avg_return": 10.0}}  # Great PnL but very far

        with patch("alpaca_wrapper.get_account", return_value=mock_account):
            with patch("alpaca_wrapper.get_orders", return_value=[order1]):
                with patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value=forecast_data):
                    with patch("alpaca_wrapper.cancel_order") as mock_cancel:
                        result = coordinator.attempt_steal(
                            symbol="MSFT",
                            side="buy",
                            limit_price=200.0,
                            qty=5.0,
                            current_price=200.1,
                            forecasted_pnl=1.0,  # Much worse PnL
                            mode="normal",
                        )

                        # Should still steal - distance is what matters
                        assert result == "AAPL"
