"""
Critical safety tests for price spread validation.
These tests prevent catastrophic bugs like inverted buy/sell prices.
"""
import pytest
from unittest.mock import Mock, patch
from hourlycrypto import trade_stock_crypto_hourly as module
from hourlycrypto.trade_stock_crypto_hourly import TradingPlan
import pandas as pd


class TestPriceSpreadSafety:
    """Test suite to prevent inverted or insufficient price spreads."""

    def test_sell_price_must_exceed_buy_price(self, monkeypatch):
        """CRITICAL: Sell price must be higher than buy price."""
        # Mock dependencies
        monkeypatch.setattr(module, "_available_cash", lambda: 10000.0)
        monkeypatch.setattr(module, "_current_inventory", lambda sym: 0.0)

        spawn_calls = []
        def track_spawn(*args, **kwargs):
            spawn_calls.append(("open", args, kwargs))
        monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", track_spawn)
        monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", track_spawn)

        # Create plan with INVERTED prices (sell < buy) - should be blocked!
        plan = TradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=100.0,
            sell_price=99.0,  # Lower than buy - INVALID!
            buy_amount=1.0,
            sell_amount=0.0,
        )

        # Call should return early without spawning any watchers
        module._spawn_watchers(plan, dry_run=False, symbol="BTCUSD")

        # No watchers should be spawned
        assert len(spawn_calls) == 0, "Inverted prices should block all trades!"

    def test_minimum_spread_enforcement_3bp(self, monkeypatch):
        """CRITICAL: Spread must be at least 3 basis points (0.03%)."""
        monkeypatch.setattr(module, "_available_cash", lambda: 10000.0)
        monkeypatch.setattr(module, "_current_inventory", lambda sym: 0.0)

        spawn_calls = []
        def track_spawn(*args, **kwargs):
            spawn_calls.append(("open", args, kwargs))
        monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", track_spawn)
        monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", track_spawn)

        # Spread of 0.02% - below minimum of 0.03%
        plan = TradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=100.0,
            sell_price=100.02,  # Only 0.02% spread - INSUFFICIENT!
            buy_amount=1.0,
            sell_amount=0.0,
        )

        module._spawn_watchers(plan, dry_run=False, symbol="BTCUSD")
        assert len(spawn_calls) == 0, "Insufficient spread should block trades!"

    def test_valid_spread_allows_trading(self, monkeypatch):
        """Valid spread (>= 3bp) should allow trading."""
        monkeypatch.setattr(module, "_available_cash", lambda: 10000.0)
        monkeypatch.setattr(module, "_current_inventory", lambda sym: 0.0)
        monkeypatch.setattr(module, "_cancel_existing_orders", lambda sym: None)

        spawn_open_calls = []
        spawn_close_calls = []

        def track_open(*args, **kwargs):
            spawn_open_calls.append((args, kwargs))

        def track_close(*args, **kwargs):
            spawn_close_calls.append((args, kwargs))

        monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", track_open)
        monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", track_close)

        # Valid spread of 0.89% (much higher than 0.03% minimum)
        plan = TradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=92063.91,
            sell_price=92884.20,  # 0.89% spread - VALID!
            buy_amount=0.7578,
            sell_amount=0.0,
        )

        module._spawn_watchers(plan, dry_run=False, symbol="BTCUSD")

        # Buy watcher should be spawned
        assert len(spawn_open_calls) == 1
        assert spawn_open_calls[0][0][1] == "buy", "Buy side should be 'buy'"

    def test_sell_side_uses_correct_side_parameter(self, monkeypatch):
        """CRITICAL: When selling, must pass 'sell' not 'buy' to spawn function."""
        monkeypatch.setattr(module, "_available_cash", lambda: 10000.0)
        monkeypatch.setattr(module, "_current_inventory", lambda sym: 1.0)  # Have inventory to sell
        monkeypatch.setattr(module, "_cancel_existing_orders", lambda sym: None)

        spawn_close_calls = []

        def track_close(*args, **kwargs):
            spawn_close_calls.append((args, kwargs))

        monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", lambda *a, **k: None)
        monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", track_close)

        # Valid spread, want to sell
        plan = TradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=92063.91,
            sell_price=92884.20,
            buy_amount=0.0,
            sell_amount=1.0,  # Want to sell
        )

        module._spawn_watchers(plan, dry_run=False, symbol="BTCUSD")

        # Verify sell watcher was spawned with correct side
        assert len(spawn_close_calls) == 1
        symbol, side, price = spawn_close_calls[0][0][:3]
        assert side == "sell", f"When selling, side must be 'sell', not '{side}'!"
        assert price == 92884.20, "Should use sell_price for selling"

    def test_buy_side_uses_correct_side_parameter(self, monkeypatch):
        """CRITICAL: When buying, must pass 'buy' to spawn function."""
        monkeypatch.setattr(module, "_available_cash", lambda: 100000.0)
        monkeypatch.setattr(module, "_current_inventory", lambda sym: 0.0)
        monkeypatch.setattr(module, "_cancel_existing_orders", lambda sym: None)

        spawn_open_calls = []

        def track_open(*args, **kwargs):
            spawn_open_calls.append((args, kwargs))

        monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", track_open)
        monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", lambda *a, **k: None)

        plan = TradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=92063.91,
            sell_price=92884.20,
            buy_amount=1.0,  # Want to buy
            sell_amount=0.0,
        )

        module._spawn_watchers(plan, dry_run=False, symbol="BTCUSD")

        # Verify buy watcher was spawned with correct side
        assert len(spawn_open_calls) == 1
        symbol, side, price = spawn_open_calls[0][0][:3]
        assert side == "buy", f"When buying, side must be 'buy', not '{side}'!"
        assert price == 92063.91, "Should use buy_price for buying"

    def test_equal_prices_blocked(self, monkeypatch):
        """CRITICAL: Buy and sell prices equal should be blocked."""
        monkeypatch.setattr(module, "_available_cash", lambda: 10000.0)
        monkeypatch.setattr(module, "_current_inventory", lambda sym: 0.0)

        spawn_calls = []
        monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit",
                           lambda *a, **k: spawn_calls.append("open"))
        monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit",
                           lambda *a, **k: spawn_calls.append("close"))

        plan = TradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=100.0,
            sell_price=100.0,  # Equal prices - INVALID!
            buy_amount=1.0,
            sell_amount=0.0,
        )

        module._spawn_watchers(plan, dry_run=False, symbol="BTCUSD")
        assert len(spawn_calls) == 0, "Equal prices should block all trades!"

    @pytest.mark.parametrize("buy_price,sell_price,expected_valid", [
        (100.0, 100.031, True),   # Slightly above 3bp - should pass
        (100.0, 100.029, False), # Just under 3bp - should fail
        (100.0, 100.5, True),    # 50bp - should pass
        (100.0, 99.0, False),    # Inverted - should fail
        (92063.91, 92092.00, True),   # Above 3bp for BTC - should pass
        (92063.91, 92091.51, False),  # Just under 3bp for BTC - should fail
        (92063.91, 92884.20, True),   # 89bp for BTC - should pass
    ])
    def test_spread_validation_edge_cases(self, buy_price, sell_price, expected_valid, monkeypatch):
        """Test various spread scenarios to ensure correct validation."""
        monkeypatch.setattr(module, "_available_cash", lambda: 100000.0)
        monkeypatch.setattr(module, "_current_inventory", lambda sym: 0.0)
        monkeypatch.setattr(module, "_cancel_existing_orders", lambda sym: None)

        spawn_calls = []
        monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit",
                           lambda *a, **k: spawn_calls.append("open"))
        monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit",
                           lambda *a, **k: spawn_calls.append("close"))

        plan = TradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=buy_price,
            sell_price=sell_price,
            buy_amount=1.0,
            sell_amount=0.0,
        )

        module._spawn_watchers(plan, dry_run=False, symbol="BTCUSD")

        if expected_valid:
            assert len(spawn_calls) > 0, f"Spread {sell_price-buy_price:.4f} should be valid"
        else:
            assert len(spawn_calls) == 0, f"Spread {sell_price-buy_price:.4f} should be invalid"


class TestBuySellSideCorrectness:
    """Ensure buy/sell sides are never flipped."""

    def test_regression_sell_side_bug(self, monkeypatch):
        """
        REGRESSION TEST: Prevent the catastrophic bug where 'buy' was
        hardcoded when spawning sell watchers, causing $2,791 loss.
        """
        monkeypatch.setattr(module, "_available_cash", lambda: 10000.0)
        monkeypatch.setattr(module, "_current_inventory", lambda sym: 1.0)
        monkeypatch.setattr(module, "_cancel_existing_orders", lambda sym: None)

        captured_calls = []

        def capture_close(symbol, side, price, **kwargs):
            captured_calls.append({
                "function": "close",
                "symbol": symbol,
                "side": side,
                "price": price,
            })

        monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", lambda *a, **k: None)
        monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", capture_close)

        plan = TradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=92063.91,
            sell_price=92884.20,
            buy_amount=0.0,
            sell_amount=1.0,
        )

        module._spawn_watchers(plan, dry_run=False, symbol="BTCUSD")

        assert len(captured_calls) == 1
        call = captured_calls[0]

        # THE CRITICAL ASSERTION THAT WOULD HAVE CAUGHT THE BUG
        assert call["side"] == "sell", (
            f"CRITICAL BUG: When selling, side must be 'sell', not '{call['side']}'! "
            f"This bug caused $2,791 loss by buying at sell_price and selling at buy_price."
        )
        assert call["price"] == 92884.20, "Must use sell_price when selling"
