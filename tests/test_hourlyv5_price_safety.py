"""
Critical safety tests for V5 hourly trading price validation.
These tests prevent catastrophic bugs like inverted buy/sell prices.

CRITICAL: These tests must NEVER be disabled or skipped.
A failure here means real money is at risk.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
import pandas as pd
import torch


@dataclass
class MockTradingPlan:
    """Mock TradingPlan for testing."""
    timestamp: pd.Timestamp
    buy_price: float
    sell_price: float
    position_length: int
    position_size: float


class TestV5PriceSafetyValidation:
    """Test that price validation blocks dangerous trades."""

    @pytest.fixture
    def mock_alpaca(self):
        """Mock Alpaca account with sufficient funds."""
        with patch('trade_hourlyv5.alpaca_wrapper') as mock:
            account = Mock()
            account.cash = "100000.0"
            mock.get_account.return_value = account
            yield mock

    @pytest.fixture
    def mock_spawn(self):
        """Mock spawn function to track calls."""
        with patch('trade_hourlyv5.spawn_open_position_at_maxdiff_takeprofit') as mock:
            mock.call_count = 0
            yield mock

    @pytest.fixture
    def mock_price_guard(self):
        """Mock price guard to return adjusted prices."""
        with patch('trade_hourlyv5.enforce_gap') as mock:
            # By default, return same prices (no adjustment)
            mock.side_effect = lambda sym, buy, sell: (buy, sell)
            yield mock

    @pytest.fixture
    def mock_min_notional(self):
        """Mock minimum notional check."""
        with patch('trade_hourlyv5._get_min_order_notional') as mock:
            mock.return_value = 1.0  # Low minimum
            yield mock

    @pytest.fixture
    def mock_record_buy(self):
        """Mock record_buy to avoid file I/O."""
        with patch('trade_hourlyv5.record_buy') as mock:
            yield mock

    def test_inverted_prices_blocked(self, mock_alpaca, mock_spawn, mock_price_guard,
                                      mock_min_notional, mock_record_buy):
        """CRITICAL: Inverted prices (buy >= sell) must be blocked."""
        from trade_hourlyv5 import execute_trade

        # Create plan with INVERTED prices
        plan = MockTradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=100.0,
            sell_price=99.0,  # INVERTED: sell < buy
            position_length=10,
            position_size=0.5,
        )

        result = execute_trade("LINKUSD", plan, dry_run=False)

        assert result is False, "Inverted prices MUST be blocked!"
        assert mock_spawn.call_count == 0, "No orders should be spawned with inverted prices!"

    def test_equal_prices_blocked(self, mock_alpaca, mock_spawn, mock_price_guard,
                                   mock_min_notional, mock_record_buy):
        """CRITICAL: Equal prices (buy == sell) must be blocked."""
        from trade_hourlyv5 import execute_trade

        plan = MockTradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=100.0,
            sell_price=100.0,  # EQUAL: guaranteed loss after fees
            position_length=10,
            position_size=0.5,
        )

        result = execute_trade("LINKUSD", plan, dry_run=False)

        assert result is False, "Equal prices MUST be blocked!"
        assert mock_spawn.call_count == 0

    def test_spread_below_fee_threshold_blocked(self, mock_alpaca, mock_spawn, mock_price_guard,
                                                 mock_min_notional, mock_record_buy):
        """CRITICAL: Spread smaller than 2x maker fee (16bps) must be blocked."""
        from trade_hourlyv5 import execute_trade

        # Spread of 0.1% = 10bps, below 16bps minimum
        plan = MockTradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=100.0,
            sell_price=100.10,  # Only 0.1% spread
            position_length=10,
            position_size=0.5,
        )

        result = execute_trade("LINKUSD", plan, dry_run=False)

        assert result is False, "Spread below fee threshold MUST be blocked!"
        assert mock_spawn.call_count == 0

    def test_valid_spread_allowed(self, mock_alpaca, mock_spawn, mock_price_guard,
                                   mock_min_notional, mock_record_buy):
        """Valid spread (well above fees) should allow trading."""
        from trade_hourlyv5 import execute_trade

        # Spread of 3% - well above minimum
        plan = MockTradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=100.0,
            sell_price=103.0,  # 3% spread - valid
            position_length=10,
            position_size=0.5,
        )

        result = execute_trade("LINKUSD", plan, dry_run=False)

        assert result is True, "Valid spread should be allowed"
        assert mock_spawn.call_count == 1, "Order should be spawned"

    def test_price_adjustment_still_validates(self, mock_alpaca, mock_spawn, mock_price_guard,
                                               mock_min_notional, mock_record_buy):
        """After price adjustment, must re-validate buy < sell."""
        from trade_hourlyv5 import execute_trade

        # Mock price_guard to return INVERTED prices after adjustment
        mock_price_guard.side_effect = lambda sym, buy, sell: (sell, buy)  # Swap them!

        plan = MockTradingPlan(
            timestamp=pd.Timestamp.now(),
            buy_price=100.0,
            sell_price=103.0,  # Valid initially
            position_length=10,
            position_size=0.5,
        )

        result = execute_trade("LINKUSD", plan, dry_run=False)

        assert result is False, "Must re-validate after price adjustment!"
        assert mock_spawn.call_count == 0


class TestV5ModelPriceSafety:
    """Test that the model architecture guarantees buy < sell."""

    def test_model_offset_structure_guarantees_buy_below_sell(self):
        """
        The V5 model uses offset-based pricing:
        - buy_price = ref * (1 - buy_offset) where buy_offset >= 0.0008
        - sell_price = ref * (1 + sell_offset) where sell_offset >= 0.0008

        This structure MATHEMATICALLY guarantees buy < sell.
        """
        from neuralhourlytradingv5.model import HourlyCryptoPolicyV5
        from neuralhourlytradingv5.config import PolicyConfigV5

        config = PolicyConfigV5(input_dim=19)
        model = HourlyCryptoPolicyV5(config)
        model.eval()

        # Generate random inputs
        batch_size = 100
        seq_len = 168
        features = torch.randn(batch_size, seq_len, config.input_dim)
        ref_prices = torch.abs(torch.randn(batch_size)) * 1000 + 10  # Random prices $10-$1010

        with torch.no_grad():
            outputs = model(features)
            actions = model.get_hard_actions(outputs, ref_prices)

        buy_prices = actions['buy_price']
        sell_prices = actions['sell_price']

        # CRITICAL ASSERTION: buy < sell for ALL samples
        assert (buy_prices < sell_prices).all(), (
            f"Model produced inverted prices! "
            f"buy_prices={buy_prices}, sell_prices={sell_prices}"
        )

        # Also verify minimum spread (2 * min_offset = 16bps)
        spreads = (sell_prices - buy_prices) / buy_prices
        min_expected_spread = 2 * config.min_price_offset_pct  # 16bps

        assert (spreads >= min_expected_spread * 0.99).all(), (  # 1% tolerance for float
            f"Model spread too small: min={spreads.min():.6f}, expected>={min_expected_spread}"
        )

    def test_model_extreme_inputs_still_safe(self):
        """Model must be safe even with extreme input values."""
        from neuralhourlytradingv5.model import HourlyCryptoPolicyV5
        from neuralhourlytradingv5.config import PolicyConfigV5

        config = PolicyConfigV5(input_dim=19)
        model = HourlyCryptoPolicyV5(config)
        model.eval()

        batch_size = 50
        seq_len = 168

        # Test with extreme values
        test_cases = [
            torch.zeros(batch_size, seq_len, config.input_dim),  # All zeros
            torch.ones(batch_size, seq_len, config.input_dim) * 1e6,  # Very large
            torch.ones(batch_size, seq_len, config.input_dim) * -1e6,  # Very negative
            torch.randn(batch_size, seq_len, config.input_dim) * 100,  # High variance
        ]

        for features in test_cases:
            ref_prices = torch.ones(batch_size) * 100  # $100 reference

            with torch.no_grad():
                outputs = model(features)
                actions = model.get_hard_actions(outputs, ref_prices)

            buy_prices = actions['buy_price']
            sell_prices = actions['sell_price']

            # MUST always maintain buy < sell
            assert (buy_prices < sell_prices).all(), "Extreme inputs broke price safety!"

            # Prices must be positive
            assert (buy_prices > 0).all(), "Buy prices must be positive!"
            assert (sell_prices > 0).all(), "Sell prices must be positive!"


class TestV5EndToEnd:
    """End-to-end tests for the complete trading flow."""

    def test_full_trading_flow_price_safety(self):
        """Test the complete flow from model to execution."""
        from neuralhourlytradingv5.model import HourlyCryptoPolicyV5
        from neuralhourlytradingv5.config import PolicyConfigV5

        # Create model
        config = PolicyConfigV5(input_dim=19)
        model = HourlyCryptoPolicyV5(config)
        model.eval()

        # Simulate trading decisions
        n_decisions = 1000
        features = torch.randn(n_decisions, 168, config.input_dim)
        ref_prices = torch.abs(torch.randn(n_decisions)) * 50000 + 100  # $100 - $50100

        with torch.no_grad():
            outputs = model(features)
            actions = model.get_hard_actions(outputs, ref_prices)

        buy_prices = actions['buy_price'].numpy()
        sell_prices = actions['sell_price'].numpy()

        # Statistics
        spreads_pct = (sell_prices - buy_prices) / buy_prices * 100

        print(f"\nPrice Safety Statistics over {n_decisions} decisions:")
        print(f"  Min spread: {spreads_pct.min():.4f}%")
        print(f"  Max spread: {spreads_pct.max():.4f}%")
        print(f"  Mean spread: {spreads_pct.mean():.4f}%")
        print(f"  Inverted prices: {(buy_prices >= sell_prices).sum()}")

        # CRITICAL: Zero tolerance for inverted prices
        assert (buy_prices < sell_prices).all(), "CRITICAL: Found inverted prices in simulation!"
