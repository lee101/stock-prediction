"""
Test that the model's decode_actions always produces valid price spreads.
"""
import pytest
import torch
from hourlycryptotraining.model import HourlyCryptoPolicy, PolicyHeadConfig


class TestModelPriceSpreadValidation:
    """Ensure the model never outputs inverted or insufficient spreads."""

    @pytest.fixture
    def policy(self):
        """Create a policy for testing."""
        config = PolicyHeadConfig(
            input_dim=50,
            hidden_dim=128,
            dropout=0.1,
            price_offset_pct=0.003,
            max_trade_qty=1.0,
            min_price_gap_pct=0.0003,  # 3 basis points
            num_heads=4,
            num_layers=2,
            max_len=256,
        )
        return HourlyCryptoPolicy(config)

    def test_model_enforces_minimum_gap(self, policy):
        """Model should enforce min_price_gap_pct between buy and sell."""
        batch_size = 4
        seq_len = 10

        # Create dummy outputs
        outputs = {
            "buy_price_logits": torch.randn(batch_size, seq_len, 1),
            "sell_price_logits": torch.randn(batch_size, seq_len, 1),
            "buy_amount_logits": torch.randn(batch_size, seq_len, 1),
            "sell_amount_logits": torch.randn(batch_size, seq_len, 1),
        }

        # Reference prices
        reference_close = torch.full((batch_size, seq_len), 100.0)
        chronos_high = torch.full((batch_size, seq_len), 101.0)
        chronos_low = torch.full((batch_size, seq_len), 99.0)

        # Decode actions
        decoded = policy.decode_actions(
            outputs,
            reference_close=reference_close,
            chronos_high=chronos_high,
            chronos_low=chronos_low,
        )

        buy_price = decoded["buy_price"]
        sell_price = decoded["sell_price"]

        # CRITICAL: Sell price must always be higher than buy price
        assert torch.all(sell_price > buy_price), (
            "Model produced inverted prices! "
            f"buy_price={buy_price.min():.4f}-{buy_price.max():.4f}, "
            f"sell_price={sell_price.min():.4f}-{sell_price.max():.4f}"
        )

        # Verify minimum gap is enforced (3 basis points)
        min_gap = reference_close * policy.min_gap_pct
        actual_gap = sell_price - buy_price

        assert torch.all(actual_gap >= min_gap), (
            f"Model produced insufficient spread! "
            f"Min gap: {min_gap[0, 0]:.4f}, "
            f"Actual gap: {actual_gap.min():.4f}-{actual_gap.max():.4f}"
        )

    def test_model_spread_with_extreme_inputs(self, policy):
        """Test model maintains valid spreads even with extreme inputs."""
        batch_size = 2
        seq_len = 5

        # Extreme outputs (all -10 or +10 after tanh will be close to -1 or +1)
        outputs = {
            "buy_price_logits": torch.full((batch_size, seq_len, 1), -10.0),
            "sell_price_logits": torch.full((batch_size, seq_len, 1), 10.0),
            "buy_amount_logits": torch.zeros(batch_size, seq_len, 1),
            "sell_amount_logits": torch.zeros(batch_size, seq_len, 1),
        }

        reference_close = torch.full((batch_size, seq_len), 50000.0)  # BTC-like price
        chronos_high = torch.full((batch_size, seq_len), 50500.0)
        chronos_low = torch.full((batch_size, seq_len), 49500.0)

        decoded = policy.decode_actions(
            outputs,
            reference_close=reference_close,
            chronos_high=chronos_high,
            chronos_low=chronos_low,
        )

        buy_price = decoded["buy_price"]
        sell_price = decoded["sell_price"]

        # Must maintain valid spread even with extreme inputs
        assert torch.all(sell_price > buy_price), "Extreme inputs caused inverted prices!"

        min_gap = reference_close * policy.min_gap_pct
        actual_gap = sell_price - buy_price
        assert torch.all(actual_gap >= min_gap), "Extreme inputs violated minimum gap!"

    def test_model_spread_across_price_ranges(self, policy):
        """Test spread validation across different price ranges."""
        batch_size = 1
        seq_len = 1

        # Test different price ranges (penny stocks, regular stocks, BTC, etc.)
        price_ranges = [
            1.0,      # Penny stock
            10.0,     # Low-price stock
            100.0,    # Regular stock
            1000.0,   # High-price stock
            50000.0,  # BTC
            90000.0,  # BTC ATH
        ]

        for ref_price in price_ranges:
            outputs = {
                "buy_price_logits": torch.zeros(batch_size, seq_len, 1),
                "sell_price_logits": torch.zeros(batch_size, seq_len, 1),
                "buy_amount_logits": torch.zeros(batch_size, seq_len, 1),
                "sell_amount_logits": torch.zeros(batch_size, seq_len, 1),
            }

            reference_close = torch.full((batch_size, seq_len), ref_price)
            chronos_high = reference_close * 1.01
            chronos_low = reference_close * 0.99

            decoded = policy.decode_actions(
                outputs,
                reference_close=reference_close,
                chronos_high=chronos_high,
                chronos_low=chronos_low,
            )

            buy_price = decoded["buy_price"]
            sell_price = decoded["sell_price"]

            # Validate spread at this price range
            assert sell_price > buy_price, f"Invalid spread at price {ref_price}"

            min_gap = reference_close * policy.min_gap_pct
            actual_gap = sell_price - buy_price

            assert actual_gap >= min_gap, (
                f"Insufficient spread at price {ref_price}: "
                f"gap={actual_gap.item():.6f}, min={min_gap.item():.6f}"
            )

    def test_model_config_min_gap_is_reasonable(self):
        """Ensure min_price_gap_pct is set to a reasonable value (3bp)."""
        config = PolicyHeadConfig(
            input_dim=50,
            hidden_dim=128,
            dropout=0.1,
            price_offset_pct=0.003,
            max_trade_qty=1.0,
            min_price_gap_pct=0.0003,  # Should be 3bp
            num_heads=4,
            num_layers=2,
            max_len=256,
        )

        # Verify minimum gap is at least 3 basis points
        assert config.min_gap_pct >= 0.0003, (
            f"min_price_gap_pct too small! {config.min_gap_pct} < 0.0003 (3bp)"
        )

        # Also shouldn't be ridiculously large (> 1%)
        assert config.min_gap_pct <= 0.01, (
            f"min_price_gap_pct too large! {config.min_gap_pct} > 0.01 (100bp)"
        )

    def test_decode_actions_implementation_has_gap_enforcement(self, policy):
        """Verify the decode_actions code actually enforces the gap."""
        batch_size = 1
        seq_len = 1

        # Set up scenario where buy and sell would be equal without gap enforcement
        outputs = {
            "buy_price_logits": torch.zeros(batch_size, seq_len, 1),
            "sell_price_logits": torch.zeros(batch_size, seq_len, 1),
            "buy_amount_logits": torch.zeros(batch_size, seq_len, 1),
            "sell_amount_logits": torch.zeros(batch_size, seq_len, 1),
        }

        reference_close = torch.full((batch_size, seq_len), 100.0)
        chronos_high = torch.full((batch_size, seq_len), 100.0)
        chronos_low = torch.full((batch_size, seq_len), 100.0)

        decoded = policy.decode_actions(
            outputs,
            reference_close=reference_close,
            chronos_high=chronos_high,
            chronos_low=chronos_low,
        )

        buy_price = decoded["buy_price"]
        sell_price = decoded["sell_price"]

        # Even with identical inputs, gap should be enforced
        expected_min_gap = 100.0 * policy.min_gap_pct  # 0.03
        actual_gap = sell_price - buy_price

        assert actual_gap >= expected_min_gap, (
            f"Gap enforcement not working! "
            f"Expected >= {expected_min_gap:.4f}, got {actual_gap.item():.4f}"
        )
