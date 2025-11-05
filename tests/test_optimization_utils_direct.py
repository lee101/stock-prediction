"""
Unit tests for optimization_utils with DIRECT optimizer.
Tests both DIRECT and differential_evolution modes.
"""

import pytest
import torch
import numpy as np
import os
from src.optimization_utils import (
    optimize_entry_exit_multipliers,
    optimize_always_on_multipliers,
    _USE_DIRECT,
)


@pytest.fixture
def sample_data():
    """Generate sample market data for testing"""
    torch.manual_seed(42)
    n = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    close_actual = torch.randn(n, device=device) * 0.02
    high_actual = close_actual + torch.abs(torch.randn(n, device=device)) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n, device=device)) * 0.01
    high_pred = torch.randn(n, device=device) * 0.01 + 0.005
    low_pred = torch.randn(n, device=device) * 0.01 - 0.005
    positions = torch.where(
        torch.abs(high_pred) > torch.abs(low_pred),
        torch.ones(n, device=device),
        -torch.ones(n, device=device)
    )

    return {
        'close_actual': close_actual,
        'high_actual': high_actual,
        'low_actual': low_actual,
        'high_pred': high_pred,
        'low_pred': low_pred,
        'positions': positions,
    }


class TestDirectOptimizer:
    """Tests for DIRECT optimizer integration"""

    def test_direct_enabled_by_default(self):
        """Test that DIRECT is enabled by default"""
        # Don't set env var, check default behavior
        import importlib
        import src.optimization_utils as opt_utils
        importlib.reload(opt_utils)
        assert opt_utils._USE_DIRECT is True

    def test_direct_can_be_disabled(self):
        """Test that DIRECT can be disabled via env var"""
        os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '0'
        import importlib
        import src.optimization_utils as opt_utils
        importlib.reload(opt_utils)
        assert opt_utils._USE_DIRECT is False
        # Reset
        os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '1'

    def test_direct_returns_valid_results(self, sample_data):
        """Test that DIRECT returns valid optimization results"""
        h_mult, l_mult, profit = optimize_entry_exit_multipliers(
            sample_data['close_actual'],
            sample_data['positions'],
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            maxiter=30,
            popsize=8,
        )

        # Check results are valid
        assert isinstance(h_mult, float)
        assert isinstance(l_mult, float)
        assert isinstance(profit, float)

        # Check bounds are respected
        assert -0.03 <= h_mult <= 0.03
        assert -0.03 <= l_mult <= 0.03

        # Profit should be finite
        assert np.isfinite(profit)

    def test_direct_vs_de_quality(self, sample_data):
        """Test that DIRECT finds similar or better solutions than DE"""
        # Run with DIRECT
        os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '1'
        import importlib
        import src.optimization_utils as opt_utils
        importlib.reload(opt_utils)

        h_direct, l_direct, p_direct = opt_utils.optimize_entry_exit_multipliers(
            sample_data['close_actual'],
            sample_data['positions'],
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            maxiter=30,
            popsize=8,
            seed=42,
        )

        # Run with DE
        os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '0'
        importlib.reload(opt_utils)

        h_de, l_de, p_de = opt_utils.optimize_entry_exit_multipliers(
            sample_data['close_actual'],
            sample_data['positions'],
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            maxiter=30,
            popsize=8,
            seed=42,
        )

        # Results should be within 10% of each other
        assert abs(p_direct - p_de) / abs(p_de) < 0.10, \
            f"DIRECT profit {p_direct} differs too much from DE profit {p_de}"

        # Reset
        os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '1'

    def test_close_at_eod_parameter(self, sample_data):
        """Test optimization with close_at_eod parameter"""
        h1, l1, p1 = optimize_entry_exit_multipliers(
            sample_data['close_actual'],
            sample_data['positions'],
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            close_at_eod=False,
            maxiter=20,
            popsize=6,
        )

        h2, l2, p2 = optimize_entry_exit_multipliers(
            sample_data['close_actual'],
            sample_data['positions'],
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            close_at_eod=True,
            maxiter=20,
            popsize=6,
        )

        # Results should differ (different policy)
        assert h1 != h2 or l1 != l2 or p1 != p2

    def test_trading_fee_effect(self, sample_data):
        """Test that trading fee affects optimization"""
        h1, l1, p1 = optimize_entry_exit_multipliers(
            sample_data['close_actual'],
            sample_data['positions'],
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            trading_fee=0.0,
            maxiter=20,
            popsize=6,
        )

        h2, l2, p2 = optimize_entry_exit_multipliers(
            sample_data['close_actual'],
            sample_data['positions'],
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            trading_fee=0.01,  # 1% fee
            maxiter=20,
            popsize=6,
        )

        # Profit with fee should be lower
        assert p2 < p1

    def test_custom_bounds(self, sample_data):
        """Test optimization with custom bounds"""
        custom_bounds = ((-0.01, 0.01), (-0.01, 0.01))

        h_mult, l_mult, profit = optimize_entry_exit_multipliers(
            sample_data['close_actual'],
            sample_data['positions'],
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            bounds=custom_bounds,
            maxiter=20,
            popsize=6,
        )

        # Check custom bounds are respected
        assert -0.01 <= h_mult <= 0.01
        assert -0.01 <= l_mult <= 0.01


class TestAlwaysOnOptimizer:
    """Tests for always-on strategy optimizer"""

    @pytest.fixture
    def always_on_data(self, sample_data):
        """Add indicators for always-on strategy"""
        n = len(sample_data['close_actual'])
        device = sample_data['close_actual'].device

        data = sample_data.copy()
        # Buy when predicted high > 0, sell when < 0
        data['buy_indicator'] = (sample_data['high_pred'] > 0).float()
        data['sell_indicator'] = (sample_data['low_pred'] < 0).float()

        return data

    def test_always_on_crypto(self, always_on_data):
        """Test always-on optimizer for crypto (buy only)"""
        h_mult, l_mult, profit = optimize_always_on_multipliers(
            always_on_data['close_actual'],
            always_on_data['buy_indicator'],
            always_on_data['sell_indicator'],
            always_on_data['high_actual'],
            always_on_data['high_pred'],
            always_on_data['low_actual'],
            always_on_data['low_pred'],
            is_crypto=True,
            maxiter=20,
            popsize=6,
        )

        assert isinstance(h_mult, float)
        assert isinstance(l_mult, float)
        assert np.isfinite(profit)

    def test_always_on_stocks(self, always_on_data):
        """Test always-on optimizer for stocks (buy + sell)"""
        h_mult, l_mult, profit = optimize_always_on_multipliers(
            always_on_data['close_actual'],
            always_on_data['buy_indicator'],
            always_on_data['sell_indicator'],
            always_on_data['high_actual'],
            always_on_data['high_pred'],
            always_on_data['low_actual'],
            always_on_data['low_pred'],
            is_crypto=False,
            maxiter=20,
            popsize=6,
        )

        assert isinstance(h_mult, float)
        assert isinstance(l_mult, float)
        assert np.isfinite(profit)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_positions(self, sample_data):
        """Test with all zero positions"""
        zero_positions = torch.zeros_like(sample_data['positions'])

        h_mult, l_mult, profit = optimize_entry_exit_multipliers(
            sample_data['close_actual'],
            zero_positions,
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            maxiter=20,
            popsize=6,
        )

        # Should complete without error
        assert np.isfinite(profit)
        # Profit should be zero (no trades)
        assert abs(profit) < 1e-6

    def test_all_long_positions(self, sample_data):
        """Test with all long positions"""
        long_positions = torch.ones_like(sample_data['positions'])

        h_mult, l_mult, profit = optimize_entry_exit_multipliers(
            sample_data['close_actual'],
            long_positions,
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            maxiter=20,
            popsize=6,
        )

        assert np.isfinite(profit)

    def test_small_dataset(self):
        """Test with small dataset (10 days)"""
        n = 10
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        close_actual = torch.randn(n, device=device) * 0.02
        high_actual = close_actual + 0.01
        low_actual = close_actual - 0.01
        high_pred = torch.randn(n, device=device) * 0.01
        low_pred = torch.randn(n, device=device) * 0.01
        positions = torch.ones(n, device=device)

        h_mult, l_mult, profit = optimize_entry_exit_multipliers(
            close_actual, positions, high_actual, high_pred,
            low_actual, low_pred,
            maxiter=10,
            popsize=4,
        )

        assert np.isfinite(profit)

    def test_zero_variance_data(self):
        """Test with constant predictions (zero variance)"""
        n = 50
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        close_actual = torch.randn(n, device=device) * 0.02
        high_actual = close_actual + 0.01
        low_actual = close_actual - 0.01
        # Constant predictions
        high_pred = torch.ones(n, device=device) * 0.01
        low_pred = torch.ones(n, device=device) * -0.01
        positions = torch.ones(n, device=device)

        h_mult, l_mult, profit = optimize_entry_exit_multipliers(
            close_actual, positions, high_actual, high_pred,
            low_actual, low_pred,
            maxiter=10,
            popsize=4,
        )

        assert np.isfinite(profit)


class TestPerformance:
    """Performance and timing tests"""

    def test_direct_is_faster_than_de(self, sample_data):
        """Verify DIRECT is actually faster than DE"""
        import time
        import importlib
        import src.optimization_utils as opt_utils

        # Time DIRECT
        os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '1'
        importlib.reload(opt_utils)

        start = time.time()
        for _ in range(5):
            opt_utils.optimize_entry_exit_multipliers(
                sample_data['close_actual'],
                sample_data['positions'],
                sample_data['high_actual'],
                sample_data['high_pred'],
                sample_data['low_actual'],
                sample_data['low_pred'],
                maxiter=30,
                popsize=8,
            )
        time_direct = time.time() - start

        # Time DE
        os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '0'
        importlib.reload(opt_utils)

        start = time.time()
        for _ in range(5):
            opt_utils.optimize_entry_exit_multipliers(
                sample_data['close_actual'],
                sample_data['positions'],
                sample_data['high_actual'],
                sample_data['high_pred'],
                sample_data['low_actual'],
                sample_data['low_pred'],
                maxiter=30,
                popsize=8,
                seed=42,
            )
        time_de = time.time() - start

        # DIRECT should be faster (allow 5% margin)
        speedup = time_de / time_direct
        print(f"\nSpeedup: {speedup:.2f}x (DIRECT: {time_direct:.2f}s, DE: {time_de:.2f}s)")
        assert speedup > 1.05, f"DIRECT not faster: {speedup:.2f}x"

        # Reset
        os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '1'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
