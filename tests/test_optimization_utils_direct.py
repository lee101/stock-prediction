"""
Unit tests for optimization_utils with DIRECT optimizer.
Tests both DIRECT and differential_evolution modes.
"""

import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.optimization_utils as opt_utils
from src.optimization_utils import (
    optimize_always_on_multipliers,
    optimize_entry_exit_multipliers,
)


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _resolve_test_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.empty(1, device="cuda")
    except Exception as exc:
        if _is_cuda_resource_pressure_error(exc):
            return "cpu"
        raise
    return "cuda"


@pytest.fixture(autouse=True)
def _reset_optimizer_mode_env():
    """Keep env-driven optimizer mode from leaking across tests."""
    original_use_direct = os.environ.get("MARKETSIM_USE_DIRECT_OPTIMIZER")
    original_fast = os.environ.get("MARKETSIM_FAST_OPTIMIZE")
    os.environ["MARKETSIM_USE_DIRECT_OPTIMIZER"] = "1"
    if original_fast is None:
        os.environ.pop("MARKETSIM_FAST_OPTIMIZE", None)
    else:
        os.environ["MARKETSIM_FAST_OPTIMIZE"] = original_fast
    importlib.reload(opt_utils)
    try:
        yield
    finally:
        if original_use_direct is None:
            os.environ.pop("MARKETSIM_USE_DIRECT_OPTIMIZER", None)
        else:
            os.environ["MARKETSIM_USE_DIRECT_OPTIMIZER"] = original_use_direct
        if original_fast is None:
            os.environ.pop("MARKETSIM_FAST_OPTIMIZE", None)
        else:
            os.environ["MARKETSIM_FAST_OPTIMIZE"] = original_fast
        importlib.reload(opt_utils)


@pytest.fixture
def sample_data():
    """Generate sample market data for testing"""
    torch.manual_seed(42)
    n = 100
    device = _resolve_test_device()

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
        import src.optimization_utils as opt_utils

        importlib.reload(opt_utils)
        assert opt_utils._USE_DIRECT is True

    def test_direct_can_be_disabled(self):
        """Test that DIRECT can be disabled via env var"""
        os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '0'
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
        objective_keep_open = opt_utils._EntryExitObjective(
            sample_data['close_actual'],
            sample_data['positions'],
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            close_at_eod=False,
            trading_fee=None,
        )
        objective_close_eod = opt_utils._EntryExitObjective(
            sample_data['close_actual'],
            sample_data['positions'],
            sample_data['high_actual'],
            sample_data['high_pred'],
            sample_data['low_actual'],
            sample_data['low_pred'],
            close_at_eod=True,
            trading_fee=None,
        )
        reference_params = (0.01, -0.01)

        assert objective_keep_open(reference_params) != objective_close_eod(reference_params)

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
            seed=42,
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
            seed=42,
        )

        assert p1 == pytest.approx(-objective_keep_open((h1, l1)))
        assert p2 == pytest.approx(-objective_close_eod((h2, l2)))

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
        device = _resolve_test_device()

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
        device = _resolve_test_device()

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

    def test_direct_uses_smaller_search_budget_than_de(self, monkeypatch):
        """Verify DIRECT is configured with the smaller deterministic search budget."""
        import src.optimization_utils as opt_utils

        calls = {}

        def objective(params):
            return float(sum(params))

        def fake_direct(func, bounds, *, maxfun):
            calls["direct"] = {"bounds": bounds, "maxfun": maxfun}
            assert func((0.0, 0.0)) == 0.0
            return SimpleNamespace(x=np.array([0.0, 0.0]), fun=0.0)

        def fake_de(func, bounds, *, maxiter, popsize, atol, seed, workers, updating):
            calls["de"] = {
                "bounds": bounds,
                "maxiter": maxiter,
                "popsize": popsize,
                "atol": atol,
                "seed": seed,
                "workers": workers,
                "updating": updating,
            }
            assert func((0.0, 0.0)) == 0.0
            return SimpleNamespace(x=np.array([0.0, 0.0]), fun=0.0)

        monkeypatch.setattr(opt_utils, "direct", fake_direct)
        monkeypatch.setattr(opt_utils, "differential_evolution", fake_de)
        monkeypatch.setattr(opt_utils, "_USE_DIRECT", True)
        opt_utils.run_bounded_optimizer(
            objective,
            bounds=((-0.03, 0.03), (-0.03, 0.03)),
            maxiter=30,
            popsize=8,
        )

        monkeypatch.setattr(opt_utils, "_USE_DIRECT", False)
        opt_utils.run_bounded_optimizer(
            objective,
            bounds=((-0.03, 0.03), (-0.03, 0.03)),
            maxiter=30,
            popsize=8,
            seed=42,
        )

        de_population_budget = (calls["de"]["maxiter"] + 1) * calls["de"]["popsize"] * 2
        assert calls["direct"]["maxfun"] == opt_utils._direct_maxfun(30, 8)
        assert calls["direct"]["maxfun"] < de_population_budget
        assert calls["de"]["seed"] == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
