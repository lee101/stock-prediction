from __future__ import annotations

import pytest

try:
    import jax  # noqa: F401
    import flax  # noqa: F401
except ImportError:
    pytest.skip("jax/flax not installed", allow_module_level=True)

import numpy as np
import torch

from binanceneural.jax_losses import (
    combined_sortino_pnl_loss as jax_combined_sortino_pnl_loss,
    compute_hourly_objective as jax_compute_hourly_objective,
    simulate_hourly_trades as jax_simulate_hourly_trades,
    simulate_hourly_trades_binary as jax_simulate_hourly_trades_binary,
)
from differentiable_loss_utils import (
    combined_sortino_pnl_loss as torch_combined_sortino_pnl_loss,
    compute_hourly_objective as torch_compute_hourly_objective,
    simulate_hourly_trades as torch_simulate_hourly_trades,
    simulate_hourly_trades_binary as torch_simulate_hourly_trades_binary,
)


def _sample_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    closes = rng.uniform(90.0, 110.0, size=(2, 8)).astype(np.float32)
    highs = closes + rng.uniform(0.1, 1.0, size=(2, 8)).astype(np.float32)
    lows = closes - rng.uniform(0.1, 1.0, size=(2, 8)).astype(np.float32)
    opens = closes + rng.uniform(-0.5, 0.5, size=(2, 8)).astype(np.float32)
    buy_prices = closes - 0.25
    sell_prices = closes + 0.35
    trade_intensity = rng.uniform(0.0, 0.8, size=(2, 8)).astype(np.float32)
    return {
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "opens": opens,
        "buy_prices": buy_prices,
        "sell_prices": sell_prices,
        "trade_intensity": trade_intensity,
        "buy_trade_intensity": trade_intensity * 0.9,
        "sell_trade_intensity": trade_intensity * 0.8,
    }


def test_simulate_hourly_trades_matches_torch() -> None:
    inputs = _sample_inputs()
    torch_result = torch_simulate_hourly_trades(
        **{key: torch.from_numpy(value) for key, value in inputs.items()},
        maker_fee=0.001,
        initial_cash=1.0,
        temperature=5e-4,
        max_leverage=2.0,
        can_short=True,
        can_long=True,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        margin_annual_rate=0.0625,
    )
    jax_result = jax_simulate_hourly_trades(
        **inputs,
        maker_fee=0.001,
        initial_cash=1.0,
        temperature=5e-4,
        max_leverage=2.0,
        can_short=True,
        can_long=True,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        margin_annual_rate=0.0625,
    )
    np.testing.assert_allclose(torch_result.returns.numpy(), np.asarray(jax_result.returns), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        torch_result.portfolio_values.numpy(),
        np.asarray(jax_result.portfolio_values),
        rtol=1e-5,
        atol=1e-5,
    )


def test_binary_sim_and_objective_match_torch() -> None:
    inputs = _sample_inputs()
    torch_sim = torch_simulate_hourly_trades_binary(
        **{key: torch.from_numpy(value) for key, value in inputs.items()},
        maker_fee=0.001,
        initial_cash=1.0,
        max_leverage=2.0,
        can_short=False,
        can_long=True,
        decision_lag_bars=0,
        market_order_entry=False,
        fill_buffer_pct=0.0,
        margin_annual_rate=0.0,
    )
    jax_sim = jax_simulate_hourly_trades_binary(
        **inputs,
        maker_fee=0.001,
        initial_cash=1.0,
        max_leverage=2.0,
        can_short=False,
        can_long=True,
        decision_lag_bars=0,
        market_order_entry=False,
        fill_buffer_pct=0.0,
        margin_annual_rate=0.0,
    )
    np.testing.assert_allclose(torch_sim.returns.numpy(), np.asarray(jax_sim.returns), rtol=1e-6, atol=1e-6)

    torch_score, torch_sortino, torch_annual = torch_compute_hourly_objective(
        torch_sim.returns,
        periods_per_year=8760.0,
        return_weight=0.15,
        smoothness_penalty=0.0,
    )
    jax_score, jax_sortino, jax_annual = jax_compute_hourly_objective(
        jax_sim.returns,
        periods_per_year=8760.0,
        return_weight=0.15,
        smoothness_penalty=0.0,
    )
    np.testing.assert_allclose(torch_score.numpy(), np.asarray(jax_score), rtol=2e-5, atol=1e-4)
    np.testing.assert_allclose(torch_sortino.numpy(), np.asarray(jax_sortino), rtol=2e-5, atol=1e-4)
    np.testing.assert_allclose(torch_annual.numpy(), np.asarray(jax_annual), rtol=2e-5, atol=1e-4)

    torch_loss = torch_combined_sortino_pnl_loss(
        torch_sim.returns,
        target_sign=1.0,
        periods_per_year=8760.0,
        return_weight=0.15,
        smoothness_penalty=0.0,
    )
    jax_loss = jax_combined_sortino_pnl_loss(
        jax_sim.returns,
        target_sign=1.0,
        periods_per_year=8760.0,
        return_weight=0.15,
        smoothness_penalty=0.0,
    )
    np.testing.assert_allclose(torch_loss.numpy(), np.asarray(jax_loss), rtol=2e-5, atol=1e-4)
