import numpy as np
import pytest
import torch

from gymrl.config import PortfolioEnvConfig
from gymrl.differentiable_utils import (
    LossShutdownParams,
    LossShutdownState,
    loss_shutdown_adjust,
    update_loss_shutdown_state,
)
from gymrl.portfolio_env import PortfolioEnv


def test_loss_shutdown_env_probe_and_release():
    T, N, F = 6, 1, 1
    features = np.zeros((T, N, F), dtype=np.float32)
    realized_returns = np.array([[-0.05], [0.04], [0.03], [0.0], [0.0], [0.0]], dtype=np.float32)
    config = PortfolioEnvConfig(
        include_cash=False,
        loss_shutdown_enabled=True,
        loss_shutdown_cooldown=2,
        loss_shutdown_probe_weight=0.1,
        loss_shutdown_penalty=0.5,
        loss_shutdown_min_position=1e-5,
        loss_shutdown_return_tolerance=1e-6,
        leverage_head=False,
        weight_cap=None,
    )

    env = PortfolioEnv(features, realized_returns, config=config, symbols=["AAPL"])
    env.reset()

    # Step 0: allocate fully, incur loss -> cooldown activates.
    action_high = np.array([6.0], dtype=np.float32)
    _, _, _, _, info_step0 = env.step(action_high)
    assert info_step0["loss_shutdown_clipped"] == pytest.approx(0.0)
    assert info_step0["loss_shutdown_active_long"] == pytest.approx(1.0)
    assert info_step0["loss_shutdown_penalty"] == pytest.approx(0.0)
    assert env.current_weights[0] == pytest.approx(1.0, rel=1e-6)

    # Step 1: cooldown clamps weight to probe size and applies penalty.
    _, _, _, _, info_step1 = env.step(action_high)
    assert env.current_weights[0] == pytest.approx(config.loss_shutdown_probe_weight, rel=1e-6)
    assert info_step1["loss_shutdown_clipped"] > 0.0
    assert info_step1["loss_shutdown_penalty"] == pytest.approx(
        config.loss_shutdown_penalty * config.loss_shutdown_probe_weight, rel=1e-6
    )
    assert info_step1["loss_shutdown_active_long"] == pytest.approx(0.0)

    # Positive return on step 1 should release cooldown for next step.
    _, _, _, _, info_step2 = env.step(action_high)
    assert env.current_weights[0] == pytest.approx(1.0, rel=1e-6)
    assert info_step2["loss_shutdown_clipped"] == pytest.approx(0.0)
    assert info_step2["loss_shutdown_active_long"] == pytest.approx(0.0)


def test_loss_shutdown_torch_utils_behaviour():
    weights = torch.tensor([0.8, -0.6], dtype=torch.float32)
    state = LossShutdownState(
        long_counters=torch.tensor([2, 0], dtype=torch.int32),
        short_counters=torch.tensor([0, 3], dtype=torch.int32),
    )
    params = LossShutdownParams(probe_weight=0.1, penalty_scale=0.5)

    adjusted, penalty, clipped = loss_shutdown_adjust(weights, state, params, allow_short=True)
    assert torch.allclose(adjusted, torch.tensor([0.1, -0.1], dtype=torch.float32), atol=1e-6)
    assert penalty.item() == pytest.approx(0.1, rel=1e-6)
    assert clipped.item() == pytest.approx((0.8 - 0.1) + (0.6 - 0.1), rel=1e-6)

    net_returns = torch.tensor([-0.02, 0.03], dtype=torch.float32)
    new_state = update_loss_shutdown_state(adjusted, net_returns, state, params, allow_short=True)
    assert torch.equal(new_state.long_counters, torch.tensor([params.cooldown_steps, 0], dtype=torch.int32))
    assert torch.equal(new_state.short_counters, torch.tensor([0, 0], dtype=torch.int32))


def test_compute_step_net_return_matches_env_costs():
    T, N, F = 4, 2, 1
    features = np.zeros((T, N, F), dtype=np.float32)
    realized_returns = np.array([[0.02, -0.01], [0.015, -0.005], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    config = PortfolioEnvConfig(include_cash=False, leverage_head=False)
    env = PortfolioEnv(features, realized_returns, config=config, symbols=["AAPL", "BTCUSD"])
    env.reset()

    action = np.array([2.0, -2.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    prev_weights = torch.from_numpy(env.last_weights.copy())
    new_weights = torch.from_numpy(env.current_weights.copy())
    realized = torch.from_numpy(realized_returns[env.start_index].copy())
    cost_vector = torch.from_numpy(env.costs_vector.copy())

    from gymrl.differentiable_utils import compute_step_net_return

    net_return, turnover, trading_cost = compute_step_net_return(prev_weights, new_weights, realized, cost_vector)

    assert net_return.item() == pytest.approx(info["net_return"], rel=1e-6)
    assert turnover.item() == pytest.approx(info["turnover"], rel=1e-6)
    assert trading_cost.item() == pytest.approx(info["trading_cost"], rel=1e-6)
