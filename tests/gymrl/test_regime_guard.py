import numpy as np
import pytest

from gymrl.config import PortfolioEnvConfig
from gymrl.portfolio_env import PortfolioEnv


def _build_env(config: PortfolioEnvConfig, returns: np.ndarray) -> PortfolioEnv:
    features = np.zeros((returns.shape[0], returns.shape[1], 3), dtype=np.float32)
    timestamps = np.arange(returns.shape[0])
    symbols = [f"SYM{i}" for i in range(returns.shape[1])]
    return PortfolioEnv(
        features=features,
        realized_returns=returns,
        config=config,
        feature_names=[f"f{i}" for i in range(3)],
        symbols=symbols,
        timestamps=timestamps,
        append_portfolio_state=True,
        start_index=0,
        episode_length=min(returns.shape[0] - 1, 10),
    )


def _allocator_action(weight_bias: float) -> np.ndarray:
    # Utility to produce deterministic allocations with the leverage head enabled.
    # First N entries affect softmax logits, final entry selects the gross leverage scale.
    return np.array([weight_bias, -weight_bias, 8.0], dtype=np.float32)


def test_regime_guard_scales_leverage_and_turnover_penalty():
    returns = np.zeros((6, 2), dtype=np.float32)
    returns[0] = np.array([-0.2, 0.0], dtype=np.float32)
    config = PortfolioEnvConfig(
        turnover_penalty=0.0,
        drawdown_penalty=0.0,
        include_cash=False,
        weight_cap=None,
        loss_shutdown_enabled=False,
        base_gross_exposure=1.0,
        max_gross_leverage=1.0,
        intraday_leverage_cap=1.0,
        closing_leverage_cap=1.0,
        leverage_head=True,
        regime_filters_enabled=True,
        regime_drawdown_threshold=0.05,
        regime_leverage_scale=0.5,
        regime_negative_return_window=2,
        regime_negative_return_threshold=0.0,
        regime_negative_return_turnover_penalty=0.01,
        regime_turnover_threshold=1.2,  # keep turnover guard inactive in this test
        regime_turnover_probe_weight=0.005,
    )
    env = _build_env(config, returns)

    env.reset()
    action = _allocator_action(weight_bias=5.0)
    _, _, _, _, info_first = env.step(action)
    assert info_first["regime_drawdown_guard"] == 0.0
    assert info_first["turnover_penalty_applied"] == pytest.approx(0.0)

    _, _, _, _, info_second = env.step(action)
    assert info_second["regime_drawdown_guard"] == 1.0
    assert info_second["regime_negative_return_guard"] == 1.0
    assert info_second["regime_leverage_scale"] == pytest.approx(0.5, rel=1e-5)
    assert info_second["gross_exposure_intraday"] == pytest.approx(0.5, rel=1e-5)
    assert info_second["turnover_penalty_applied"] == pytest.approx(0.01, rel=1e-5)


def test_regime_guard_turnover_probe_override_and_reset():
    returns = np.zeros((8, 2), dtype=np.float32)
    returns[0] = np.array([-0.1, 0.0], dtype=np.float32)
    config = PortfolioEnvConfig(
        turnover_penalty=0.0,
        drawdown_penalty=0.0,
        include_cash=False,
        weight_cap=None,
        loss_shutdown_enabled=True,
        loss_shutdown_probe_weight=0.02,
        loss_shutdown_cooldown=2,
        base_gross_exposure=1.0,
        max_gross_leverage=1.0,
        intraday_leverage_cap=1.0,
        closing_leverage_cap=1.0,
        leverage_head=True,
        regime_filters_enabled=True,
        regime_drawdown_threshold=0.3,  # avoid drawdown guard triggering
        regime_leverage_scale=0.8,
        regime_negative_return_window=3,
        regime_negative_return_threshold=-0.2,  # keep negative guard inactive
        regime_negative_return_turnover_penalty=None,
        regime_turnover_threshold=1.5,
        regime_turnover_probe_weight=0.003,
    )
    env = _build_env(config, returns)

    env.reset()
    action_long_asset0 = _allocator_action(weight_bias=5.0)
    _, _, _, _, info_step0 = env.step(action_long_asset0)
    assert info_step0["regime_turnover_guard"] == 0.0
    assert info_step0["loss_shutdown_probe_applied"] == pytest.approx(0.02, rel=1e-5)

    action_long_asset1 = np.array([-5.0, 5.0, 8.0], dtype=np.float32)
    _, _, _, _, info_step1 = env.step(action_long_asset1)
    assert info_step1["regime_turnover_guard"] == 1.0
    assert info_step1["loss_shutdown_probe_applied"] == pytest.approx(0.003, rel=1e-5)

    # Low-turnover step should reset the probe to the base value.
    _, _, _, _, info_step2 = env.step(action_long_asset1)
    assert info_step2["regime_turnover_guard"] == 0.0
    assert info_step2["loss_shutdown_probe_applied"] == pytest.approx(0.02, rel=1e-5)
