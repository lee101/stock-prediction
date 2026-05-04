from __future__ import annotations

import gpu_trading_env


def test_portfolio_bracket_defaults_use_production_costs() -> None:
    cfg = gpu_trading_env.PortfolioBracketConfig()

    assert gpu_trading_env.PRODUCTION_FEE_BPS == 10.0
    assert gpu_trading_env.PRODUCTION_FILL_BUFFER_BPS == 5.0
    assert cfg.fee_bps == gpu_trading_env.PRODUCTION_FEE_BPS
    assert cfg.fill_buffer_bps == gpu_trading_env.PRODUCTION_FILL_BUFFER_BPS


def test_base_env_defaults_use_same_production_costs() -> None:
    cfg = gpu_trading_env.EnvConfig()

    assert cfg.fee_bps == gpu_trading_env.PRODUCTION_FEE_BPS
    assert cfg.buffer_bps == gpu_trading_env.PRODUCTION_FILL_BUFFER_BPS
