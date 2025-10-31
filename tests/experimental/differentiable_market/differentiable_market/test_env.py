from __future__ import annotations

import math

import torch

from differentiable_market.config import EnvironmentConfig
from differentiable_market.env import DifferentiableMarketEnv, smooth_abs
from src.alpaca_utils import ANNUAL_MARGIN_RATE, TRADING_DAYS_PER_YEAR
from stockagent.constants import CRYPTO_TRADING_FEE, TRADING_FEE


def test_env_symbol_specific_fees() -> None:
    cfg = EnvironmentConfig(
        transaction_cost=0.0,
        risk_aversion=0.0,
        max_intraday_leverage=4.0,
        max_overnight_leverage=4.0,
        cash_transaction_cost=0.0,
    )
    env = DifferentiableMarketEnv(cfg)
    env.set_asset_universe(["AAPL", "BTCUSD", "CASH"])
    env.reset()

    prev_weights = torch.tensor([0.2, 0.1, 0.7], dtype=torch.float32)
    weights = torch.tensor([0.4, 0.0, 0.6], dtype=torch.float32)
    rewards = env.step(weights, torch.zeros_like(weights), prev_weights)

    turnover = smooth_abs(weights - prev_weights, cfg.smooth_abs_eps)
    _, forced_cost = env._enforce_overnight_cap(env._apply_crypto_limits(weights))
    expected_fee = env._turnover_cost(turnover) + forced_cost
    assert torch.isclose(rewards, -expected_fee.to(dtype=torch.float32), atol=5e-7)


def test_env_blocks_crypto_shorting() -> None:
    cfg = EnvironmentConfig(
        transaction_cost=0.0,
        risk_aversion=0.0,
        max_intraday_leverage=2.0,
        max_overnight_leverage=2.0,
    )
    env = DifferentiableMarketEnv(cfg)
    env.set_asset_universe(["BTCUSD"])
    env.reset()

    prev_weights = torch.zeros(1, dtype=torch.float32)
    weights = torch.tensor([-0.5], dtype=torch.float32)
    reward = env.step(weights, torch.zeros_like(weights), prev_weights)
    limited = env._apply_crypto_limits(weights)
    turnover = smooth_abs(limited - prev_weights, cfg.smooth_abs_eps)
    _, forced = env._enforce_overnight_cap(limited)
    expected = -(env._turnover_cost(turnover) + forced)
    assert torch.isclose(reward, expected.to(dtype=torch.float32), atol=5e-7)


def test_env_applies_leverage_interest_daily() -> None:
    cfg = EnvironmentConfig(
        transaction_cost=0.0,
        risk_aversion=0.0,
        max_intraday_leverage=4.0,
        max_overnight_leverage=4.0,
    )
    env = DifferentiableMarketEnv(cfg)
    env.set_asset_universe(["AAPL", "MSFT"])
    env.reset()

    prev_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
    weights = prev_weights.clone()
    reward = env.step(weights, torch.zeros_like(weights), prev_weights)

    daily_rate = math.pow(1.0 + ANNUAL_MARGIN_RATE, 1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    turnover = smooth_abs(weights - prev_weights, cfg.smooth_abs_eps)
    overnight_weights, forced_cost = env._enforce_overnight_cap(weights)
    gross_close = torch.sum(torch.abs(overnight_weights), dim=-1)
    interest = torch.clamp(
        gross_close - torch.tensor(env._base_gross, dtype=torch.float32),
        min=0.0,
    ) * daily_rate
    expected_total = -(env._turnover_cost(turnover) + forced_cost + interest)
    assert torch.isclose(reward, expected_total.to(dtype=torch.float32), atol=5e-7)
