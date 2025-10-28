import math

import numpy as np
import pytest
import torch

from pufferlibtraining3.envs.market_env import MarketEnv, MarketEnvConfig
from pufferlibtraining3 import pufferrl


def _build_prices() -> torch.Tensor:
    # Columns: open, high, low, close
    data = torch.tensor(
        [
            [100.0, 101.0, 99.0, 100.5],
            [101.0, 102.5, 100.0, 101.8],
            [102.0, 104.5, 101.5, 103.7],
            [103.0, 105.5, 102.0, 104.2],
            [104.0, 105.9, 103.2, 104.7],
            [105.0, 106.1, 104.4, 105.5],
        ],
        dtype=torch.float32,
    )
    return data


def test_market_env_maxdiff_fills_only_when_limit_touched():
    prices = _build_prices()
    cfg = MarketEnvConfig(
        mode="maxdiff",
        context_len=3,
        horizon=1,
        trading_fee=0.0005,
        slip_bps=1.5,
        maxdiff_limit_scale=0.05,
        maxdiff_deadband=0.01,
        seed=123,
        device="cpu",
    )
    env = MarketEnv(prices=prices, price_columns=("open", "high", "low", "close"), cfg=cfg)
    env.reset()

    action = np.array([3.0, 0.1], dtype=np.float32)
    _, reward, _, _, info = env.step(action)

    assert info["maxdiff_filled"] is True
    limit_price = info["limit_price"]
    expected_limit = 103.0 * (1.0 + math.tanh(0.1) * cfg.maxdiff_limit_scale)
    assert limit_price == pytest.approx(expected_limit, rel=1e-5)

    size = math.tanh(3.0)
    gross_return = (104.2 - expected_limit) / expected_limit
    gross = size * gross_return
    fee_rate = cfg.trading_fee
    slip_rate = cfg.slip_bps / 10_000.0
    total_cost = size * 2.0 * (fee_rate + slip_rate)
    expected_reward = gross - total_cost
    assert reward == pytest.approx(expected_reward, rel=1e-5, abs=1e-6)


def test_market_env_maxdiff_no_fill_without_cross():
    prices = _build_prices()
    cfg = MarketEnvConfig(
        mode="maxdiff",
        context_len=3,
        horizon=1,
        trading_fee=0.0005,
        slip_bps=1.5,
        maxdiff_limit_scale=0.05,
        maxdiff_deadband=0.01,
        seed=321,
        device="cpu",
    )
    env = MarketEnv(prices=prices, price_columns=("open", "high", "low", "close"), cfg=cfg)
    env.reset()

    action = np.array([3.0, 1.0], dtype=np.float32)  # limit well above day's high
    _, reward, _, _, info = env.step(action)

    assert info["maxdiff_filled"] is False
    assert reward == pytest.approx(0.0, abs=1e-9)


def test_pufferrl_build_configs_maps_cli_arguments():
    args = pufferrl.parse_args(
        [
            "--data-root",
            "trainingdata",
            "--symbol",
            "AAPL",
            "--mode",
            "open_close",
            "--is-crypto",
            "false",
            "--device",
            "cpu",
            "--num-envs",
            "4",
        ]
    )
    env_cfg, ppo_cfg, vec_cfg, device = pufferrl.build_configs(args)

    assert env_cfg.symbol == "AAPL"
    assert env_cfg.mode == "open_close"
    assert env_cfg.is_crypto is False
    assert env_cfg.data_root == "trainingdata"
    assert vec_cfg.num_envs == 4
    assert device.type == "cpu"
