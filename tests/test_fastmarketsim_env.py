from __future__ import annotations

import math

import numpy as np
import torch

from fastmarketsim import FastMarketEnv


def _make_prices(T: int = 64) -> torch.Tensor:
    timeline = torch.linspace(0, T - 1, steps=T, dtype=torch.float32)
    opens = 100.0 + 0.1 * timeline
    highs = opens + 0.5
    lows = opens - 0.5
    closes = opens + 0.05
    volume = torch.full_like(opens, 1_000_000.0)
    return torch.stack([opens, highs, lows, closes, volume], dim=-1)


def test_crypto_actions_are_long_only():
    prices = _make_prices()
    env = FastMarketEnv(prices=prices, cfg={"context_len": 16, "horizon": 1, "is_crypto": True})

    obs, info = env.reset()
    assert obs.shape == (16, prices.shape[-1] + 3)
    assert np.isfinite(obs).all()

    # Negative action must clamp to 0 exposure for crypto assets.
    obs, reward, terminated, truncated, info = env.step(-1.0)
    assert not terminated and not truncated
    assert math.isclose(info["position"], 0.0, abs_tol=1e-6)
    assert info["trading_cost"] == 0.0
    assert info["deleverage_notional"] == 0.0
    assert math.isclose(info["equity"], 1.0, rel_tol=1e-6)
    assert np.isfinite(reward)


def test_equity_leverage_and_financing_fees():
    prices = _make_prices()
    env = FastMarketEnv(
        prices=prices,
        cfg={
            "context_len": 16,
            "horizon": 1,
            "intraday_leverage_max": 4.0,
            "overnight_leverage_max": 2.0,
            "annual_leverage_rate": 0.065,
            "is_crypto": False,
        },
    )

    env.reset()
    _, reward, _, _, info = env.step(1.0)

    # Intraday target 4x, auto-deleveraged to 2x overnight exposure.
    assert math.isclose(info["position"], 2.0, rel_tol=1e-3)
    assert info["trading_cost"] > 0.0
    assert info["financing_cost"] > 0.0
    assert info["deleverage_cost"] >= 0.0
    assert info["deleverage_notional"] > 0.0
    assert info["equity"] < 1.0
    assert np.isfinite(reward)
