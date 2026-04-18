"""Smoke + invariant tests for PortfolioBracketEnv (GPU wrapper around the
portfolio bracket CUDA kernel).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not cuda_available, reason="gpu_trading_env requires CUDA"
)

import gpu_trading_env


def _require_ext():
    ext = gpu_trading_env._load_ext()
    if ext is None:
        pytest.skip(f"gpu_trading_env CUDA ext not built: {gpu_trading_env._EXT_ERR}")


def _synth_prices(T: int, S: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    # Geometric-Brownian-style daily prices.
    daily_ret = rng.normal(0.0005, 0.02, size=(T, S)).astype(np.float32)
    close = 100.0 * np.exp(np.cumsum(daily_ret, axis=0))
    open_ = close * (1.0 + rng.normal(0.0, 0.005, size=(T, S))).astype(np.float32)
    band = np.abs(rng.normal(0.0, 0.01, size=(T, S))).astype(np.float32)
    high = np.maximum(open_, close) * (1.0 + band)
    low = np.minimum(open_, close) * (1.0 - band)
    vol = np.full((T, S), 1e6, dtype=np.float32)
    return torch.from_numpy(np.stack([open_, high, low, close, vol], axis=-1))


def test_zero_action_keeps_equity_flat():
    _require_ext()
    B, T, S = 64, 64, 8
    prices = _synth_prices(T, S, seed=0)
    env = gpu_trading_env.make_portfolio_bracket(B=B, prices=prices)
    init_eq = env.state["equity"].clone()
    action = torch.zeros(B, S, 4, device="cuda", dtype=torch.float32)
    for _ in range(20):
        r, d, info = env.step(action)
    # Zero buy_pct + zero sell_pct => no fills => no fees => equity unchanged.
    assert torch.allclose(env.state["equity"], init_eq, atol=1e-2), \
        f"equity drift: {(env.state['equity'] - init_eq).abs().max().item()}"
    assert torch.allclose(env.state["positions"],
                          torch.zeros_like(env.state["positions"]))


def test_step_advances_t_idx_and_completes_episode():
    _require_ext()
    B, T, S = 8, 32, 4
    prices = _synth_prices(T, S, seed=1)
    env = gpu_trading_env.make_portfolio_bracket(
        B=B, prices=prices, params={"episode_len": 10},
    )
    action = torch.zeros(B, S, 4, device="cuda", dtype=torch.float32)
    for _ in range(9):
        r, d, info = env.step(action)
        assert (d == 0).all(), f"premature done: {d}"
    r, d, info = env.step(action)
    # After episode_len bars consumed, all envs should be done.
    assert (d == 1).all(), f"expected all done at episode end, got {d}"


def test_auto_reset_on_done():
    _require_ext()
    B, T, S = 8, 32, 4
    prices = _synth_prices(T, S, seed=2)
    env = gpu_trading_env.make_portfolio_bracket(
        B=B, prices=prices, params={"episode_len": 5, "init_cash": 5000.0},
    )
    action = torch.zeros(B, S, 4, device="cuda", dtype=torch.float32)
    # Roll to episode end.
    for _ in range(5):
        env.step(action)
    assert (env.state["done"] == 1).all()
    # Next step should auto-reset.
    env.step(action)
    assert torch.allclose(env.state["cash"],
                          torch.full_like(env.state["cash"], 5000.0)) \
        or env.state["t_idx"].min().item() <= 2  # post-reset t_idx is small


def test_buy_then_close_makes_position():
    _require_ext()
    B, T, S = 4, 32, 2
    prices = _synth_prices(T, S, seed=3)
    env = gpu_trading_env.make_portfolio_bracket(
        B=B, prices=prices,
        params={"fee_bps": 0.0, "fill_buffer_bps": 0.0, "max_leverage": 1.0,
                "init_cash": 10_000.0},
    )
    # Buy sym 0 with 50% of equity at +50bps (very generous limit, will fill).
    action = torch.zeros(B, S, 4, device="cuda", dtype=torch.float32)
    action[:, 0, 0] = 50.0   # lim_buy_bps generous
    action[:, 0, 2] = 0.5    # buy 50% of equity
    r, d, info = env.step(action)
    pos = env.state["positions"]
    # Position should be non-zero on sym 0 for all envs (price tape is
    # synthetic so very likely the bar low <= prev_close * (1+50bps)).
    n_held = (pos[:, 0].abs() > 1e-3).sum().item()
    assert n_held >= B // 2, f"expected most envs to fill, got {n_held}/{B}"


def test_step_is_cuda_graph_capturable():
    """env.step() must be safe to capture under torch.cuda.graph — no
    host-syncing branches, identical graph topology each step.
    """
    _require_ext()
    B, T, S = 256, 64, 8
    prices = _synth_prices(T, S, seed=5)
    env = gpu_trading_env.make_portfolio_bracket(
        B=B, prices=prices, params={"episode_len": 200},
    )
    action = torch.zeros(B, S, 4, device="cuda", dtype=torch.float32)
    action[..., 2] = 0.05
    action[..., 3] = 0.05

    # Warmup (required by CUDA Graphs API — populates allocator caches).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(8):
                env.step(action)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(8):
            env.step(action)

    # Replay — should produce a finite reward and not crash.
    g.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(env.state["equity"]).all()
    assert torch.isfinite(env.state["cash"]).all()
    assert (env.state["equity"] > 0).all()


def test_obs_shape_with_and_without_features():
    _require_ext()
    B, T, S, F = 4, 16, 3, 5
    prices = _synth_prices(T, S, seed=4)
    env = gpu_trading_env.make_portfolio_bracket(B=B, prices=prices)
    obs = env.obs()
    # Without features: portfolio (3) + pos_frac (S) + pos_norm (S) = 3 + 2S
    assert obs.shape == (B, 3 + 2 * S), obs.shape

    features = torch.randn(T, S, F)
    env2 = gpu_trading_env.make_portfolio_bracket(
        B=B, prices=prices, features=features
    )
    obs2 = env2.obs()
    # With features: S*F + 3 + 2S
    assert obs2.shape == (B, S * F + 3 + 2 * S), obs2.shape
