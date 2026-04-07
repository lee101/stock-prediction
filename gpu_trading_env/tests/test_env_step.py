"""Unit tests for the SoA fused env_step kernel.

Skips with a reason on non-CUDA boxes. On CUDA, asserts invariants
(leverage cap, fees>0 on fills, drawdown monotone in dd_peak) and the
zero-action analytical parity (zero PnL, zero fees, equity unchanged).
"""
import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not cuda_available, reason="gpu_trading_env requires CUDA"
)

import gpu_trading_env


def _require_ext():
    # Trigger the lazy JIT load and skip if it failed.
    ext = gpu_trading_env._load_ext()
    if ext is None:
        pytest.skip(f"gpu_trading_env CUDA ext not built: {gpu_trading_env._EXT_ERR}")


def test_zero_action_parity():
    _require_ext()
    B = 256
    env = gpu_trading_env.make(B=B, T_synth=4096, seed=42)
    init_eq = env.state["equity"].clone()
    a = torch.zeros(B, 4, device="cuda", dtype=torch.float32)
    for _ in range(128):
        obs, r, d, c = env.step(a)
    # q_bid = q_ask = 0 -> no fills -> no fees -> equity unchanged, reward ~0.
    assert torch.allclose(env.state["equity"], init_eq, atol=1e-3), \
        f"zero-action equity drift: {(env.state['equity']-init_eq).abs().max().item()}"
    assert torch.allclose(env.state["pos_qty"], torch.zeros_like(env.state["pos_qty"]))
    assert torch.allclose(env.state["cash"], init_eq, atol=1e-3)


def test_invariants_random_actions():
    _require_ext()
    B = 512
    torch.manual_seed(0)
    env = gpu_trading_env.make(B=B, T_synth=8192, seed=1)
    max_lev = env.cfg.max_leverage
    for step in range(200):
        a = torch.rand(B, 4, device="cuda") * 2 - 1  # in [-1, 1]
        a[:, 2:] = a[:, 2:].abs() * 0.2  # modest sizes
        obs, r, d, c = env.step(a)
        eq = env.state["equity"]
        pos = env.state["pos_qty"]
        dd_peak = env.state["dd_peak"]
        drawdown = env.state["drawdown"]
        # Invariant 1: drawdown in [0,1]
        assert torch.all(drawdown >= -1e-5)
        assert torch.all(drawdown <= 1.0 + 1e-5)
        # Invariant 2: dd_peak >= equity (up to fp eps)
        assert torch.all(dd_peak + 1e-3 >= eq)
        # Invariant 3: gross leverage <= max_leverage * (1 + small slack from force-reduce)
        bar = obs["bar"]
        close = bar[:, 3]
        gross = (pos.abs() * close) / eq.clamp_min(1e-6)
        # Allow slack for liquidated episodes (eq collapsed)
        alive = env.state["done"] == 0
        if alive.any():
            assert torch.all(gross[alive] <= max_lev + 0.25), \
                f"leverage breach: max={gross[alive].max().item()}"
        # Invariant 4: reward is finite
        assert torch.all(torch.isfinite(r))


def test_fills_produce_fees_and_turnover():
    _require_ext()
    B = 64
    env = gpu_trading_env.make(B=B, T_synth=4096, seed=7)
    # Aggressive buy: max inside quote, max size.
    a = torch.zeros(B, 4, device="cuda", dtype=torch.float32)
    a[:, 0] = 1.0  # max inside bid
    a[:, 2] = 1.0  # full size
    obs, r, d, c = env.step(a)
    # turnover > 0 for at least some episodes
    assert (c[:, 3] > 0).any(), "expected turnover > 0 on aggressive buy"
    # And equity slightly < initial due to fees + buffer slippage on the filled ones.
    filled = c[:, 3] > 0
    assert (env.state["equity"][filled] <= env.cfg.init_cash + 1e-3).all()


def test_reset_mask():
    _require_ext()
    B = 32
    env = gpu_trading_env.make(B=B, T_synth=2048)
    a = torch.rand(B, 4, device="cuda") * 0.1
    for _ in range(10):
        env.step(a)
    env.reset()
    assert torch.all(env.state["t_idx"] == 0)
    assert torch.all(env.state["pos_qty"] == 0)
    assert torch.all(env.state["equity"] == env.cfg.init_cash)
