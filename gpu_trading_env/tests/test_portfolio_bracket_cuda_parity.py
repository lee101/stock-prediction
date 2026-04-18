"""CUDA-vs-numpy golden test for the portfolio bracket env step.

Pins the CUDA kernel at gpu_trading_env/csrc/env_step_portfolio.cu to the
numpy reference at
  pufferlib_cpp_market_sim/python/market_sim_py/multisym_bracket_ref.py
within fp32 rounding on randomized inputs (multiple seeds, varying
leverage / fee / fill-buffer settings).
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

from pufferlib_cpp_market_sim.python.market_sim_py.multisym_bracket_ref import (
    MultiSymBracketConfig,
    step as np_step,
)


def _require_ext():
    ext = gpu_trading_env._load_ext()
    if ext is None:
        pytest.skip(f"gpu_trading_env CUDA ext not built: {gpu_trading_env._EXT_ERR}")
    return ext


def _gen_inputs(B: int, S: int, seed: int):
    rng = np.random.default_rng(seed)
    cash = rng.uniform(5_000.0, 50_000.0, size=(B,)).astype(np.float32)
    positions = rng.normal(0.0, 50.0, size=(B, S)).astype(np.float32)
    # Actions: lim_buy_bps in [-30, 30], lim_sell_bps in [-30, 30],
    # qty_pct in [0, 1.5] (>1 exercises leverage clip).
    actions = np.zeros((B, S, 4), dtype=np.float32)
    actions[..., 0] = rng.uniform(-30.0, 30.0, size=(B, S))
    actions[..., 1] = rng.uniform(-30.0, 30.0, size=(B, S))
    actions[..., 2] = rng.uniform(0.0, 1.2, size=(B, S))
    actions[..., 3] = rng.uniform(0.0, 1.2, size=(B, S))
    prev_close = rng.uniform(20.0, 200.0, size=(B, S)).astype(np.float32)
    bar_open = prev_close * (1.0 + rng.normal(0.0, 0.005, size=(B, S))).astype(np.float32)
    drift = rng.normal(0.0, 0.01, size=(B, S)).astype(np.float32)
    bar_close = prev_close * (1.0 + drift)
    band = np.abs(rng.normal(0.0, 0.015, size=(B, S))).astype(np.float32)
    bar_high = np.maximum(bar_open, bar_close) * (1.0 + band)
    bar_low = np.minimum(bar_open, bar_close) * (1.0 - band)
    tradable = rng.random(size=(B, S)) > 0.05  # ~5% halted
    return (cash, positions, actions, prev_close,
            bar_open.astype(np.float32), bar_high.astype(np.float32),
            bar_low.astype(np.float32), bar_close.astype(np.float32), tradable)


def _run_cuda(ext, cfg: MultiSymBracketConfig, *,
              cash, positions, actions, prev_close, bar_open, bar_high, bar_low, bar_close,
              tradable):
    B, S = positions.shape
    dev = "cuda"
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    cash_in = t(cash)
    pos_in = t(positions)
    actions_t = t(actions)
    prev_close_t = t(prev_close)
    bar_open_t = t(bar_open)
    bar_high_t = t(bar_high)
    bar_low_t = t(bar_low)
    bar_close_t = t(bar_close)
    tradable_t = torch.from_numpy(np.ascontiguousarray(tradable.astype(np.uint8))).to(dev)

    cash_out = torch.empty(B, device=dev, dtype=torch.float32)
    pos_out = torch.empty((B, S), device=dev, dtype=torch.float32)
    reward = torch.empty(B, device=dev, dtype=torch.float32)
    eq_prev = torch.empty(B, device=dev, dtype=torch.float32)
    new_eq = torch.empty(B, device=dev, dtype=torch.float32)
    fees = torch.empty(B, device=dev, dtype=torch.float32)
    margin = torch.empty(B, device=dev, dtype=torch.float32)
    borrowed = torch.empty(B, device=dev, dtype=torch.float32)

    ext.portfolio_bracket_step(
        cash_in, pos_in, actions_t,
        prev_close_t, bar_open_t, bar_high_t, bar_low_t, bar_close_t,
        tradable_t,
        cash_out, pos_out, reward,
        eq_prev, new_eq, fees, margin, borrowed,
        cfg.fee_bps, cfg.fill_buffer_bps, cfg.max_leverage,
        cfg.annual_margin_rate, int(cfg.trading_days_per_year), int(S),
    )
    torch.cuda.synchronize()
    return dict(
        cash=cash_out.cpu().numpy(),
        positions=pos_out.cpu().numpy(),
        reward=reward.cpu().numpy(),
        eq_prev=eq_prev.cpu().numpy(),
        new_eq=new_eq.cpu().numpy(),
        fees=fees.cpu().numpy(),
        margin=margin.cpu().numpy(),
        borrowed=borrowed.cpu().numpy(),
    )


@pytest.mark.parametrize("seed,B,S,cfg_kw", [
    (0, 64, 8,   {}),
    (1, 32, 16,  dict(fee_bps=10.0, fill_buffer_bps=5.0, max_leverage=2.0)),
    (2, 128, 4,  dict(fee_bps=0.0, fill_buffer_bps=0.0, max_leverage=1.0)),
    (3, 16, 32,  dict(fee_bps=2.78, fill_buffer_bps=2.0, max_leverage=1.5,
                      annual_margin_rate=0.08, trading_days_per_year=252)),
    (4, 256, 12, dict(fee_bps=0.278, fill_buffer_bps=5.0, max_leverage=2.0)),
])
def test_cuda_matches_numpy_ref(seed, B, S, cfg_kw):
    ext = _require_ext()
    cfg = MultiSymBracketConfig(**cfg_kw)
    cash, positions, actions, prev_close, bar_open, bar_high, bar_low, bar_close, tradable = \
        _gen_inputs(B, S, seed)

    np_cash, np_pos, np_reward, np_info = np_step(
        cash.copy(), positions.copy(), actions.copy(),
        prev_close, bar_open, bar_high, bar_low, bar_close,
        tradable.copy(), cfg,
    )
    cuda_out = _run_cuda(
        ext, cfg,
        cash=cash, positions=positions, actions=actions,
        prev_close=prev_close, bar_open=bar_open, bar_high=bar_high, bar_low=bar_low,
        bar_close=bar_close, tradable=tradable,
    )

    # Floating point: relative tolerance 1e-4 covers fp32 round + sum-order
    # divergence on Σ over S symbols at S=32.
    np.testing.assert_allclose(cuda_out["positions"], np_pos.astype(np.float32),
                               rtol=2e-4, atol=1e-4)
    np.testing.assert_allclose(cuda_out["cash"], np_cash.astype(np.float32),
                               rtol=2e-4, atol=5e-3)
    # Reward = (new_eq - eq_prev) / eq_prev. When eq_prev ~ 1e4 the numerator
    # is a difference of two near-equal ~1e4 sums → fp32 sum-order divergence
    # floors the absolute tolerance around 1e-5. The leverage-clip alpha path
    # (headroom / delta_notional) adds another div that pushes the relative
    # floor into ~1e-3 on near-boundary cases.
    np.testing.assert_allclose(cuda_out["reward"], np_reward.astype(np.float32),
                               rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(cuda_out["eq_prev"], np_info["equity_prev"].astype(np.float32),
                               rtol=2e-4, atol=5e-3)
    np.testing.assert_allclose(cuda_out["new_eq"], np_info["new_equity"].astype(np.float32),
                               rtol=2e-4, atol=5e-3)
    np.testing.assert_allclose(cuda_out["fees"], np_info["fees"].astype(np.float32),
                               rtol=2e-4, atol=1e-3)
    np.testing.assert_allclose(cuda_out["margin"], np_info["margin_cost"].astype(np.float32),
                               rtol=2e-4, atol=1e-5)
    # `borrowed = max(0, notional_close - equity_prev)` is a ReLU on the diff
    # of two near-equal sums — fp32 summation-order noise can flip 0↔tiny.
    # The downstream margin_cost = borrowed * (rate / 252) ≈ 2.5e-4 already
    # passes its tighter atol of 1e-5, so accept slack here.
    np.testing.assert_allclose(cuda_out["borrowed"], np_info["borrowed"].astype(np.float32),
                               rtol=2e-4, atol=2e-2)
