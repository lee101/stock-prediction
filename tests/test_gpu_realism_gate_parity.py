"""Parity test for `pufferlib_market.gpu_realism_gate.run_cell_gpu`.

Ground-truth reference is `hourly_replay.simulate_daily_policy` driven by the
same softmax-avg ensemble policy used by `scripts/screened32_realism_gate.py`.
CLAUDE.md rule: "pufferlib C sim with binary fills is ground truth" — the GPU
fast path MUST match on total_return / max_drawdown / sortino to fp32 epsilon
or we silently break the deploy gate.

We sample random window starts from screened32_single_offset_val_full.bin and
run both sims at (fill_buffer=5, leverage=1.0, fee_rate=0.001, slippage=0).
Tolerances:
  * total_return      : abs ≤ 1e-5
  * max_drawdown      : abs ≤ 1e-5
  * sortino           : abs ≤ 5e-4  (sortino divides by noisy downside_std)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
VAL_PATH = REPO / "pufferlib_market/data/screened32_single_offset_val_full.bin"


@pytest.fixture(scope="module")
def env():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not VAL_PATH.exists():
        pytest.skip(f"val data missing: {VAL_PATH}")
    from pufferlib_market.hourly_replay import read_mktd
    from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS
    data = read_mktd(VAL_PATH)
    ckpts = [str(REPO / DEFAULT_CHECKPOINT)] + [str(REPO / p) for p in DEFAULT_EXTRA_CHECKPOINTS]
    return data, ckpts


@pytest.fixture(scope="module")
def window_starts(env):
    data, _ = env
    window_days = 50
    window_len = window_days + 1
    candidate_count = int(data.num_timesteps) - window_len + 1
    assert candidate_count > 10, f"too few candidates: {candidate_count}"
    rng = random.Random(0)
    # mix: some at boundaries + random sample to probe clamp-at-last-bar and t-1 clamping
    starts = sorted(set(
        [0, candidate_count - 1]
        + rng.sample(range(candidate_count), k=min(12, candidate_count))
    ))
    return starts, window_days


def _run_cpu(data, ckpts, starts, window_days, *, fill_buffer_bps, max_leverage,
             fee_rate, slippage_bps, decision_lag=2):
    """Drive CPU simulate_daily_policy the same way screened32_realism_gate does."""
    from pufferlib_market.evaluate_holdout import _slice_window
    from pufferlib_market.hourly_replay import simulate_daily_policy
    # Use the exact same _build_ensemble_policy_fn path the gate uses so
    # parity is pinned against the actual deploy-gate ensemble code, not a
    # re-implementation.
    import sys as _sys
    _sys.path.insert(0, str(REPO))
    from scripts.screened32_realism_gate import _build_ensemble_policy_fn

    device = torch.device("cuda")
    policy_fn, reset_buffer, head = _build_ensemble_policy_fn(
        checkpoints=[Path(p) for p in ckpts],
        num_symbols=int(data.num_symbols),
        features_per_sym=int(data.features.shape[2]),
        decision_lag=int(decision_lag),
        disable_shorts=True,
        device=device,
        deterministic=True,
        ensemble_mode="softmax_avg",
    )
    rets, sorts, dds = [], [], []
    for s in starts:
        window = _slice_window(data, start=int(s), steps=int(window_days))
        reset_buffer()
        res = simulate_daily_policy(
            window, policy_fn,
            max_steps=int(window_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            max_leverage=float(max_leverage),
            periods_per_year=365.0,
            fill_buffer_bps=float(fill_buffer_bps),
            action_allocation_bins=int(head.action_allocation_bins),
            action_level_bins=int(head.action_level_bins),
            action_max_offset_bps=float(head.action_max_offset_bps),
            enable_drawdown_profit_early_exit=False,
        )
        rets.append(float(res.total_return))
        sorts.append(float(res.sortino))
        dds.append(float(res.max_drawdown))
    return np.array(rets), np.array(sorts), np.array(dds)


def _run_gpu(data, ckpts, starts, window_days, *, fill_buffer_bps, max_leverage,
             fee_rate, slippage_bps, decision_lag=2):
    from pufferlib_market.gpu_realism_gate import run_cell_gpu
    res = run_cell_gpu(
        data,
        checkpoints=ckpts,
        num_symbols=int(data.num_symbols),
        features_per_sym=int(data.features.shape[2]),
        starts=starts,
        window_days=int(window_days),
        fill_buffer_bps=float(fill_buffer_bps),
        max_leverage=float(max_leverage),
        fee_rate=float(fee_rate),
        slippage_bps=float(slippage_bps),
        decision_lag=int(decision_lag),
        ensemble_mode="softmax_avg",
    )
    return res.total_returns, res.sortinos, res.max_drawdowns


@pytest.mark.parametrize("cell", [
    {"fill_buffer_bps": 5.0, "max_leverage": 1.0, "fee_rate": 0.001, "slippage_bps": 0.0},
    {"fill_buffer_bps": 0.0, "max_leverage": 1.5, "fee_rate": 0.001, "slippage_bps": 0.0},
    {"fill_buffer_bps": 20.0, "max_leverage": 1.0, "fee_rate": 0.001, "slippage_bps": 10.0},
])
def test_gpu_matches_cpu(env, window_starts, cell):
    data, ckpts = env
    starts, window_days = window_starts
    cpu_ret, cpu_sor, cpu_dd = _run_cpu(data, ckpts, starts, window_days, **cell)
    gpu_ret, gpu_sor, gpu_dd = _run_gpu(data, ckpts, starts, window_days, **cell)

    ret_delta = np.max(np.abs(cpu_ret - gpu_ret))
    dd_delta = np.max(np.abs(cpu_dd - gpu_dd))
    sor_delta = np.max(np.abs(cpu_sor - gpu_sor))

    print(f"\ncell={cell}")
    print(f"  total_return max|delta|={ret_delta:.2e}")
    print(f"  max_drawdown max|delta|={dd_delta:.2e}")
    print(f"  sortino      max|delta|={sor_delta:.2e}")
    print(f"  starts={starts}")
    print(f"  cpu_ret={cpu_ret}")
    print(f"  gpu_ret={gpu_ret}")

    assert ret_delta < 1e-5, f"total_return drift {ret_delta}"
    assert dd_delta < 1e-5, f"max_drawdown drift {dd_delta}"
    assert sor_delta < 5e-4, f"sortino drift {sor_delta}"
