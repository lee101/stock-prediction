#!/usr/bin/env python3
"""Benchmark: current sim (Triton fwd + Python bwd) vs compiled sim+loss fusion."""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from differentiable_loss_utils import (
    simulate_hourly_trades,
    compute_loss_by_type,
    HOURLY_PERIODS_PER_YEAR,
)

try:
    from trainingefficiency.triton_sim_kernel import simulate_hourly_trades_triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    simulate_hourly_trades_triton = None

from trainingefficiency.compiled_sim_loss import compiled_sim_and_loss, _sim_loop_sortino

device = torch.device("cuda")
B, T = 16, 72
torch.manual_seed(42)

closes = (1.0 + 0.01 * torch.randn(B, T, device=device)).cumprod(dim=-1)
highs = closes * (1.0 + 0.005 * torch.rand(B, T, device=device))
lows = closes * (1.0 - 0.005 * torch.rand(B, T, device=device))
buy_prices = closes * 0.998
sell_prices = closes * 1.002
buy_frac = torch.rand(B, T, device=device) * 0.5
sell_frac = torch.rand(B, T, device=device) * 0.5
max_lev = torch.ones(B, T, device=device)
can_short = torch.zeros(B, device=device)
can_long = torch.ones(B, device=device)

# require gradients on model outputs (buy_prices, sell_prices, fractions)
buy_prices.requires_grad_(True)
sell_prices.requires_grad_(True)
buy_frac.requires_grad_(True)
sell_frac.requires_grad_(True)


def bench_current():
    """Current: simulate_hourly_trades (Python or Triton) + compute_loss_by_type."""
    sim_fn = simulate_hourly_trades_triton if HAS_TRITON else simulate_hourly_trades
    sim = sim_fn(
        highs=highs, lows=lows, closes=closes,
        buy_prices=buy_prices, sell_prices=sell_prices,
        trade_intensity=buy_frac,
        buy_trade_intensity=buy_frac,
        sell_trade_intensity=sell_frac,
        maker_fee=0.001, initial_cash=1.0,
        can_short=False, can_long=True,
        max_leverage=1.0, temperature=0.01,
        fill_buffer_pct=0.0005,
    )
    returns = sim.returns.float()
    loss, score, sortino, annual_return = compute_loss_by_type(
        returns, "sortino", periods_per_year=HOURLY_PERIODS_PER_YEAR, return_weight=0.05,
    )
    loss.backward()
    return loss.item()


def bench_compiled():
    """Compiled: fused sim+loss in one torch.compile'd function."""
    loss, score, sortino, annual_return = compiled_sim_and_loss(
        highs=highs, lows=lows, closes=closes,
        buy_prices=buy_prices, sell_prices=sell_prices,
        buy_frac=buy_frac, sell_frac=sell_frac,
        max_leverage=max_lev,
        can_short=can_short, can_long=can_long,
        initial_cash=1.0, maker_fee=0.001,
        temperature=0.01, fill_buffer_pct=0.0005,
        periods_per_year=HOURLY_PERIODS_PER_YEAR, return_weight=0.05,
    )
    loss.backward()
    return loss.item()


def bench_compiled_eager():
    """Eager (non-compiled) fused sim+loss for comparison."""
    loss, score, sortino, annual_return = _sim_loop_sortino(
        closes, highs, lows, buy_prices, sell_prices,
        buy_frac, sell_frac, max_lev, can_short, can_long,
        1.0, 0.0, 1.001, 0.999, 0.01, 0.0005,
        0.0, HOURLY_PERIODS_PER_YEAR, 0.05,
    )
    loss.backward()
    return loss.item()


def run_bench(name, fn, warmup=5, iters=50):
    for p in [buy_prices, sell_prices, buy_frac, sell_frac]:
        if p.grad is not None:
            p.grad = None

    # warmup
    for _ in range(warmup):
        fn()
        for p in [buy_prices, sell_prices, buy_frac, sell_frac]:
            if p.grad is not None:
                p.grad = None
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        for p in [buy_prices, sell_prices, buy_frac, sell_frac]:
            if p.grad is not None:
                p.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times) * 1000
    std = (sum((t - avg/1000)**2 for t in times) / len(times))**0.5 * 1000
    med = sorted(times)[len(times)//2] * 1000
    print(f"{name:30s}: avg={avg:.2f}ms  med={med:.2f}ms  std={std:.2f}ms")
    return avg


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: B={B}, T={T}")
    print(f"Triton sim available: {HAS_TRITON}")
    print()

    t_current = run_bench("current (triton fwd+py bwd)", bench_current)
    t_eager = run_bench("eager fused sim+loss", bench_compiled_eager)

    print("\nCompiling fused sim+loss (may take 30-60s)...")
    # trigger compilation
    bench_compiled()
    for p in [buy_prices, sell_prices, buy_frac, sell_frac]:
        if p.grad is not None:
            p.grad = None

    t_compiled = run_bench("compiled fused sim+loss", bench_compiled)

    print(f"\nSpeedup (compiled vs current): {t_current/t_compiled:.2f}x")
    print(f"Speedup (compiled vs eager):   {t_eager/t_compiled:.2f}x")
