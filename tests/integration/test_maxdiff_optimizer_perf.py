import time

import numpy as np
import pandas as pd
import pytest
import torch

from loss_utils import calculate_profit_torch_with_entry_buysell_profit_values
from src.maxdiff_optimizer import optimize_maxdiff_entry_exit, optimize_maxdiff_always_on


def _load_sample_tensors(length: int = 256, device: torch.device | None = None):
    device = device or torch.device("cpu")
    df = pd.read_csv("trainingdata/AAPL.csv").tail(length + 1).reset_index(drop=True)
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close_moves = np.diff(close) / close[:-1]
    high_moves = np.diff(high) / high[:-1]
    low_moves = np.diff(low) / low[:-1]

    # Predictions are slightly conservative versions of the actual moves
    high_pred = high_moves * 0.85
    low_pred = low_moves * 0.85

    maxdiff_trades = np.sign(close_moves)
    maxdiff_trades[maxdiff_trades == 0] = 1.0

    tensors = {
        "close_actual": torch.tensor(close_moves, dtype=torch.float32, device=device),
        "high_actual": torch.tensor(high_moves, dtype=torch.float32, device=device),
        "low_actual": torch.tensor(low_moves, dtype=torch.float32, device=device),
        "high_pred": torch.tensor(high_pred, dtype=torch.float32, device=device),
        "low_pred": torch.tensor(low_pred, dtype=torch.float32, device=device),
        "maxdiff_trades": torch.tensor(maxdiff_trades, dtype=torch.float32, device=device),
    }
    return tensors


def _run_entry_exit(device: torch.device):
    data = _load_sample_tensors(device=device)
    start = time.perf_counter()
    result = optimize_maxdiff_entry_exit(
        data["close_actual"],
        data["maxdiff_trades"],
        data["high_actual"],
        data["high_pred"],
        data["low_actual"],
        data["low_pred"],
        close_at_eod_candidates=[False, True],
        trading_fee=0.0005,
        optim_kwargs={"maxiter": 20, "popsize": 8, "workers": 1},
    )
    duration = time.perf_counter() - start
    baseline = float(result.base_profit.sum().item())
    optimized = float(result.final_profit.sum().item())
    return duration, baseline, optimized


def _run_always_on(device: torch.device, is_crypto: bool = False):
    data = _load_sample_tensors(device=device)
    length = data["close_actual"].numel()
    buy_indicator = torch.ones(length, dtype=torch.float32, device=device)
    sell_indicator = torch.zeros(length, dtype=torch.float32, device=device) if is_crypto else -torch.ones(length, dtype=torch.float32, device=device)

    baseline_buy = calculate_profit_torch_with_entry_buysell_profit_values(
        data["close_actual"],
        data["high_actual"],
        data["high_pred"],
        data["low_actual"],
        data["low_pred"],
        buy_indicator,
        trading_fee=0.0005,
    )
    if is_crypto:
        baseline_sell = torch.zeros_like(baseline_buy)
    else:
        baseline_sell = calculate_profit_torch_with_entry_buysell_profit_values(
            data["close_actual"],
            data["high_actual"],
            data["high_pred"],
            data["low_actual"],
            data["low_pred"],
            sell_indicator,
            trading_fee=0.0005,
        )

    start = time.perf_counter()
    result = optimize_maxdiff_always_on(
        data["close_actual"],
        buy_indicator,
        sell_indicator,
        data["high_actual"],
        data["high_pred"],
        data["low_actual"],
        data["low_pred"],
        is_crypto=is_crypto,
        close_at_eod_candidates=[False, True],
        trading_fee=0.0005,
        optim_kwargs={"maxiter": 15, "popsize": 6, "workers": 1},
    )
    duration = time.perf_counter() - start
    baseline = float((baseline_buy + baseline_sell).sum().item())
    optimized = float((result.buy_returns + result.sell_returns).sum().item())
    return duration, baseline, optimized


@pytest.mark.integration
@pytest.mark.parametrize("runner", [_run_entry_exit, _run_always_on])
def test_maxdiff_optimizer_improves_pnl_cpu(runner):
    duration, baseline, optimized = runner(torch.device("cpu"))
    assert optimized >= baseline - 5e-4
    assert optimized > 1e-4
    assert duration < 1.5, f"CPU optimization took too long: {duration:.2f}s"


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU benchmark")
@pytest.mark.parametrize("runner", [_run_entry_exit, _run_always_on])
def test_maxdiff_optimizer_gpu_vs_cpu(runner):
    cpu_duration, cpu_baseline, cpu_optimized = runner(torch.device("cpu"))
    # Warm-up GPU call to avoid first-use penalty
    runner(torch.device("cuda"))
    gpu_duration, gpu_baseline, gpu_optimized = runner(torch.device("cuda"))

    assert abs(cpu_baseline - gpu_baseline) < 1e-5
    assert abs(cpu_optimized - gpu_optimized) < 1e-5
    assert gpu_duration <= cpu_duration * 1.5
