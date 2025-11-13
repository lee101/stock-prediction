import importlib

import numpy as np
import torch
from loss_utils import calculate_trading_profit_torch_with_entry_buysell


def _reload_fast(monkeypatch, steps="11,7,5"):
    monkeypatch.setenv("MARKETSIM_USE_TORCH_GRID", "1")
    monkeypatch.setenv("MARKETSIM_TORCH_GRID_STEPS", steps)
    monkeypatch.setenv("MARKETSIM_TORCH_GRID_SHRINK", "0.6")
    monkeypatch.setenv("MARKETSIM_TORCH_GRID_MIN_WINDOW", "1e-3")
    # Ensure nevergrad path does not short-circuit
    module = importlib.import_module("src.optimization_utils_fast")
    return importlib.reload(module)


def _sample_data(n=48, seed=7):
    rng = np.random.default_rng(seed)
    close_actual = torch.tensor(rng.normal(0, 0.01, size=n), dtype=torch.float32)
    swings = torch.tensor(rng.normal(0.015, 0.005, size=n), dtype=torch.float32)
    high_actual = close_actual + torch.abs(swings)
    low_actual = close_actual - torch.abs(swings)
    high_pred = close_actual + 0.008 + torch.tensor(rng.normal(0, 0.002, size=n), dtype=torch.float32)
    low_pred = close_actual - 0.008 + torch.tensor(rng.normal(0, 0.002, size=n), dtype=torch.float32)
    positions = torch.sign(torch.tensor(rng.normal(0, 1, size=n), dtype=torch.float32))
    positions[positions == 0] = 1
    return close_actual, positions, high_actual, high_pred, low_actual, low_pred


def test_grid_optimizer_improves_profit(monkeypatch):
    fast = _reload_fast(monkeypatch)
    close_actual, positions, high_actual, high_pred, low_actual, low_pred = _sample_data()

    baseline_profit = calculate_trading_profit_torch_with_entry_buysell(
        None,
        None,
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
    ).item()

    h_mult, l_mult, opt_profit = fast.optimize_entry_exit_multipliers(
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        maxiter=10,
        popsize=6,
    )

    assert opt_profit >= baseline_profit, "Grid optimizer should not degrade PnL"
    assert -0.03 <= h_mult <= 0.03
    assert -0.03 <= l_mult <= 0.03


def test_grid_optimizer_matches_scipy(monkeypatch):
    fast = _reload_fast(monkeypatch, steps="9,7,5")
    close_actual, positions, high_actual, high_pred, low_actual, low_pred = _sample_data(seed=11)

    h_fast, l_fast, p_fast = fast.optimize_entry_exit_multipliers(
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        maxiter=12,
        popsize=6,
    )

    from src import optimization_utils as scipy_utils

    h_ref, l_ref, p_ref = scipy_utils.optimize_entry_exit_multipliers(
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        maxiter=20,
        popsize=8,
    )

    assert abs(p_fast - p_ref) < 0.1
    assert abs(h_fast - h_ref) < 5e-3
    assert abs(l_fast - l_ref) < 5e-3
