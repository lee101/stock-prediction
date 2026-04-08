"""Tests for fp4.objective_27pct (P6-2)."""
from __future__ import annotations

import pytest
import torch

from fp4.objective_27pct import (
    BARS_PER_MONTH,
    max_dd_loss,
    monthly_pnl_loss,
    monthly_return_value,
)


def _make_equity(monthly_return: float, months: float = 3.0, batch: int = 4) -> torch.Tensor:
    T = max(2, int(round(months * BARS_PER_MONTH)))
    # Construct equity so that log(eq[-1]/eq[0]) = log(1+monthly) * (T/bars_per_month)
    # exactly (up to float32 precision). Use a linear log-equity ramp.
    total_log = torch.log(torch.tensor(1.0 + monthly_return, dtype=torch.float64)) * (
        T / BARS_PER_MONTH
    )
    log_eq = torch.linspace(0.0, float(total_log.item()), T, dtype=torch.float64)
    eq = torch.exp(log_eq).to(torch.float32).unsqueeze(0).expand(batch, T).clone()
    eq.requires_grad_(True)
    return eq


def test_monthly_pnl_loss_zero_when_on_target():
    eq = _make_equity(monthly_return=0.27)
    loss = monthly_pnl_loss(eq, target=0.27)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_monthly_pnl_loss_zero_when_above_target():
    eq = _make_equity(monthly_return=0.40)
    loss = monthly_pnl_loss(eq, target=0.27)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_monthly_pnl_loss_positive_when_below_target():
    eq = _make_equity(monthly_return=0.05)
    loss = monthly_pnl_loss(eq, target=0.27)
    assert loss.item() > 0.0
    # Sanity: differentiable.
    loss.backward()
    assert eq.grad is not None
    assert torch.isfinite(eq.grad).all()


def test_monthly_return_value_matches():
    eq = _make_equity(monthly_return=0.10, months=4.0, batch=2)
    val = monthly_return_value(eq).item()
    assert val == pytest.approx(0.10, abs=1e-4)


def test_max_dd_loss_zero_when_below_target():
    dd = torch.tensor([[0.02, 0.05, 0.03], [0.01, 0.04, 0.02]], requires_grad=True)
    loss = max_dd_loss(dd, target=0.08)
    assert loss.item() == pytest.approx(0.0)


def test_max_dd_loss_positive_when_above_target():
    dd = torch.tensor([[0.02, 0.15, 0.10], [0.01, 0.12, 0.02]], requires_grad=True)
    loss = max_dd_loss(dd, target=0.08)
    # worst per row: 0.15, 0.12 → mean 0.135 → loss 0.055
    assert loss.item() == pytest.approx(0.055, abs=1e-5)
    loss.backward()
    assert dd.grad is not None


def test_max_dd_loss_1d_input():
    dd = torch.tensor([0.02, 0.05, 0.15, 0.01])
    loss = max_dd_loss(dd, target=0.08)
    # mean of drawdowns = 0.0575 → below 0.08 → zero
    assert loss.item() == pytest.approx(0.0)


def test_losses_pluggable_into_lagrangian_apply():
    """Smoke test: both losses are scalar tensors and combine with a
    Lagrangian.apply(...) dict without shape errors."""
    from fp4.lagrangian import Lagrangian

    lag = Lagrangian(
        ["monthly_pnl", "max_dd"],
        target_d={"monthly_pnl": 0.0, "max_dd": 0.0},
        init_lambda=1.0,
    )
    eq = _make_equity(monthly_return=0.05)
    dd = torch.tensor([[0.02, 0.15]], requires_grad=True)
    losses = {
        "monthly_pnl": monthly_pnl_loss(eq, target=0.27),
        "max_dd": max_dd_loss(dd, target=0.08),
    }
    total = lag.apply(losses)
    assert total.dim() == 0
    total.backward()
