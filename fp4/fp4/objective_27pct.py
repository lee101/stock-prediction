"""27%/month PnL target + max-drawdown constraint feeders (Phase 6, Unit P6-2).

Produces differentiable scalar tensors suitable for plugging into the
existing P4-3 `Lagrangian.apply({...})` constraint dict. Both losses are
zero when the target is met or beaten, strictly positive otherwise, and
carry a live autograd path through the equity/drawdown tensor.

All tensor ops are GPU-friendly (no `.item()`, no `.cpu()`) so they can
be used inside CUDA graphs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# A trading month is ~21 trading days for daily bars. We stay consistent
# with pufferlib_market's annualisation (252 trading days / year).
BARS_PER_YEAR = 252.0
MONTHS_PER_YEAR = 12.0
BARS_PER_MONTH = BARS_PER_YEAR / MONTHS_PER_YEAR  # 21.0


def monthly_return_value(
    equity_curve: torch.Tensor,
    bars_per_month: float = BARS_PER_MONTH,
) -> torch.Tensor:
    """Estimated monthly return from an equity curve, averaged over batch axes.

    `equity_curve` has shape `[..., T]` with T >= 2; the per-bar compounding
    rate is `(eq[-1] / eq[0]) ** (1/T)`, raised to `bars_per_month` for the
    monthly return. Returns a scalar tensor (zero with live grad when T < 2).
    """
    if equity_curve.dim() < 1:
        raise ValueError(
            f"monthly_return_value: expected >=1-d input, got shape {tuple(equity_curve.shape)}"
        )
    T = equity_curve.shape[-1]
    if T < 2:
        return equity_curve.sum() * 0.0
    eq0 = equity_curve[..., 0].clamp_min(1e-8)
    eqT = equity_curve[..., -1].clamp_min(1e-8)
    log_total = torch.log(eqT) - torch.log(eq0)
    log_monthly = log_total * (float(bars_per_month) / float(T))
    return torch.expm1(log_monthly).mean()


def monthly_pnl_loss(
    equity_curve: torch.Tensor,
    target: float = 0.27,
    bars_per_month: float = BARS_PER_MONTH,
) -> torch.Tensor:
    """Return `relu(target - actual_monthly_return)`.

    `equity_curve` is any tensor of shape `[..., T]` where the last axis is
    time (T >= 2). The per-bar compounding rate is estimated as
    `(eq[..., -1] / eq[..., 0]) ** (1/T)`, then raised to `bars_per_month`
    to get the monthly return. Averaged over the leading axes before
    comparing to `target`.

    Zero when `actual_monthly >= target`; strictly positive otherwise.
    """
    actual = monthly_return_value(equity_curve, bars_per_month=bars_per_month)
    target_t = torch.as_tensor(float(target), device=actual.device, dtype=actual.dtype)
    return F.relu(target_t - actual)


def max_dd_loss(
    drawdown_tensor: torch.Tensor,
    target: float = 0.08,
) -> torch.Tensor:
    """Return `relu(actual_max_dd - target)`.

    `drawdown_tensor` is expected to hold non-negative drawdown magnitudes
    (i.e. `1 - equity/peak`). The worst (max) drawdown is averaged over
    any leading batch axes before the hinge.

    Zero when `actual_max_dd <= target`; strictly positive otherwise.
    """
    if drawdown_tensor.numel() == 0:
        return drawdown_tensor.sum() * 0.0
    # If a full drawdown path is supplied, reduce over the time axis first.
    if drawdown_tensor.dim() >= 2:
        worst = drawdown_tensor.amax(dim=-1)
    else:
        worst = drawdown_tensor
    actual = worst.mean()
    return F.relu(actual - torch.as_tensor(float(target), device=actual.device, dtype=actual.dtype))


