"""Monthly PnL metrics evaluator for equity curves.

Slices a rollout equity curve into calendar months using POSIX-second
timestamps, and reports per-month return statistics plus the fraction of
months hitting the 27%/month production target.

Pure torch.  Bucketisation into calendar months is done via
``torch.searchsorted`` so the per-sample assignment is on-device; the only
host work is (a) pulling the first/last timestamp to enumerate calendar
edges and (b) a small loop over months (typically <=24) to compute
within-month drawdown.  Callers treating this as a hot-path metric should
batch many (B, T) curves through a single call.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import torch

MONTHLY_TARGET = 0.27  # 27%/month production target


def _month_boundaries_posix(ts0: int, ts1: int) -> list[int]:
    """Return POSIX-second boundaries for each calendar-month edge that
    falls inside ``[ts0, ts1]`` (UTC), including ``ts0`` as the first edge.

    Example: ts0=2025-01-15, ts1=2025-04-03 -> [ts0, 2025-02-01, 2025-03-01,
    2025-04-01].
    """
    start = _dt.datetime.fromtimestamp(int(ts0), tz=_dt.timezone.utc)
    end = _dt.datetime.fromtimestamp(int(ts1), tz=_dt.timezone.utc)
    edges: list[int] = [int(ts0)]
    y, m = start.year, start.month
    while True:
        m += 1
        if m > 12:
            m = 1
            y += 1
        edge = _dt.datetime(y, m, 1, tzinfo=_dt.timezone.utc)
        if edge > end:
            break
        edges.append(int(edge.timestamp()))
    return edges


def compute_monthly_metrics(
    equity: torch.Tensor,
    timestamps: torch.Tensor,
) -> dict[str, Any]:
    """Compute monthly metrics from an equity curve.

    Args:
        equity: ``[B, T]`` or ``[T]`` tensor of equity values.  Must be
            strictly positive (equity, not returns).
        timestamps: ``[T]`` int64 POSIX seconds (UTC).  Monotone non-decreasing.

    Returns:
        Dict with keys:
            n_months: number of calendar months covered
            monthly_returns: list[float] mean monthly return across batch
            mean_monthly: float
            p10_monthly: float  (10th percentile across all (batch, month) pairs)
            max_dd_monthly: float (worst within-month drawdown, batch mean)
            hit_27pct: float (fraction of (batch, month) with return >= 0.27)
    """
    if equity.dim() == 1:
        equity = equity.unsqueeze(0)
    if equity.dim() != 2:
        raise ValueError(f"equity must be [B,T] or [T], got shape {tuple(equity.shape)}")
    B, T = equity.shape
    if timestamps.shape != (T,):
        raise ValueError(f"timestamps must be [T={T}], got {tuple(timestamps.shape)}")
    if T < 2:
        return {
            "n_months": 0,
            "monthly_returns": [],
            "mean_monthly": 0.0,
            "p10_monthly": 0.0,
            "max_dd_monthly": 0.0,
            "hit_27pct": 0.0,
        }

    ts_cpu = timestamps.detach().to("cpu", dtype=torch.int64)
    ts0 = int(ts_cpu[0].item())
    ts1 = int(ts_cpu[-1].item())
    edges_list = _month_boundaries_posix(ts0, ts1)
    if len(edges_list) < 2:
        # Less than one full month of data
        return {
            "n_months": 0,
            "monthly_returns": [],
            "mean_monthly": 0.0,
            "p10_monthly": 0.0,
            "max_dd_monthly": 0.0,
            "hit_27pct": 0.0,
        }

    # For each calendar month edge (after the start), find the last index with
    # timestamp < edge; then month-i return = equity[end_i]/equity[start_i] - 1.
    # start_0 = 0.  start_i = end_{i-1}+1 for i>=1 (mapped to the next month's
    # first sample; if gap, we just use start_i = end_{i-1}).
    edges = torch.tensor(edges_list[1:], dtype=torch.int64, device=ts_cpu.device)
    # searchsorted(right=True) gives first index where ts > edge; -1 is the
    # last index with ts <= edge, i.e. a sample exactly at the month boundary
    # is treated as the closing tick of the prior month.
    right = torch.searchsorted(ts_cpu, edges, right=True) - 1
    right = torch.clamp(right, min=0, max=T - 1)
    # Build month index ranges: [start_i, end_i] inclusive.
    # start_i is the sample at the boundary of month i (same index used as the
    # end of month i-1), so month returns telescope cleanly when the underlying
    # curve samples each boundary exactly once.
    starts = torch.empty_like(right)
    starts[0] = 0
    if right.numel() > 1:
        starts[1:] = right[:-1]
    valid = starts <= right
    starts = starts[valid]
    ends = right[valid]
    n_months = int(starts.numel())
    if n_months == 0:
        return {
            "n_months": 0,
            "monthly_returns": [],
            "mean_monthly": 0.0,
            "p10_monthly": 0.0,
            "max_dd_monthly": 0.0,
            "hit_27pct": 0.0,
        }

    eq = equity.detach().to(torch.float64)
    device = eq.device
    starts_d = starts.to(device)
    ends_d = ends.to(device)

    # Gather per-month start/end equity: [B, M]
    eq_start = eq.index_select(1, starts_d)
    eq_end = eq.index_select(1, ends_d)
    # Guard against zero/negative equity.
    eq_start = torch.clamp(eq_start, min=1e-12)
    monthly_ret = (eq_end / eq_start) - 1.0  # [B, M]

    # Within-month max drawdown: iterate months on host (M small, typically <24).
    month_dd = torch.zeros(B, n_months, dtype=torch.float64, device=device)
    for i in range(n_months):
        s = int(starts[i].item())
        e = int(ends[i].item()) + 1
        if e - s < 2:
            continue
        seg = eq[:, s:e]
        cummax = torch.cummax(seg, dim=1).values
        dd = (seg - cummax) / torch.clamp(cummax, min=1e-12)
        month_dd[:, i] = dd.min(dim=1).values  # most negative

    flat = monthly_ret.reshape(-1)
    mean_monthly = float(flat.mean().item())
    # p10 across all (batch, month) pairs
    if flat.numel() > 0:
        k = max(1, int(round(0.10 * flat.numel())))
        p10_monthly = float(torch.kthvalue(flat, k).values.item())
    else:
        p10_monthly = 0.0
    max_dd_monthly = float(month_dd.mean(dim=0).min().item()) if n_months > 0 else 0.0
    hit_27pct = float((flat >= MONTHLY_TARGET).to(torch.float64).mean().item())
    monthly_returns_mean = monthly_ret.mean(dim=0).detach().cpu().tolist()

    return {
        "n_months": n_months,
        "monthly_returns": monthly_returns_mean,
        "mean_monthly": mean_monthly,
        "p10_monthly": p10_monthly,
        "max_dd_monthly": max_dd_monthly,
        "hit_27pct": hit_27pct,
    }
