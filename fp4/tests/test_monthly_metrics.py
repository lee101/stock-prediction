"""Unit tests for fp4.bench.monthly_metrics."""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fp4.bench.monthly_metrics import MONTHLY_TARGET, compute_monthly_metrics


def _month_start(y: int, m: int) -> int:
    return int(_dt.datetime(y, m, 1, tzinfo=_dt.timezone.utc).timestamp())


def _build_curve(month_returns: list[float], samples_per_month: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an equity curve with piecewise-linear growth hitting the given
    end-of-month returns.  Returns (equity[1,T], timestamps[T])."""
    eq: list[float] = []
    ts: list[int] = []
    cur = 1.0
    y, m = 2025, 1
    # Seed the first sample at exactly month 1 start.
    eq.append(cur)
    ts.append(_month_start(y, m))
    for r in month_returns:
        start_ts = _month_start(y, m)
        # next month
        nm = m + 1
        ny = y
        if nm > 12:
            nm = 1
            ny += 1
        end_ts = _month_start(ny, nm)
        target = cur * (1.0 + r)
        # Intra-month samples at fracs 1/(N+1) .. N/(N+1); the month boundary
        # (equity = target, ts = next month start) is appended separately as
        # the "final tick" so every month closes exactly at its true end equity.
        for k in range(samples_per_month):
            frac = (k + 1) / (samples_per_month + 1)
            eq.append(cur + (target - cur) * frac)
            ts.append(int(start_ts + (end_ts - start_ts) * frac))
        # Boundary tick: end-of-month equity at next-month start timestamp.
        eq.append(target)
        ts.append(end_ts)
        cur = target
        y, m = ny, nm
    return torch.tensor(eq, dtype=torch.float64).unsqueeze(0), torch.tensor(ts, dtype=torch.int64)


def test_monthly_returns_analytic_three_months():
    returns = [0.30, 0.10, 0.27]  # 2 hits of 27% target
    eq, ts = _build_curve(returns)
    out = compute_monthly_metrics(eq, ts)
    assert out["n_months"] == 3
    assert len(out["monthly_returns"]) == 3
    for observed, expected in zip(out["monthly_returns"], returns):
        assert observed == pytest.approx(expected, rel=1e-3, abs=1e-3)
    assert out["mean_monthly"] == pytest.approx(sum(returns) / 3, rel=1e-3, abs=1e-3)
    # 0.30 and 0.27 both >= 0.27
    assert out["hit_27pct"] == pytest.approx(2.0 / 3.0, rel=0, abs=1e-9)


def test_monthly_returns_all_hit():
    returns = [0.30, 0.28, 0.35, 0.27]
    eq, ts = _build_curve(returns)
    out = compute_monthly_metrics(eq, ts)
    assert out["hit_27pct"] == pytest.approx(1.0)
    assert out["mean_monthly"] == pytest.approx(sum(returns) / 4, rel=1e-3, abs=1e-3)
    assert out["n_months"] == 4


def test_monthly_returns_none_hit():
    returns = [0.05, -0.02, 0.10]
    eq, ts = _build_curve(returns)
    out = compute_monthly_metrics(eq, ts)
    assert out["hit_27pct"] == pytest.approx(0.0)
    assert out["mean_monthly"] == pytest.approx(sum(returns) / 3, rel=1e-3, abs=1e-3)


def test_short_curve_returns_empty():
    eq = torch.tensor([[1.0]])
    ts = torch.tensor([_month_start(2025, 1)], dtype=torch.int64)
    out = compute_monthly_metrics(eq, ts)
    assert out["n_months"] == 0
    assert out["hit_27pct"] == 0.0


def test_batched_curve_mean_over_batch():
    r_a = [0.30, 0.30]
    r_b = [0.10, 0.10]
    eq_a, ts = _build_curve(r_a)
    eq_b, _ = _build_curve(r_b)
    eq = torch.cat([eq_a, eq_b], dim=0)
    out = compute_monthly_metrics(eq, ts)
    assert out["n_months"] == 2
    # mean over (2 batch * 2 months) = mean(0.30,0.30,0.10,0.10) = 0.20
    assert out["mean_monthly"] == pytest.approx(0.20, rel=1e-3, abs=1e-3)
    # 2/4 months hit
    assert out["hit_27pct"] == pytest.approx(0.5)
