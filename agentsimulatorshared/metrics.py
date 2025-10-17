from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReturnMetrics:
    daily_pct: float
    monthly_pct: float
    annual_pct: float


def compute_return_metrics(
    *,
    net_pnl: float,
    starting_nav: float,
    periods: int,
    trading_days_per_month: int = 21,
    trading_days_per_year: int = 252,
) -> ReturnMetrics:
    if starting_nav <= 0:
        raise ValueError("starting_nav must be positive.")
    if periods <= 0:
        raise ValueError("periods must be positive.")

    daily_return = net_pnl / starting_nav / periods
    daily_pct = daily_return * 100.0
    monthly_pct = ((1.0 + daily_return) ** trading_days_per_month - 1.0) * 100.0
    annual_pct = ((1.0 + daily_return) ** trading_days_per_year - 1.0) * 100.0
    return ReturnMetrics(
        daily_pct=daily_pct,
        monthly_pct=monthly_pct,
        annual_pct=annual_pct,
    )


def format_return_metrics(metrics: ReturnMetrics, *, decimals: int = 4) -> str:
    return (
        f"daily={metrics.daily_pct:.{decimals}f}% | "
        f"monthly={metrics.monthly_pct:.{decimals}f}% | "
        f"annual={metrics.annual_pct:.{decimals}f}%"
    )
