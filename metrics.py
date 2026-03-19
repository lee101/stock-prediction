from __future__ import annotations


def annualize_total_return(total_return: float, *, periods: float, periods_per_year: float) -> float:
    if periods <= 0:
        return 0.0
    mult = 1.0 + float(total_return)
    if mult <= 0.0:
        return -1.0
    periods_per_year = float(periods_per_year)
    if periods_per_year <= 0.0:
        return 0.0
    years = float(periods) / periods_per_year
    if years <= 0.0:
        return 0.0
    return float(mult ** (1.0 / years) - 1.0)


def stock_market_hours_per_year(*, trading_days_per_year: float = 252.0, hours_per_day: float = 6.5) -> float:
    trading_days_per_year = max(0.0, float(trading_days_per_year))
    hours_per_day = max(0.0, float(hours_per_day))
    return float(trading_days_per_year * hours_per_day)
