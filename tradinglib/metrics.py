from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from src.metrics_utils import annualized_sharpe, annualized_sortino, compute_step_returns


@dataclass(frozen=True)
class TradeStats:
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float
    expectancy: float


@dataclass(frozen=True)
class PnlMetrics:
    total_return: float
    annualized_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    profit_factor: float
    avg_return: float
    volatility: float


def equity_to_returns(equity_curve: Iterable[float]) -> np.ndarray:
    """Convert an equity curve into step returns.

    Uses a safe divide to avoid infinities when equity is zero.
    """
    return compute_step_returns(equity_curve)


def total_return(equity_curve: Iterable[float]) -> float:
    series = np.asarray(list(equity_curve), dtype=np.float64)
    if series.size < 2:
        return 0.0
    start = series[0]
    end = series[-1]
    if not np.isfinite(start) or not np.isfinite(end) or start == 0.0:
        return 0.0
    return float((end - start) / start)


def annualized_return_from_returns(returns: Iterable[float], periods_per_year: float) -> float:
    arr = np.asarray(list(returns), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    compounded = float(np.prod(1.0 + arr))
    if compounded <= 0.0:
        return 0.0
    years = arr.size / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float(compounded ** (1.0 / years) - 1.0)


def drawdown_curve(equity_curve: Iterable[float]) -> np.ndarray:
    series = np.asarray(list(equity_curve), dtype=np.float64)
    if series.size == 0:
        return np.array([], dtype=np.float64)
    running_max = np.maximum.accumulate(series)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(running_max > 0, (series - running_max) / running_max, 0.0)
    return drawdown


def max_drawdown(equity_curve: Iterable[float]) -> float:
    drawdown = drawdown_curve(equity_curve)
    if drawdown.size == 0:
        return 0.0
    return float(drawdown.min())


def profit_factor(returns: Iterable[float]) -> float:
    arr = np.asarray(list(returns), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    gains = arr[arr > 0].sum()
    losses = arr[arr < 0].sum()
    if losses == 0.0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / abs(losses))


def trade_stats(trade_returns: Iterable[float]) -> TradeStats:
    arr = np.asarray(list(trade_returns), dtype=np.float64)
    if arr.size == 0:
        return TradeStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = float(wins.size / arr.size)
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    payoff_ratio = float(avg_win / abs(avg_loss)) if avg_win != 0.0 and avg_loss < 0.0 else 0.0
    expectancy = float(arr.mean())
    return TradeStats(
        num_trades=int(arr.size),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        expectancy=expectancy,
    )


def pnl_metrics(
    *,
    equity_curve: Iterable[float],
    periods_per_year: float,
) -> PnlMetrics:
    series = np.asarray(list(equity_curve), dtype=np.float64)
    returns = equity_to_returns(series)
    avg_return = float(returns.mean()) if returns.size else 0.0
    volatility = float(returns.std(ddof=1)) if returns.size > 1 else 0.0

    ann_return = annualized_return_from_returns(returns, periods_per_year=periods_per_year)
    sharpe = annualized_sharpe(returns, periods_per_year=periods_per_year)
    sortino = annualized_sortino(returns, periods_per_year=periods_per_year)
    mdd = max_drawdown(series)
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else 0.0
    pf = profit_factor(returns)

    return PnlMetrics(
        total_return=total_return(series),
        annualized_return=ann_return,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        profit_factor=pf,
        avg_return=avg_return,
        volatility=volatility,
    )


__all__ = [
    "PnlMetrics",
    "TradeStats",
    "annualized_return_from_returns",
    "drawdown_curve",
    "equity_to_returns",
    "max_drawdown",
    "pnl_metrics",
    "profit_factor",
    "total_return",
    "trade_stats",
]
