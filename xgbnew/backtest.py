"""Backtest simulation for XGBoost daily open-to-close strategy.

Cost model (per trade):
    total_cost = spread_bps + 2 * commission_bps   (round-trip, in bps)

With leverage L (default 1.0):
    gross_return_leveraged  = L * actual_oc_return
    total_cost_leveraged    = L * (spread_bps + 2*commission_bps) / 10_000
    margin_cost             ≈ (L - 1) * annual_rate / 252  (negligible intraday)
    net_return              = gross_return_leveraged - total_cost_leveraged - margin_cost

Selection logic:
    Each day, score every stock using the XGBStockModel + optional Chronos2 blend.
    Pick the top-N by score.  Among ties, prefer higher Chronos2 oc_return.
    Only pick stocks that actually have an open/close recorded for that day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .features import CHRONOS_FEATURE_COLS, DAILY_FEATURE_COLS
from .model import XGBStockModel, combined_scores

logger = logging.getLogger(__name__)

ANNUAL_MARGIN_RATE = 0.0625   # 6.25% per year (Alpaca rate)
TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestConfig:
    top_n: int = 2
    initial_cash: float = 10_000.0
    commission_bps: float = 10.0        # per side
    leverage: float = 1.0               # 1.0 = no leverage, max 2.0
    xgb_weight: float = 0.5             # blend weight for XGB vs Chronos2
    min_score: float = 0.0              # min combined score to trade
    min_dollar_vol: float = 1e6         # skip illiquid stocks (min avg daily $ vol)
    chronos_col: str = "chronos_oc_return"


@dataclass
class DayTrade:
    symbol: str
    score: float
    actual_open: float
    actual_close: float
    spread_bps: float
    commission_bps: float
    leverage: float
    gross_return_pct: float
    net_return_pct: float


@dataclass
class DayResult:
    day: date
    equity_start: float
    equity_end: float
    daily_return_pct: float
    trades: list[DayTrade] = field(default_factory=list)
    n_candidates: int = 0


@dataclass
class BacktestResult:
    config: BacktestConfig
    day_results: list[DayResult]
    initial_cash: float
    final_equity: float
    total_return_pct: float
    monthly_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    avg_spread_bps: float
    directional_accuracy_pct: float  # % of picks that had gross_return > 0


def _day_margin_cost(leverage: float) -> float:
    """Intraday margin cost fraction (open-to-close only)."""
    if leverage <= 1.0:
        return 0.0
    return (leverage - 1.0) * ANNUAL_MARGIN_RATE / TRADING_DAYS_PER_YEAR


def simulate(
    test_df: pd.DataFrame,
    model: XGBStockModel,
    config: BacktestConfig,
) -> BacktestResult:
    """Run the backtest on ``test_df``.

    ``test_df`` must have columns: date, symbol, actual_open, actual_close,
    spread_bps, dolvol_20d_log, target_oc, + DAILY_FEATURE_COLS + CHRONOS_FEATURE_COLS.
    """
    required = {"date", "symbol", "actual_open", "actual_close", "spread_bps"}
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"test_df missing columns: {missing}")

    # Compute combined scores for every row
    scores = combined_scores(
        test_df, model,
        xgb_weight=config.xgb_weight,
        chronos_col=config.chronos_col,
    )
    test_df = test_df.copy()
    test_df["_score"] = scores.values

    # Liquidity filter
    min_dolvol_log = np.log1p(config.min_dollar_vol)
    if "dolvol_20d_log" in test_df.columns:
        test_df = test_df[test_df["dolvol_20d_log"] >= min_dolvol_log]

    # Drop rows without valid actual prices
    test_df = test_df.dropna(subset=["actual_open", "actual_close"])
    test_df = test_df[(test_df["actual_open"] > 0) & (test_df["actual_close"] > 0)]

    equity = config.initial_cash
    day_results: list[DayResult] = []

    for day, day_df in test_df.groupby("date", sort=True):
        # Rank by combined score, descending
        day_df = day_df.sort_values("_score", ascending=False)

        picks = day_df[day_df["_score"] >= config.min_score].head(config.top_n * 3)

        trades: list[DayTrade] = []
        for _, row in picks.iterrows():
            if len(trades) >= config.top_n:
                break

            o = float(row["actual_open"])
            c = float(row["actual_close"])
            if o <= 0 or c <= 0:
                continue

            spread = float(row.get("spread_bps", 25.0))
            if not np.isfinite(spread) or spread <= 0:
                spread = 25.0

            gross_oc = (c - o) / o
            gross_leveraged = config.leverage * gross_oc

            cost_frac = (
                config.leverage * (spread + 2.0 * config.commission_bps) / 10_000.0
                + _day_margin_cost(config.leverage)
            )
            net = gross_leveraged - cost_frac

            trades.append(DayTrade(
                symbol=str(row["symbol"]),
                score=float(row["_score"]),
                actual_open=o,
                actual_close=c,
                spread_bps=spread,
                commission_bps=config.commission_bps,
                leverage=config.leverage,
                gross_return_pct=gross_oc * 100.0,
                net_return_pct=net * 100.0,
            ))

        if not trades:
            continue

        daily_ret_pct = float(np.mean([t.net_return_pct for t in trades]))
        equity_end = equity * (1.0 + daily_ret_pct / 100.0)

        day_results.append(DayResult(
            day=day,  # type: ignore[arg-type]
            equity_start=equity,
            equity_end=equity_end,
            daily_return_pct=daily_ret_pct,
            trades=trades,
            n_candidates=len(day_df),
        ))
        equity = equity_end

    return _compute_result(day_results, config)


def _compute_result(day_results: list[DayResult], config: BacktestConfig) -> BacktestResult:
    if not day_results:
        return BacktestResult(
            config=config, day_results=[], initial_cash=config.initial_cash,
            final_equity=config.initial_cash, total_return_pct=0.0,
            monthly_return_pct=0.0, annualized_return_pct=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown_pct=0.0,
            win_rate_pct=0.0, total_trades=0, avg_spread_bps=0.0,
            directional_accuracy_pct=0.0,
        )

    rets = np.array([r.daily_return_pct / 100.0 for r in day_results])
    eq = np.array([config.initial_cash] + [r.equity_end for r in day_results])
    n = len(day_results)

    total_ret = (eq[-1] - eq[0]) / eq[0]
    ann_ret = (1.0 + total_ret) ** (TRADING_DAYS_PER_YEAR / n) - 1.0
    monthly_ret = (1.0 + total_ret) ** (21.0 / n) - 1.0

    mean_r = float(np.mean(rets))
    std_r  = float(np.std(rets, ddof=1)) if n > 1 else 1e-9
    sharpe = mean_r / std_r * np.sqrt(TRADING_DAYS_PER_YEAR) if std_r > 0 else 0.0

    down_r = rets[rets < 0]
    down_std = float(np.std(down_r, ddof=1)) if len(down_r) > 1 else 1e-9
    sortino = mean_r / down_std * np.sqrt(TRADING_DAYS_PER_YEAR) if down_std > 0 else 0.0

    running_max = np.maximum.accumulate(eq)
    max_dd = float(np.abs(np.min((eq - running_max) / running_max)))

    win_rate = float(np.mean(rets > 0)) * 100.0

    all_trades = [t for r in day_results for t in r.trades]
    spreads = [t.spread_bps for t in all_trades]
    dir_acc = (float(np.mean([t.gross_return_pct > 0 for t in all_trades])) * 100.0
               if all_trades else 0.0)

    return BacktestResult(
        config=config,
        day_results=day_results,
        initial_cash=config.initial_cash,
        final_equity=float(eq[-1]),
        total_return_pct=total_ret * 100.0,
        monthly_return_pct=monthly_ret * 100.0,
        annualized_return_pct=ann_ret * 100.0,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_dd * 100.0,
        win_rate_pct=win_rate,
        total_trades=len(all_trades),
        avg_spread_bps=float(np.mean(spreads)) if spreads else 0.0,
        directional_accuracy_pct=dir_acc,
    )


def print_summary(res: BacktestResult, label: str = "") -> None:
    lbl = f" [{label}]" if label else ""
    print(f"\n{'='*68}")
    print(f"  XGBoost Open→Close Backtest{lbl}")
    print(f"  top_n={res.config.top_n}  leverage={res.config.leverage:.1f}x"
          f"  xgb_weight={res.config.xgb_weight:.2f}")
    print(f"{'='*68}")
    print(f"  Initial cash      : ${res.initial_cash:,.0f}")
    print(f"  Final equity      : ${res.final_equity:,.2f}")
    print(f"  Total return      : {res.total_return_pct:+.2f}%")
    print(f"  Monthly return    : {res.monthly_return_pct:+.2f}%  (21-day equiv)")
    print(f"  Ann. return       : {res.annualized_return_pct:+.2f}%")
    print(f"  Sharpe (ann.)     : {res.sharpe_ratio:.3f}")
    print(f"  Sortino (ann.)    : {res.sortino_ratio:.3f}")
    print(f"  Max drawdown      : {res.max_drawdown_pct:.2f}%")
    print(f"  Win rate          : {res.win_rate_pct:.1f}%")
    print(f"  Directional acc.  : {res.directional_accuracy_pct:.1f}%")
    print(f"  Total trades      : {res.total_trades}")
    print(f"  Trading days      : {len(res.day_results)}")
    print(f"  Avg spread        : {res.avg_spread_bps:.1f} bps")
    print(f"  Commission/side   : {res.config.commission_bps:.0f} bps")
    print(f"{'='*68}")


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "DayResult",
    "DayTrade",
    "simulate",
    "print_summary",
]
