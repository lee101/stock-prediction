"""Utilities for evaluating PnL-focused trading agents and RL rewards."""

from .benchmarks import buy_and_hold_returns, oracle_long_flat_returns, oracle_long_short_returns
from .metrics import (
    PnlMetrics,
    TradeStats,
    annualized_return_from_returns,
    drawdown_curve,
    equity_to_returns,
    max_drawdown,
    pnl_metrics,
    profit_factor,
    total_return,
    trade_stats,
)
from .rewards import DrawdownTracker, RewardState, RunningMoments, risk_adjusted_reward, sharpe_like_reward
from .targets import PnlTargetResult, PnlTargets, evaluate_targets

__all__ = [
    "PnlMetrics",
    "TradeStats",
    "DrawdownTracker",
    "RewardState",
    "RunningMoments",
    "PnlTargetResult",
    "PnlTargets",
    "annualized_return_from_returns",
    "buy_and_hold_returns",
    "drawdown_curve",
    "equity_to_returns",
    "max_drawdown",
    "oracle_long_flat_returns",
    "oracle_long_short_returns",
    "pnl_metrics",
    "profit_factor",
    "evaluate_targets",
    "risk_adjusted_reward",
    "sharpe_like_reward",
    "total_return",
    "trade_stats",
]
