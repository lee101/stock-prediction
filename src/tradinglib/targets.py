from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .metrics import PnlMetrics


@dataclass(frozen=True)
class PnlTargets:
    sharpe: float = 1.0
    sortino: float = 1.5
    max_drawdown: float = -0.2
    profit_factor: float = 1.2
    annualized_return: float = 0.15


@dataclass(frozen=True)
class PnlTargetResult:
    score: float
    passed: Dict[str, bool]


def evaluate_targets(metrics: PnlMetrics, targets: PnlTargets) -> PnlTargetResult:
    passed = {
        "sharpe": metrics.sharpe >= targets.sharpe,
        "sortino": metrics.sortino >= targets.sortino,
        "max_drawdown": metrics.max_drawdown >= targets.max_drawdown,
        "profit_factor": metrics.profit_factor >= targets.profit_factor,
        "annualized_return": metrics.annualized_return >= targets.annualized_return,
    }
    score = sum(1.0 for value in passed.values() if value) / float(len(passed))
    return PnlTargetResult(score=score, passed=passed)


__all__ = ["PnlTargets", "PnlTargetResult", "evaluate_targets"]
