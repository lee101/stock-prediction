from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    total_return: float
    sharpe: float
    positions: np.ndarray
    returns: np.ndarray
    scale: float
    cap: float


def _compute_fee_changes(positions: np.ndarray, fee: float) -> np.ndarray:
    # Fee when position direction changes (including from/to zero)
    pos_change = np.diff(np.concatenate(([0.0], positions)))
    # Charge fee per change magnitude (use indicator of change)
    change_fee = (np.abs(pos_change) > 1e-9).astype(float) * fee
    return change_fee


def run_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_crypto: bool = True,
    fee: float = 0.0023,
    scale: float = 1.0,
    cap: float = 0.3,
) -> BacktestResult:
    # Positions are scaled predictions; cap absolute size; disallow negative for crypto shorts
    positions = np.clip(scale * y_pred, -cap, cap)
    if is_crypto:
        positions = np.clip(positions, 0.0, cap)

    fees = _compute_fee_changes(positions, fee)
    rets = positions * y_true - fees

    # Compound: convert single-period pct returns to cumulative return
    # If these are daily returns and small, sum is close; but we keep compounding to be safe
    cumulative = (1.0 + rets).prod() - 1.0
    std = rets.std()
    sharpe = (rets.mean() / std * np.sqrt(252)) if std > 1e-12 else 0.0
    return BacktestResult(float(cumulative), float(sharpe), positions, rets, float(scale), float(cap))


def grid_search_sizing(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_crypto: bool = True,
    fee: float = 0.0023,
    scales: Iterable[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    caps: Iterable[float] = (0.1, 0.2, 0.3, 0.5, 1.0),
) -> BacktestResult:
    best: Tuple[float, float, BacktestResult] | None = None
    for s in scales:
        for c in caps:
            res = run_backtest(y_true, y_pred, is_crypto=is_crypto, fee=fee, scale=s, cap=c)
            key = res.total_return
            if best is None or key > best[0]:
                best = (key, res.sharpe, res)
    return best[2] if best else run_backtest(y_true, y_pred, is_crypto=is_crypto, fee=fee)

