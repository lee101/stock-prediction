"""Rolling-window CVaR portfolio backtest on our daily panel.

At each rebalance date:
  1. take the trailing `fit_window` days of log returns
  2. solve Mean-CVaR LP → weight vector w + cash
  3. hold that allocation for `hold_days` trading days, realising daily returns
  4. at the next rebalance: feed w as `weights_previous` for turnover tracking

Output: per-day portfolio return series + summary (sortino, mean, worst DD, neg frac).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .optimize import OptimResult, solve_cvar_portfolio


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    portfolio_returns: pd.Series  # daily log returns
    weights_history: pd.DataFrame  # one row per rebalance date
    summary: dict
    solve_times: list[float] = field(default_factory=list)


def _sortino(returns: np.ndarray, target: float = 0.0) -> float:
    excess = returns - target
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(excess.mean() / downside.std() * np.sqrt(252))


def _max_drawdown(cum_log_ret: np.ndarray) -> float:
    """Worst peak-to-trough drawdown from a cumulative log-return series."""
    equity = np.exp(cum_log_ret)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def run_backtest(
    prices: pd.DataFrame,
    *,
    fit_window: int = 252,
    hold_days: int = 21,
    num_scen: int = 2500,
    fit_type: str = "kde",
    kde_device: str = "CPU",
    kde_bandwidth: float = 0.01,
    risk_aversion: float = 1.0,
    confidence: float = 0.95,
    w_max: float = 0.10,
    w_min: float = 0.0,
    L_tar: float = 1.0,
    cardinality: Optional[int] = 20,
    c_min: float = 0.0,
    c_max: float = 1.0,
    api: str = "cvxpy",
    alpha_fn: Optional[Callable[[pd.Timestamp, list[str]], np.ndarray]] = None,
    universe_fn: Optional[Callable[[pd.Timestamp, list[str]], list[str]]] = None,
    rng_seed: int = 0,
    verbose: bool = False,
) -> BacktestResult:
    """Run a rolling rebalance backtest.

    Parameters
    ----------
    prices : wide close-price DataFrame (dates x tickers), sorted asc.
    alpha_fn : optional callable(date, tickers) -> np.ndarray expected log-return
        per asset for the upcoming hold period. Use this to inject XGB/RL scores.
    universe_fn : optional callable(date, tickers) -> list[str] restricting the
        LP's asset set at each rebalance. Tickers not in the returned subset
        get zero weight. Use this to feed the LP a short-list of XGB/RL top-K
        picks while still letting the LP size risk.
    """
    log_ret = (np.log(prices) - np.log(prices.shift(1))).fillna(0.0)
    dates = log_ret.index
    tickers = log_ret.columns.tolist()

    # schedule rebalance every `hold_days` starting once we have `fit_window` history
    rebalance_idx = list(range(fit_window, len(dates), hold_days))
    port_rets = pd.Series(0.0, index=dates[fit_window:])
    weights_history = []
    solve_times = []
    prev_w = None

    n_tickers = len(tickers)
    ticker_to_idx = {t: j for j, t in enumerate(tickers)}
    for k, i in enumerate(rebalance_idx):
        window = log_ret.iloc[i - fit_window : i]
        asof = dates[i]

        if universe_fn is not None:
            selected = [t for t in universe_fn(asof, tickers) if t in ticker_to_idx]
            if len(selected) < 2:
                # Nothing to solve — hold all cash this window
                w_full = np.zeros(n_tickers, dtype=float)
                weights_history.append({"date": asof, **{t: 0.0 for t in tickers}, "_cash": 1.0})
                solve_times.append(0.0)
                end_i = min(i + hold_days, len(dates))
                for j in range(i, end_i):
                    port_rets.iloc[j - fit_window] = 0.0
                prev_w = w_full
                if verbose:
                    print(f"[{asof.date()}] no selected syms → hold cash")
                continue
            window_sub = window[selected]
        else:
            selected = tickers
            window_sub = window

        mean_override = None
        if alpha_fn is not None:
            mean_override = alpha_fn(asof, selected)

        res: OptimResult = solve_cvar_portfolio(
            window_sub,
            mean_override=mean_override,
            num_scen=num_scen,
            fit_type=fit_type,
            kde_device=kde_device,
            kde_bandwidth=kde_bandwidth,
            risk_aversion=risk_aversion,
            confidence=confidence,
            w_max=w_max,
            w_min=w_min,
            L_tar=L_tar,
            cardinality=cardinality,
            c_min=c_min,
            c_max=c_max,
            api=api,
            rng_seed=rng_seed + k,
        )
        # Expand selected-sym weights back onto the full ticker vector
        w_full = np.zeros(n_tickers, dtype=float)
        for j_sel, t in enumerate(selected):
            w_full[ticker_to_idx[t]] = float(res.weights[j_sel])
        weights_history.append({"date": asof, **{t: float(w_full[j]) for j, t in enumerate(tickers)}, "_cash": res.cash})
        solve_times.append(res.solve_time)

        # realise return over the next `hold_days` using the full panel
        end_i = min(i + hold_days, len(dates))
        for j in range(i, end_i):
            port_rets.iloc[j - fit_window] = float(np.dot(w_full, log_ret.iloc[j].values))
        prev_w = w_full
        if verbose:
            top5 = np.sort(w_full)[::-1][:5]
            print(
                f"[{asof.date()}] nnz={int((w_full>1e-6).sum())}/{len(selected)} top5_w={np.round(top5,3).tolist()} cash={res.cash:.3f} solve={res.solve_time:.2f}s exp_ret={res.expected_return:.5f} cvar={res.cvar:.4f}"
            )

    cum = port_rets.cumsum()
    summary = {
        "n_days": int(len(port_rets)),
        "mean_daily_log_ret": float(port_rets.mean()),
        "ann_return_pct": float((np.exp(port_rets.mean() * 252) - 1) * 100),
        "ann_vol_pct": float(port_rets.std() * np.sqrt(252) * 100),
        "sortino": _sortino(port_rets.values),
        "max_drawdown_pct": _max_drawdown(cum.values) * 100,
        "neg_day_frac": float((port_rets < 0).mean()),
        "mean_solve_s": float(np.mean(solve_times)) if solve_times else 0.0,
        "rebalances": int(len(rebalance_idx)),
    }
    return BacktestResult(
        dates=port_rets.index,
        portfolio_returns=port_rets,
        weights_history=pd.DataFrame(weights_history).set_index("date"),
        summary=summary,
        solve_times=solve_times,
    )
