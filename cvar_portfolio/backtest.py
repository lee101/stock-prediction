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


def _rolling_drawdown_stats(port_rets: pd.Series) -> dict:
    """Worst trough-from-rolling-peak drawdown over rolling windows.

    For each window length k, slide a k-day window across the portfolio
    log-return series and compute the min-over-that-window of
    (equity_j / max_equity_in_window − 1) × 100. Report the deepest such
    dip + the fraction of windows where it exceeded common thresholds
    (10/25/40%).

    Windows: 5d ~ 1 week, 21d ~ 1 month, 63d ~ 1 quarter.
    """
    if len(port_rets) == 0:
        return {
            "worst_5d_drawdown_pct": 0.0, "worst_21d_drawdown_pct": 0.0,
            "worst_63d_drawdown_pct": 0.0,
            "frac_5d_dd_gt_10pct": 0.0, "frac_21d_dd_gt_25pct": 0.0,
            "frac_63d_dd_gt_40pct": 0.0,
        }
    equity = np.exp(port_rets.cumsum().values)
    out = {}
    for win_days, label, thresh in [(5, "5d", 10.0), (21, "21d", 25.0), (63, "63d", 40.0)]:
        if len(equity) < win_days:
            out[f"worst_{label}_drawdown_pct"] = 0.0
            out[f"frac_{label}_dd_gt_{int(thresh)}pct"] = 0.0
            continue
        # Rolling max of equity within each trailing win_days-sized window.
        roll_peak = pd.Series(equity).rolling(win_days, min_periods=1).max().values
        dd = (equity - roll_peak) / roll_peak * 100.0  # vector, units = %
        # Worst per-window min: slide a window of size win_days, min of dd.
        dd_series = pd.Series(dd)
        window_min = dd_series.rolling(win_days, min_periods=1).min().values
        out[f"worst_{label}_drawdown_pct"] = float(np.min(window_min))
        out[f"frac_{label}_dd_gt_{int(thresh)}pct"] = float((window_min < -thresh).mean())
    return out


def _goodness_score(summary: dict) -> float:
    """Composite user-defined goodness: reward monthly return, heavily
    penalize rolling drawdowns that breach comfort thresholds.

    Target (per user standing request 2026-04-23): sustain 25–30%/mo
    gains while keeping rolling drawdowns under 20–25%.

        goodness =   median_monthly_return_pct
                   − 2.0 × max(0, |worst_21d_dd| − 25)
                   − 1.5 × max(0, |worst_5d_dd| − 10)
                   − 0.5 × max(0, |worst_63d_dd| − 40)
                   − 5.0 × frac_21d_dd_gt_25pct × 100

    Penalties kick in only when the drawdown exceeds the comfort band,
    and they scale linearly with how deep the breach is. A portfolio
    with +28%/mo median and only shallow rolling DDs will score near
    its median_monthly; one with +28%/mo but repeated 40%+ rolling DDs
    will score strongly negative.
    """
    med = float(summary.get("median_monthly_return_pct", 0.0))
    w5 = abs(float(summary.get("worst_5d_drawdown_pct", 0.0)))
    w21 = abs(float(summary.get("worst_21d_drawdown_pct", 0.0)))
    w63 = abs(float(summary.get("worst_63d_drawdown_pct", 0.0)))
    frac21 = float(summary.get("frac_21d_dd_gt_25pct", 0.0))
    penalty = (
        2.0 * max(0.0, w21 - 25.0)
        + 1.5 * max(0.0, w5 - 10.0)
        + 0.5 * max(0.0, w63 - 40.0)
        + 5.0 * (frac21 * 100.0)
    )
    return float(med - penalty)


def _monthly_stats(port_rets: pd.Series) -> dict:
    if len(port_rets) == 0:
        return {
            "n_months": 0,
            "median_monthly_return_pct": 0.0,
            "p10_monthly_return_pct": 0.0,
            "worst_monthly_return_pct": 0.0,
            "neg_months": 0,
            "worst_monthly_drawdown_pct": 0.0,
            "p90_monthly_drawdown_pct": 0.0,
        }
    month_index = port_rets.index
    if getattr(month_index, "tz", None) is not None:
        month_index = month_index.tz_localize(None)
    monthly_groups = port_rets.groupby(month_index.to_period("M"))
    monthly_ret = monthly_groups.sum().map(lambda x: (np.exp(float(x)) - 1.0) * 100.0)
    monthly_dd = monthly_groups.apply(lambda s: _max_drawdown(s.cumsum().values) * 100.0)
    return {
        "n_months": int(len(monthly_ret)),
        "median_monthly_return_pct": float(np.median(monthly_ret.values)),
        "p10_monthly_return_pct": float(np.quantile(monthly_ret.values, 0.10)),
        "worst_monthly_return_pct": float(np.min(monthly_ret.values)),
        "neg_months": int((monthly_ret < 0.0).sum()),
        "worst_monthly_drawdown_pct": float(np.min(monthly_dd.values)),
        "p90_monthly_drawdown_pct": float(np.quantile(monthly_dd.values, 0.90)),
    }


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
    api: str = "cvxpy",  # "cvxpy" | "cuopt_python" | "pytorch_kelly"
    alpha_fn: Optional[Callable[[pd.Timestamp, list[str]], np.ndarray]] = None,
    universe_fn: Optional[Callable[[pd.Timestamp, list[str]], list[str]]] = None,
    fee_bps: float = 0.0,
    slip_bps: float = 0.0,
    # Adaptive horizon: let the optimizer rebalance at max_hold_days but
    # exit early on per-asset stop-loss. min_hold_days is the minimum
    # gap between forced rebalances.
    min_hold_days: Optional[int] = None,
    per_asset_stop_loss_pct: float = 0.0,
    portfolio_stop_loss_pct: float = 0.0,
    kelly_lr: float = 0.01,
    kelly_steps: int = 1500,
    kelly_l2_reg: float = 0.0,
    kelly_turnover_penalty: float = 0.0,
    kelly_warm_start: bool = False,
    kelly_device: Optional[str] = None,
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
    fee_bps, slip_bps : one-way proportional transaction cost (bps) applied
        per unit of turnover. Total rebalance cost = Σ|Δw| × (fee+slip)/1e4
        is subtracted as a negative log-return on the rebalance day.
        Defaults to zero (backwards-compat); set to e.g. (10, 5) to match
        CLAUDE.md production realism.
    """
    log_ret = (np.log(prices) - np.log(prices.shift(1))).fillna(0.0)
    dates = log_ret.index
    tickers = log_ret.columns.tolist()

    # schedule rebalance every `hold_days` starting once we have `fit_window` history
    rebalance_idx = list(range(fit_window, len(dates), hold_days))
    port_rets = pd.Series(0.0, index=dates[fit_window:])
    weights_history = []
    solve_times = []
    turnovers: list[float] = []
    fee_costs: list[float] = []
    prev_w = None
    cost_per_turnover = (float(fee_bps) + float(slip_bps)) / 1e4

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
                # Pay fees on the liquidation turnover (unwind prev_w → 0)
                turnover = float(np.abs(w_full - (prev_w if prev_w is not None else w_full)).sum())
                fee_cost = turnover * cost_per_turnover
                turnovers.append(turnover)
                fee_costs.append(fee_cost)
                weights_history.append({"date": asof, **{t: 0.0 for t in tickers}, "_cash": 1.0})
                solve_times.append(0.0)
                end_i = min(i + hold_days, len(dates))
                for j in range(i, end_i):
                    port_rets.iloc[j - fit_window] = 0.0
                if fee_cost > 0:
                    port_rets.iloc[i - fit_window] -= fee_cost
                prev_w = w_full
                if verbose:
                    print(f"[{asof.date()}] no selected syms → hold cash  turnover={turnover:.3f} fee={fee_cost*1e4:.1f}bps")
                continue
            window_sub = window[selected]
        else:
            selected = tickers
            window_sub = window

        mean_override = None
        if alpha_fn is not None:
            mean_override = alpha_fn(asof, selected)

        if api == "pytorch_kelly":
            from .differentiable_optimizer import solve_kelly_cvar_portfolio
            prev_w_sub = None
            if prev_w is not None:
                prev_w_sub = np.asarray(
                    [float(prev_w[ticker_to_idx[t]]) for t in selected],
                    dtype=np.float32,
                )
            res: OptimResult = solve_kelly_cvar_portfolio(
                window_sub,
                mean_override=mean_override,
                num_scen=num_scen,
                fit_type=fit_type,
                kde_bandwidth=kde_bandwidth,
                risk_aversion=risk_aversion,
                confidence=confidence,
                w_max=w_max,
                L_tar=L_tar,
                kelly_lr=kelly_lr,
                kelly_steps=kelly_steps,
                kelly_l2_reg=kelly_l2_reg,
                kelly_turnover_penalty=kelly_turnover_penalty,
                kelly_prev_weights=prev_w_sub,
                kelly_warm_start=kelly_warm_start,
                kelly_device=kelly_device,
                rng_seed=rng_seed + k,
            )
        else:
            res = solve_cvar_portfolio(
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

        # Transaction cost: turnover from previous weight vector
        turnover = float(np.abs(w_full - (prev_w if prev_w is not None else np.zeros_like(w_full))).sum())
        fee_cost = turnover * cost_per_turnover
        turnovers.append(turnover)
        fee_costs.append(fee_cost)

        # Realise return over the next `hold_days` using the full panel.
        # Support adaptive early-exit on per-asset / portfolio stop-loss:
        # we track an active weight vector `active_w` that starts at
        # `w_full` and zeroes assets whose cumulative log-return since
        # entry breaches `per_asset_stop_loss_pct`. If the portfolio's
        # cum log-return since rebalance breaches `portfolio_stop_loss_pct`
        # we liquidate everything for the remainder of the window.
        end_i = min(i + hold_days, len(dates))
        active_w = w_full.copy()
        asset_cum_ret = np.zeros(n_tickers, dtype=float)
        port_cum_ret_since_rebal = 0.0
        per_asset_sl = float(per_asset_stop_loss_pct) / 100.0 if per_asset_stop_loss_pct > 0 else 0.0
        port_sl = float(portfolio_stop_loss_pct) / 100.0 if portfolio_stop_loss_pct > 0 else 0.0
        stopped_out_turnover = 0.0
        min_hold = int(min_hold_days) if min_hold_days is not None else 0
        for j in range(i, end_i):
            day_day_ret = log_ret.iloc[j].values
            port_day_ret = float(np.dot(active_w, day_day_ret))
            port_rets.iloc[j - fit_window] = port_day_ret
            # Update per-asset and portfolio running cum log-returns only
            # for still-active assets. Stopped-out assets stay frozen.
            active_mask = active_w > 0.0
            asset_cum_ret[active_mask] += day_day_ret[active_mask]
            port_cum_ret_since_rebal += port_day_ret
            days_held = j - i + 1
            if days_held >= max(1, min_hold):
                if per_asset_sl > 0.0:
                    sl_hit = active_mask & (asset_cum_ret < -per_asset_sl)
                    if sl_hit.any():
                        # Liquidate the tripped assets: pay fee on their weight
                        # and zero active_w entries.
                        stop_turn = float(np.abs(active_w[sl_hit]).sum())
                        stopped_out_turnover += stop_turn
                        fee_hit = stop_turn * cost_per_turnover
                        port_rets.iloc[j - fit_window] -= fee_hit
                        active_w[sl_hit] = 0.0
                if port_sl > 0.0 and port_cum_ret_since_rebal < -port_sl:
                    # Full liquidation: liquidate everything still active.
                    stop_turn = float(np.abs(active_w).sum())
                    if stop_turn > 0:
                        stopped_out_turnover += stop_turn
                        fee_hit = stop_turn * cost_per_turnover
                        port_rets.iloc[j - fit_window] -= fee_hit
                        active_w = np.zeros_like(active_w)
        if fee_cost > 0:
            port_rets.iloc[i - fit_window] -= fee_cost
        # Carry through stop-outs to the next rebalance's turnover calc:
        # next rebalance compares new w vs `active_w` (post-stops), not
        # the original w_full, so we don't double-count the liquidation.
        prev_w = active_w
        if stopped_out_turnover > 0:
            turnovers[-1] += stopped_out_turnover
            fee_costs[-1] += stopped_out_turnover * cost_per_turnover
        if verbose:
            top5 = np.sort(w_full)[::-1][:5]
            print(
                f"[{asof.date()}] nnz={int((w_full>1e-6).sum())}/{len(selected)} top5_w={np.round(top5,3).tolist()} cash={res.cash:.3f} solve={res.solve_time:.2f}s exp_ret={res.expected_return:.5f} cvar={res.cvar:.4f} turnover={turnover:.3f} fee={fee_cost*1e4:.1f}bps"
            )

    cum = port_rets.cumsum()
    monthly = _monthly_stats(port_rets)
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
        "fee_bps": float(fee_bps),
        "slip_bps": float(slip_bps),
        "mean_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "total_fee_cost_log": float(np.sum(fee_costs)),
        "total_fee_cost_pct": float((1.0 - np.exp(-np.sum(fee_costs))) * 100),
        "kelly_lr": float(kelly_lr) if api == "pytorch_kelly" else None,
        "kelly_steps": int(kelly_steps) if api == "pytorch_kelly" else None,
        "kelly_l2_reg": float(kelly_l2_reg) if api == "pytorch_kelly" else None,
        "kelly_turnover_penalty": float(kelly_turnover_penalty) if api == "pytorch_kelly" else None,
        "kelly_warm_start": bool(kelly_warm_start) if api == "pytorch_kelly" else None,
        "kelly_device": str(kelly_device) if (api == "pytorch_kelly" and kelly_device is not None) else None,
        "per_asset_stop_loss_pct": float(per_asset_stop_loss_pct),
        "portfolio_stop_loss_pct": float(portfolio_stop_loss_pct),
        "min_hold_days": int(min_hold_days) if min_hold_days is not None else None,
        "hold_days": int(hold_days),
    }
    summary.update(monthly)
    roll_dd = _rolling_drawdown_stats(port_rets)
    summary.update(roll_dd)
    summary["goodness_score"] = _goodness_score(summary)
    return BacktestResult(
        dates=port_rets.index,
        portfolio_returns=port_rets,
        weights_history=pd.DataFrame(weights_history).set_index("date"),
        summary=summary,
        solve_times=solve_times,
    )
