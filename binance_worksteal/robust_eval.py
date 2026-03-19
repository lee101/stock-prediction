"""Robust multi-window evaluation helpers for work-steal strategies."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from binance_worksteal.strategy import WorkStealConfig, run_worksteal_backtest
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, summarize_scenario_results


@dataclass(frozen=True)
class EvaluationWindow:
    label: str
    start_date: str
    end_date: str


def normalize_start_state(raw_state: str) -> str:
    value = str(raw_state or "").strip().upper()
    if not value:
        raise ValueError("start state cannot be empty.")
    if value in {"FLAT", "CASH", "USDT", "FDUSD"}:
        return "FLAT"
    if value.endswith("FDUSD"):
        value = f"{value[:-5]}USD"
    elif value.endswith("USDT"):
        value = f"{value[:-4]}USD"
    elif not value.endswith("USD"):
        value = f"{value}USD"
    return value


def _resolve_starting_holding_quantity(
    *,
    all_bars: dict[str, pd.DataFrame],
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
    starting_equity: float,
) -> float:
    bars = all_bars.get(symbol)
    if bars is None or bars.empty:
        raise ValueError(f"Cannot seed start state {symbol}: no bars loaded.")

    window = bars.copy()
    if start_date:
        window = window[window["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        window = window[window["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
    if window.empty:
        raise ValueError(
            f"Cannot seed start state {symbol}: no bars available in {start_date or 'start'} to {end_date or 'end'}."
        )

    first_price = float(window.iloc[0]["close"])
    if first_price <= 0.0:
        raise ValueError(f"Cannot seed start state {symbol}: invalid first close {first_price}.")
    return float(starting_equity / first_price)


def build_start_state_config(
    *,
    base_config: WorkStealConfig,
    all_bars: dict[str, pd.DataFrame],
    start_state: str,
    start_date: Optional[str],
    end_date: Optional[str],
    starting_equity: float,
) -> tuple[str, WorkStealConfig]:
    normalized = normalize_start_state(start_state)
    if normalized == "FLAT":
        return "flat", replace(base_config, initial_cash=float(starting_equity), initial_holdings={})

    qty = _resolve_starting_holding_quantity(
        all_bars=all_bars,
        symbol=normalized,
        start_date=start_date,
        end_date=end_date,
        starting_equity=starting_equity,
    )
    label = normalized.lower()
    return label, replace(base_config, initial_cash=0.0, initial_holdings={normalized: qty})


def build_recent_windows(
    *,
    end_date: str,
    window_days: int = 60,
    window_count: int = 3,
) -> list[EvaluationWindow]:
    if window_days <= 0:
        raise ValueError(f"window_days must be > 0, got {window_days}.")
    if window_count <= 0:
        raise ValueError(f"window_count must be > 0, got {window_count}.")

    end_ts = pd.Timestamp(end_date, tz="UTC").normalize()
    windows: list[EvaluationWindow] = []
    for idx in range(window_count):
        window_end = end_ts - pd.Timedelta(days=idx * window_days)
        window_start = window_end - pd.Timedelta(days=window_days)
        windows.append(
            EvaluationWindow(
                label=f"w{idx + 1}_{window_start.date()}_{window_end.date()}",
                start_date=str(window_start.date()),
                end_date=str(window_end.date()),
            )
        )
    return windows


def _compute_annualized_return_pct(total_return: float, n_days: float) -> float:
    days = max(float(n_days), 1.0)
    gross = 1.0 + float(total_return)
    if gross <= 0.0:
        return -100.0
    annualized = gross ** (365.0 / days) - 1.0
    return float(annualized * 100.0)


def evaluate_config_scenarios(
    *,
    all_bars: dict[str, pd.DataFrame],
    config: WorkStealConfig,
    windows: Sequence[EvaluationWindow],
    start_states: Iterable[str],
    starting_equity: float | None = None,
    intraday_bars: Optional[dict[str, pd.DataFrame]] = None,
) -> list[dict[str, float | str]]:
    scenario_rows: list[dict[str, float | str]] = []
    base_equity = float(config.initial_cash if starting_equity is None else starting_equity)

    for window in windows:
        for start_state in start_states:
            start_label, scenario_config = build_start_state_config(
                base_config=config,
                all_bars=all_bars,
                start_state=start_state,
                start_date=window.start_date,
                end_date=window.end_date,
                starting_equity=base_equity,
            )
            equity_df, trades, metrics = run_worksteal_backtest(
                {sym: bars.copy() for sym, bars in all_bars.items()},
                scenario_config,
                start_date=window.start_date,
                end_date=window.end_date,
                intraday_bars=intraday_bars,
            )
            if not metrics:
                raise ValueError(
                    f"No metrics returned for {window.label} start={start_label}."
                )

            total_return = float(metrics.get("total_return", 0.0))
            n_days = float(metrics.get("n_days", 0.0) or 0.0)
            equity_values = equity_df["equity"].astype(float).to_numpy(copy=False) if not equity_df.empty else np.asarray([], dtype=np.float64)
            scenario_rows.append(
                {
                    "window_label": window.label,
                    "start_state": start_label,
                    "return_pct": float(metrics.get("total_return_pct", 0.0)),
                    "annualized_return_pct": _compute_annualized_return_pct(total_return, n_days),
                    "sortino": float(metrics.get("sortino", 0.0)),
                    "max_drawdown_pct": abs(float(metrics.get("max_drawdown_pct", 0.0))),
                    "pnl_smoothness": float(compute_pnl_smoothness_from_equity(equity_values)),
                    "trade_count": float(len(trades)),
                    "n_days": n_days,
                }
            )
    return scenario_rows


def summarize_config_robustness(
    *,
    all_bars: dict[str, pd.DataFrame],
    config: WorkStealConfig,
    windows: Sequence[EvaluationWindow],
    start_states: Iterable[str],
    starting_equity: float | None = None,
    intraday_bars: Optional[dict[str, pd.DataFrame]] = None,
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    scenario_rows = evaluate_config_scenarios(
        all_bars=all_bars,
        config=config,
        windows=windows,
        start_states=start_states,
        starting_equity=starting_equity,
        intraday_bars=intraday_bars,
    )
    summary = summarize_scenario_results(scenario_rows)
    return scenario_rows, summary


__all__ = [
    "EvaluationWindow",
    "build_recent_windows",
    "build_start_state_config",
    "evaluate_config_scenarios",
    "normalize_start_state",
    "summarize_config_robustness",
]
