"""Utilities for per-symbol meta strategy selection on hourly stock actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


SUPPORTED_META_METRICS = (
    "return",
    "sortino",
    "sharpe",
    "calmar",
    "omega",
    "gain_pain",
    "p10",
    "median",
)

_RATIO_DENOM_FLOOR = 1e-6
_RATIO_SCORE_CLIP = 50.0


def daily_returns_from_equity(equity_curve: pd.Series) -> pd.Series:
    """Convert an intraday equity curve into close-to-close daily returns."""
    if equity_curve.empty:
        return pd.Series(dtype=float)
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        raise ValueError("equity_curve index must be a DatetimeIndex")
    eq = equity_curve.sort_index().astype(float)
    daily_eq = eq.resample("1D").last().dropna()
    if daily_eq.empty:
        return pd.Series(dtype=float)
    return daily_eq.pct_change().fillna(0.0)


def score_trailing_returns(
    returns: Sequence[float],
    metric: str,
    *,
    sample_weights: Sequence[float] | None = None,
) -> float:
    """Score trailing returns with supported meta metrics."""
    name = metric.strip().lower()
    if name not in SUPPORTED_META_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Expected one of {SUPPORTED_META_METRICS}.")

    arr = np.asarray(list(returns), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("-inf")

    weights = _normalize_sample_weights(arr, sample_weights)
    if weights is None:
        cum_ret = float(np.prod(1.0 + arr) - 1.0)
    else:
        safe = np.clip(1.0 + arr, 1e-12, None)
        cum_ret = float(np.exp(np.sum(weights * np.log(safe))) - 1.0)
    score = cum_ret
    if name != "return":
        mean_ret = _weighted_mean(arr, weights)
        if name == "sharpe":
            std_ret = _weighted_std(arr, weights)
            score = _safe_ratio(mean_ret, std_ret)
        elif name == "sortino":
            downside = arr[arr < 0.0]
            downside_weights = _subset_weights(arr, weights, arr < 0.0)
            downside_std = _weighted_std(downside, downside_weights) if downside.size else 0.0
            score = _safe_ratio(mean_ret, downside_std)
        elif name == "calmar":
            drawdown = _max_drawdown_from_returns(arr)
            score = _safe_ratio(cum_ret, drawdown)
        elif name == "omega":
            gains = _weighted_sum(np.clip(arr, 0.0, None), weights)
            losses = _weighted_sum(np.clip(-arr, 0.0, None), weights)
            score = _safe_ratio(gains, losses)
        elif name == "gain_pain":
            losses = _weighted_sum(np.clip(-arr, 0.0, None), weights)
            score = _safe_ratio(cum_ret, losses)
        elif name == "p10":
            score = _weighted_percentile(arr, weights, 10.0)
        elif name == "median":
            score = _weighted_percentile(arr, weights, 50.0)
    if name in {"return", "p10", "median"}:
        return float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
    if not np.isfinite(score):
        return float(np.sign(score) * _RATIO_SCORE_CLIP) if score != 0 else 0.0
    return float(np.clip(score, -_RATIO_SCORE_CLIP, _RATIO_SCORE_CLIP))


def select_daily_winners(
    daily_returns_by_strategy: Mapping[str, pd.Series],
    *,
    lookback_days: int,
    metric: str,
    fallback_strategy: str | None = None,
    tie_break_order: Sequence[str] | None = None,
    require_full_window: bool = True,
    sit_out_threshold: float | None = None,
    selection_mode: str = "winner",
    switch_margin: float = 0.0,
    min_score_gap: float = 0.0,
    recency_halflife_days: float | None = None,
) -> pd.Series:
    """Select one strategy per day based on trailing daily performance.

    The winner for day ``D`` is computed using returns strictly before ``D``.
    """
    if lookback_days <= 0:
        raise ValueError(f"lookback_days must be > 0, got {lookback_days}.")
    if not daily_returns_by_strategy:
        raise ValueError("daily_returns_by_strategy cannot be empty.")

    strategy_names = list(daily_returns_by_strategy.keys())
    ordered_names = _ordered_strategy_names(strategy_names, tie_break_order)

    if fallback_strategy is None:
        fallback_strategy = ordered_names[0]
    if fallback_strategy not in daily_returns_by_strategy:
        raise ValueError(f"fallback_strategy '{fallback_strategy}' is not in strategy set {strategy_names}.")
    mode = str(selection_mode).strip().lower()
    if mode not in ("winner", "winner_cash", "sticky"):
        raise ValueError(
            f"Unsupported selection_mode '{selection_mode}'. Expected one of ('winner', 'winner_cash', 'sticky')."
        )
    if switch_margin < 0:
        raise ValueError(f"switch_margin must be >= 0, got {switch_margin}.")
    if min_score_gap < 0:
        raise ValueError(f"min_score_gap must be >= 0, got {min_score_gap}.")
    if recency_halflife_days is not None and float(recency_halflife_days) <= 0:
        raise ValueError(f"recency_halflife_days must be > 0 when provided, got {recency_halflife_days}.")

    all_days = sorted(
        {
            pd.Timestamp(day).floor("D")
            for series in daily_returns_by_strategy.values()
            for day in series.index
        }
    )
    if not all_days:
        return pd.Series(dtype=object)

    all_days_idx = pd.DatetimeIndex(all_days)
    aligned_returns = {
        name: pd.to_numeric(series.reindex(all_days_idx).fillna(0.0), errors="coerce").fillna(0.0)
        for name, series in daily_returns_by_strategy.items()
    }

    winners: list[str | None] = []
    prev_winner: str | None = fallback_strategy
    for day_idx in range(len(all_days_idx)):
        window_start = max(0, day_idx - lookback_days)
        window_len = day_idx - window_start
        if window_len <= 0:
            winners.append(fallback_strategy)
            prev_winner = fallback_strategy
            continue
        if require_full_window and window_len < lookback_days:
            winners.append(fallback_strategy)
            prev_winner = fallback_strategy
            continue

        scores: dict[str, float] = {}
        for name in ordered_names:
            window = aligned_returns[name].iloc[window_start:day_idx].to_numpy(dtype=np.float64)
            window_weights = _recency_weights(len(window), recency_halflife_days)
            scores[name] = score_trailing_returns(window, metric, sample_weights=window_weights)
        ranked = sorted(ordered_names, key=lambda n: scores[n], reverse=True)
        best_name = ranked[0]
        best_score = scores[best_name]
        second_score = scores[ranked[1]] if len(ranked) > 1 else float("-inf")
        candidate: str | None = best_name

        if mode == "sticky" and prev_winner in scores:
            prev_score = scores[prev_winner]
            if prev_score + float(switch_margin) >= best_score:
                candidate = prev_winner

        if min_score_gap > 0 and np.isfinite(second_score) and (best_score - second_score) < float(min_score_gap):
            if mode == "winner_cash":
                candidate = None
            elif mode == "sticky" and prev_winner in scores:
                candidate = prev_winner

        selected_score = scores.get(candidate, best_score) if candidate is not None else best_score
        if sit_out_threshold is not None and selected_score < float(sit_out_threshold):
            winners.append(None)
        else:
            winners.append(candidate)
            if candidate is not None:
                prev_winner = candidate

    return pd.Series(winners, index=all_days_idx, dtype=object)


def _recency_weights(window_len: int, recency_halflife_days: float | None) -> np.ndarray | None:
    if recency_halflife_days is None:
        return None
    if window_len <= 0:
        return None
    half_life = float(recency_halflife_days)
    # Newer samples receive larger weights; oldest sample has highest age.
    ages = np.arange(window_len - 1, -1, -1, dtype=np.float64)
    raw = np.power(0.5, ages / half_life).astype(np.float64, copy=False)
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return raw / total


def _normalize_sample_weights(values: np.ndarray, sample_weights: Sequence[float] | None) -> np.ndarray | None:
    if sample_weights is None:
        return None
    weights = np.asarray(list(sample_weights), dtype=np.float64)
    if weights.shape != values.shape:
        raise ValueError(
            "sample_weights must match returns shape, "
            f"got weights={weights.shape}, returns={values.shape}."
        )
    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        weights = weights[finite_mask]
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return weights / total


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if values.size == 0:
        return 0.0
    if weights is None:
        return float(np.mean(values))
    return float(np.sum(values * weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray | None) -> float:
    if values.size == 0:
        return 0.0
    if weights is None:
        return float(np.std(values, ddof=0))
    mean = _weighted_mean(values, weights)
    var = float(np.sum(weights * np.square(values - mean)))
    var = max(var, 0.0)
    return float(np.sqrt(var))


def _weighted_sum(values: np.ndarray, weights: np.ndarray | None) -> float:
    if values.size == 0:
        return 0.0
    if weights is None:
        return float(np.sum(values))
    return float(np.sum(values * weights))


def _weighted_percentile(values: np.ndarray, weights: np.ndarray | None, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    q = float(np.clip(percentile, 0.0, 100.0)) / 100.0
    if weights is None:
        return float(np.percentile(values, percentile))
    order = np.argsort(values, kind="mergesort")
    vals = values[order]
    w = np.clip(weights[order], 0.0, None)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return float(np.percentile(values, percentile))
    w = w / total
    cdf = np.cumsum(w)
    return float(np.interp(q, cdf, vals, left=float(vals[0]), right=float(vals[-1])))


def _subset_weights(
    full_values: np.ndarray,
    full_weights: np.ndarray | None,
    subset_mask: np.ndarray,
) -> np.ndarray | None:
    if full_weights is None:
        return None
    sub = full_weights[subset_mask]
    total = float(np.sum(sub))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return sub / total


def combine_actions_by_winners(
    actions_by_strategy: Mapping[str, pd.DataFrame],
    winners_by_symbol: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Build a single action dataframe by selecting strategy rows per symbol/day."""
    if not actions_by_strategy:
        raise ValueError("actions_by_strategy cannot be empty.")
    if not winners_by_symbol:
        raise ValueError("winners_by_symbol cannot be empty.")

    strategy_names = set(actions_by_strategy.keys())
    first_df = next(iter(actions_by_strategy.values()))
    required_cols = {"timestamp", "symbol"}
    if not required_cols.issubset(first_df.columns):
        raise ValueError(f"actions dataframe must include {sorted(required_cols)}.")

    winner_rows = []
    for symbol, winners in winners_by_symbol.items():
        symbol_u = str(symbol).upper()
        for day, winner in winners.items():
            if winner is None or (isinstance(winner, float) and np.isnan(winner)):
                winner_rows.append((symbol_u, pd.Timestamp(day).floor("D"), None))
                continue
            winner_name = str(winner)
            if winner_name not in strategy_names:
                raise ValueError(f"Winner '{winner_name}' for symbol {symbol_u} is not a provided strategy.")
            winner_rows.append((symbol_u, pd.Timestamp(day).floor("D"), winner_name))

    if not winner_rows:
        raise ValueError("No winner rows were provided.")

    winners_df = pd.DataFrame(winner_rows, columns=["symbol", "day", "winner"])
    winners_df = winners_df.drop_duplicates(subset=["symbol", "day"], keep="last")

    selected_parts = []
    key_template = first_df[["timestamp", "symbol"]].copy()
    key_template["timestamp"] = pd.to_datetime(key_template["timestamp"], utc=True)
    key_template["symbol"] = key_template["symbol"].astype(str).str.upper()
    key_template["day"] = key_template["timestamp"].dt.floor("D")

    for name, actions in actions_by_strategy.items():
        if not required_cols.issubset(actions.columns):
            raise ValueError(f"Strategy '{name}' actions dataframe is missing required columns {sorted(required_cols)}.")

        df = actions.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["day"] = df["timestamp"].dt.floor("D")

        winners_for_name = winners_df[winners_df["winner"] == name][["symbol", "day"]]
        if winners_for_name.empty:
            continue
        picked = df.merge(winners_for_name, on=["symbol", "day"], how="inner")
        if not picked.empty:
            selected_parts.append(picked[actions.columns])

    cash_days = winners_df[winners_df["winner"].isna()][["symbol", "day"]]
    if not cash_days.empty:
        template = first_df.copy()
        template["timestamp"] = pd.to_datetime(template["timestamp"], utc=True)
        template["symbol"] = template["symbol"].astype(str).str.upper()
        template["day"] = template["timestamp"].dt.floor("D")
        cash_rows = template.merge(cash_days, on=["symbol", "day"], how="inner")
        if not cash_rows.empty:
            for col in ("buy_amount", "sell_amount", "trade_amount"):
                if col in cash_rows.columns:
                    cash_rows[col] = 0.0
            selected_parts.append(cash_rows[first_df.columns])

    if not selected_parts:
        raise ValueError("No selected action rows after applying winners.")

    combined = pd.concat(selected_parts, ignore_index=True)
    combined = combined.sort_values(["timestamp", "symbol"]).drop_duplicates(["timestamp", "symbol"], keep="first")
    combined = combined.reset_index(drop=True)

    combined_keys = combined[["timestamp", "symbol"]].copy()
    combined_keys["timestamp"] = pd.to_datetime(combined_keys["timestamp"], utc=True)
    combined_keys["symbol"] = combined_keys["symbol"].astype(str).str.upper()
    expected = key_template[["timestamp", "symbol"]].drop_duplicates()
    merged_keys = expected.merge(combined_keys.drop_duplicates(), on=["timestamp", "symbol"], how="left", indicator=True)
    if (merged_keys["_merge"] == "left_only").any():
        missing = int((merged_keys["_merge"] == "left_only").sum())
        raise ValueError(f"Meta-selected actions are incomplete: missing {missing} timestamp/symbol rows.")

    return combined


def _ordered_strategy_names(
    strategy_names: Sequence[str],
    tie_break_order: Sequence[str] | None,
) -> list[str]:
    names = list(strategy_names)
    if tie_break_order:
        order = [str(name) for name in tie_break_order if str(name) in names]
        order_set = set(order)
        order.extend(sorted([name for name in names if name not in order_set]))
        return order
    return sorted(names)


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    growth = np.maximum(1.0 + returns, 1e-9)
    equity = np.cumprod(growth)
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / np.maximum(running_max, 1e-9)
    return float(np.max(drawdowns))


def _safe_ratio(numerator: float, denominator: float) -> float:
    num = float(numerator)
    den = abs(float(denominator))
    if not np.isfinite(num):
        return float(np.sign(num) * _RATIO_SCORE_CLIP) if num != 0 else 0.0
    if not np.isfinite(den):
        den = _RATIO_DENOM_FLOOR
    den = max(den, _RATIO_DENOM_FLOOR)
    raw = num / den
    if not np.isfinite(raw):
        return float(np.sign(raw) * _RATIO_SCORE_CLIP) if raw != 0 else 0.0
    compressed = float(np.sign(raw) * np.log1p(abs(raw)))
    return float(np.clip(compressed, -_RATIO_SCORE_CLIP, _RATIO_SCORE_CLIP))
