from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Tuple


def _coerce_float(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if result != result:  # NaN
        return 0.0
    return result


def _strategy_key(strategy: object) -> str:
    return str(strategy or "").strip().lower()


def get_selected_strategy_forecast(entry: Mapping[str, object]) -> float:
    """
    Return the forecasted PnL for the entry's currently selected strategy.

    Priority:
        1. strategy_candidate_forecasted_pnl[strategy]
        2. ``{strategy}_forecasted_pnl`` if present at top-level
        3. fallback to ``avg_return`` (to preserve legacy behaviour when forecasts unavailable)
    """
    strategy = _strategy_key(entry.get("strategy"))
    candidate_map = entry.get("strategy_candidate_forecasted_pnl")
    if isinstance(candidate_map, Mapping):
        forecast = candidate_map.get(strategy)
        if forecast is not None:
            return _coerce_float(forecast)
    direct_key = f"{strategy}_forecasted_pnl"
    if direct_key in entry:
        return _coerce_float(entry.get(direct_key))
    return _coerce_float(entry.get("avg_return", 0.0))


@dataclass(frozen=True)
class DropRecord:
    forecast: float
    avg_return: float


def filter_positive_forecasts(
    picks: MutableMapping[str, Dict[str, object]],
    *,
    require_positive_forecast: bool = True,
    require_positive_avg_return: bool = True,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, DropRecord]]:
    """
    Filter picks based on forecasted PnL and historical avg_return guards.

    Returns ``(filtered, dropped)`` where ``dropped`` captures the rejected entries.
    """
    filtered: Dict[str, Dict[str, object]] = {}
    dropped: Dict[str, DropRecord] = {}
    for symbol, data in picks.items():
        forecast = get_selected_strategy_forecast(data)
        avg_return = _coerce_float(data.get("avg_return", 0.0))
        if require_positive_forecast and forecast <= 0.0:
            dropped[symbol] = DropRecord(forecast=forecast, avg_return=avg_return)
            continue
        if require_positive_avg_return and avg_return <= 0.0:
            dropped[symbol] = DropRecord(forecast=forecast, avg_return=avg_return)
            continue
        filtered[symbol] = data
    return filtered, dropped


__all__ = ["filter_positive_forecasts", "DropRecord", "get_selected_strategy_forecast"]
