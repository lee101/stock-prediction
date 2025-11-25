"""Utilities for loading and parsing forecast data."""

from pathlib import Path
from typing import Dict

from loguru import logger

from src.trade_stock_forecast_snapshot import load_latest_forecast_snapshot as _load_latest_forecast_snapshot
from src.trade_stock_utils import coerce_optional_float, parse_float_list


def get_results_dir() -> Path:
    """Get results directory for forecast snapshots."""
    from stock.state import get_state_dir

    return get_state_dir() / "results"


def load_latest_forecast_snapshot() -> Dict[str, Dict[str, object]]:
    """Load the most recent forecast snapshot."""
    return _load_latest_forecast_snapshot(
        get_results_dir(),
        logger=logger,
        parse_float_list=parse_float_list,
        coerce_optional_float=coerce_optional_float,
    )


def extract_forecasted_pnl(forecast: Dict[str, object], default: float = 0.0) -> float:
    """Extract forecasted PnL from forecast dict, checking multiple fields."""
    for field in [
        "maxdiff_forecasted_pnl",
        "maxdiffalwayson_forecasted_pnl",
        "highlow_forecasted_pnl",
        "avg_return",
    ]:
        value = forecast.get(field)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default
