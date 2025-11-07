"""Helper utilities for loading and caching latest forecast snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import pandas as pd

ForecastSnapshot = Dict[str, Dict[str, object]]
ParseFloatList = Callable[[object], Optional[Sequence[float]]]
CoerceOptionalFloat = Callable[[object], Optional[float]]

_LATEST_FORECAST_CACHE: ForecastSnapshot = {}
_LATEST_FORECAST_PATH: Optional[Path] = None

__all__ = [
    "ForecastSnapshot",
    "find_latest_prediction_file",
    "load_latest_forecast_snapshot",
    "reset_forecast_cache",
]


def reset_forecast_cache() -> None:
    """Clear in-memory forecast cache, primarily for tests."""
    global _LATEST_FORECAST_CACHE, _LATEST_FORECAST_PATH
    _LATEST_FORECAST_CACHE = {}
    _LATEST_FORECAST_PATH = None


def find_latest_prediction_file(results_path: Path) -> Optional[Path]:
    """Return the most recent predictions CSV within the provided directory."""
    if not results_path.exists():
        return None
    candidates = list(results_path.glob("predictions-*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_latest_forecast_snapshot(
    results_dir: Path,
    *,
    logger=None,
    parse_float_list: ParseFloatList,
    coerce_optional_float: CoerceOptionalFloat,
) -> ForecastSnapshot:
    """Load the most recent forecast snapshot, caching by path to avoid re-parsing."""
    global _LATEST_FORECAST_CACHE, _LATEST_FORECAST_PATH

    latest_file = find_latest_prediction_file(results_dir)
    if latest_file is None:
        reset_forecast_cache()
        return {}
    if _LATEST_FORECAST_PATH == latest_file and _LATEST_FORECAST_CACHE:
        return _LATEST_FORECAST_CACHE

    desired_columns = {
        "maxdiffprofit_profit",
        "maxdiffprofit_high_price",
        "maxdiffprofit_low_price",
        "maxdiffprofit_profit_high_multiplier",
        "maxdiffprofit_profit_low_multiplier",
        "maxdiffprofit_profit_values",
        "entry_takeprofit_profit",
        "entry_takeprofit_high_price",
        "entry_takeprofit_low_price",
        "entry_takeprofit_profit_values",
        "takeprofit_profit",
        "takeprofit_high_price",
        "takeprofit_low_price",
    }

    try:
        df = pd.read_csv(
            latest_file,
            usecols=lambda column: column == "instrument" or column in desired_columns,
        )
    except Exception as exc:  # pragma: no cover - exercised when CSV missing/corrupt
        if logger is not None:
            logger.warning(f"Failed to load latest prediction snapshot {latest_file}: {exc}")
        reset_forecast_cache()
        _LATEST_FORECAST_PATH = latest_file
        return _LATEST_FORECAST_CACHE

    snapshot: ForecastSnapshot = {}

    for row in df.to_dict("records"):
        instrument = row.get("instrument")
        if not instrument:
            continue
        entry: Dict[str, object] = {}
        for key in desired_columns:
            if key not in row:
                continue
            if key.endswith("_values"):
                parsed_values = parse_float_list(row.get(key))
                if parsed_values is not None:
                    entry[key] = parsed_values
            else:
                parsed_float = coerce_optional_float(row.get(key))
                if parsed_float is not None:
                    entry[key] = parsed_float
        if entry:
            snapshot[str(instrument)] = entry

    _LATEST_FORECAST_CACHE = snapshot
    _LATEST_FORECAST_PATH = latest_file
    return snapshot
