from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.trade_stock_forecast_snapshot import (
    load_latest_forecast_snapshot,
    reset_forecast_cache,
)


def simple_parse_list(raw: object) -> Optional[Sequence[float]]:
    if raw is None:
        return None
    try:
        parts = str(raw).split("|")
        return [float(part) for part in parts if part != ""]
    except ValueError:
        return None


def simple_coerce(value: object) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def test_load_latest_forecast_snapshot_returns_empty_when_missing(tmp_path: Path):
    reset_forecast_cache()
    snapshot = load_latest_forecast_snapshot(
        tmp_path,
        logger=None,
        parse_float_list=simple_parse_list,
        coerce_optional_float=simple_coerce,
    )
    assert snapshot == {}


def test_load_latest_forecast_snapshot_parses_values(tmp_path: Path):
    reset_forecast_cache()
    csv_path = tmp_path / "predictions-20240101.csv"
    csv_path.write_text(
        "instrument,maxdiffprofit_profit,maxdiffprofit_profit_values,entry_takeprofit_profit\n"
        "AAPL,1.5,0.1|0.2|0.3,0.5\n"
        "MSFT,,0.3|not-a-number,\n"
    )

    snapshot = load_latest_forecast_snapshot(
        tmp_path,
        logger=None,
        parse_float_list=simple_parse_list,
        coerce_optional_float=simple_coerce,
    )

    assert set(snapshot.keys()) == {"AAPL"}
    assert snapshot["AAPL"]["maxdiffprofit_profit"] == 1.5
    assert snapshot["AAPL"]["maxdiffprofit_profit_values"] == [0.1, 0.2, 0.3]
    assert snapshot["AAPL"]["entry_takeprofit_profit"] == 0.5
    # Rows without usable data should be skipped entirely
    assert "MSFT" not in snapshot
