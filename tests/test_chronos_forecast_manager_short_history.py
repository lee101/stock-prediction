from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from binanceneural.config import ForecastConfig
from binanceneural.forecasts import ChronosForecastManager


def _write_hourly_csv(path: Path, *, symbol: str, rows: int) -> None:
    start = pd.Timestamp("2026-01-01 00:00:00+00:00")
    timestamps = [start + pd.Timedelta(hours=i) for i in range(rows)]
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(i) for i in range(rows)],
            "high": [float(i) for i in range(rows)],
            "low": [float(i) for i in range(rows)],
            "close": [float(i) for i in range(rows)],
            "volume": [1.0 for _ in range(rows)],
            "trade_count": [1 for _ in range(rows)],
            "vwap": [float(i) for i in range(rows)],
            "symbol": [symbol for _ in range(rows)],
        }
    )
    path.write_text(frame.to_csv(index=False))


def test_ensure_latest_writes_cache_for_short_history(tmp_path: Path) -> None:
    # History is shorter than context_hours, but longer than the minimum context window.
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    data_root.mkdir()
    cache_root.mkdir()

    symbol = "BTCU"
    _write_hourly_csv(data_root / f"{symbol}.csv", symbol=symbol, rows=120)

    cfg = ForecastConfig(
        symbol=symbol,
        data_root=data_root,
        context_hours=512,  # intentionally larger than the dataset
        prediction_horizon_hours=1,
        batch_size=16,
        cache_dir=cache_root,
    )

    mgr = ChronosForecastManager(cfg, wrapper_factory=lambda: None)
    out = mgr.ensure_latest()

    assert (cache_root / f"{symbol}.parquet").exists()
    assert not out.empty


def test_ensure_latest_raises_when_history_too_short(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    data_root.mkdir()
    cache_root.mkdir()

    symbol = "BTCU"
    _write_hourly_csv(data_root / f"{symbol}.csv", symbol=symbol, rows=10)

    cfg = ForecastConfig(
        symbol=symbol,
        data_root=data_root,
        context_hours=512,
        prediction_horizon_hours=1,
        batch_size=16,
        cache_dir=cache_root,
    )

    mgr = ChronosForecastManager(cfg, wrapper_factory=lambda: None)
    with pytest.raises(RuntimeError, match="Insufficient hourly history"):
        mgr.ensure_latest()

