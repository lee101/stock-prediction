from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

FetchFn = Callable[[str, datetime, datetime], pd.DataFrame | None]


def append_hourly_binance_bars(
    csv_path: Path,
    *,
    fetch_symbol: str,
    csv_symbol: str,
    fetcher: FetchFn,
    overlap_hours: int = 2,
    end_time: datetime | None = None,
) -> dict[str, object]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {"status": "missing_csv", "path": str(csv_path)}

    existing = pd.read_csv(csv_path)
    if "timestamp" not in existing.columns:
        return {"status": "invalid_csv", "path": str(csv_path)}

    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce")
    existing = existing.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if existing.empty:
        return {"status": "empty_csv", "path": str(csv_path)}

    last_ts = existing["timestamp"].max()
    start_ts = last_ts - timedelta(hours=max(1, int(overlap_hours)))
    end_ts = end_time or datetime.now(timezone.utc)
    fetched = fetcher(fetch_symbol, start=start_ts, end=end_ts)
    if fetched is None or len(fetched) == 0:
        return {
            "status": "no_data",
            "path": str(csv_path),
            "start": str(start_ts),
            "end": str(end_ts),
        }

    new = fetched.copy()
    if isinstance(new.index, pd.MultiIndex) and "symbol" in new.index.names:
        new = new.reset_index(level="symbol", drop=True)
    if isinstance(new.index, pd.DatetimeIndex):
        new = new.reset_index()
    if "timestamp" not in new.columns:
        return {"status": "invalid_fetch", "path": str(csv_path)}

    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True, errors="coerce")
    new = new.dropna(subset=["timestamp"]).copy()
    if new.empty:
        return {
            "status": "no_data",
            "path": str(csv_path),
            "start": str(start_ts),
            "end": str(end_ts),
        }
    new["symbol"] = csv_symbol.upper()

    combined = pd.concat([existing, new], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    added_rows = max(0, len(combined) - len(existing))
    combined.to_csv(csv_path, index=False)
    return {
        "status": "updated" if added_rows > 0 else "synced",
        "path": str(csv_path),
        "rows_added": int(added_rows),
        "start": str(combined["timestamp"].min()),
        "end": str(combined["timestamp"].max()),
    }


__all__ = ["append_hourly_binance_bars"]
