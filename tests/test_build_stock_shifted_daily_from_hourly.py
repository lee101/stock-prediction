from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_stock_shifted_daily_from_hourly import build_shifted_daily_dataset


def _write_hourly_symbol(path: Path, symbol: str, *, days: int = 4, bars_per_day: int = 4) -> None:
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2026-01-05T14:30:00Z")
    price = 100.0
    for day_index in range(days):
        day_start = base + pd.Timedelta(days=day_index)
        for bar_index in range(bars_per_day):
            ts = day_start + pd.Timedelta(hours=bar_index)
            open_price = price
            close_price = open_price + 1.0 + 0.1 * day_index + 0.01 * bar_index
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "open": open_price,
                    "high": close_price + 0.5,
                    "low": open_price - 0.5,
                    "close": close_price,
                    "volume": 1_000.0 + 10.0 * bar_index,
                    "symbol": symbol,
                }
            )
            price = close_price
    pd.DataFrame(rows).to_csv(path / f"{symbol}.csv", index=False)


def test_build_shifted_daily_dataset_combined_mode(tmp_path: Path) -> None:
    hourly_root = tmp_path / "hourly"
    output_root = tmp_path / "daily_shifted"
    hourly_root.mkdir()
    _write_hourly_symbol(hourly_root, "AAPL")

    manifest = build_shifted_daily_dataset(
        hourly_root=hourly_root,
        output_root=output_root,
        symbols=("AAPL",),
        offsets=(0, 1),
        mode="combined",
        bars_per_session=4,
        separator_days=5,
        force=True,
    )

    combined_path = output_root / "AAPL.csv"
    manifest_path = output_root / "shift_manifest.json"
    assert combined_path.exists()
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["symbols"]["AAPL"]["bars_per_session"] == 4
    assert payload["symbols"]["AAPL"]["combined_rows"] == 7
    assert payload["symbols"]["AAPL"]["offset_rows"] == {"0": 4, "1": 3}
    assert manifest["symbols"]["AAPL"]["combined_rows"] == 7

    combined = pd.read_csv(combined_path)
    timestamps = pd.to_datetime(combined["timestamp"], utc=True)
    assert len(combined) == 7
    assert combined["symbol"].tolist() == ["AAPL"] * 7
    gap_days = (timestamps.iloc[4] - timestamps.iloc[3]).days
    assert gap_days >= 6
