#!/usr/bin/env python3
"""Prepare a date-sliced local CSV dataset for HF vs Puffer benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_symbols(raw_symbols: str) -> list[str]:
    symbols = [part.strip().upper() for part in raw_symbols.split(",") if part.strip()]
    if not symbols:
        raise ValueError("Expected at least one symbol")
    return symbols


def _resolve_csv_path(source_root: Path, symbol: str) -> Path:
    exact = source_root / f"{symbol}.csv"
    if exact.exists():
        return exact

    candidates = sorted(source_root.glob(f"**/{symbol}.csv"))
    candidates.extend(sorted(source_root.glob(f"**/{symbol}_*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No CSV found for symbol '{symbol}' under '{source_root}'")
    return candidates[0]


def _find_time_column(columns: list[str]) -> str:
    lower_map = {column.lower(): column for column in columns}
    for candidate in ("timestamp", "date"):
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError("CSV is missing a 'timestamp' or 'date' column")


def _to_utc_timestamp(raw_value: str | None) -> pd.Timestamp | None:
    if raw_value is None:
        return None
    timestamp = pd.to_datetime(raw_value, utc=True, errors="raise")
    if not isinstance(timestamp, pd.Timestamp):
        raise TypeError(f"Expected Timestamp, received {type(timestamp)!r}")
    return timestamp


def _slice_frame(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DataFrame, str, pd.Series]:
    time_column = _find_time_column(list(frame.columns))
    timestamps = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    if timestamps.isna().all():
        raise ValueError(f"Unable to parse any timestamps from column '{time_column}'")

    mask = timestamps.notna()
    start_ts = _to_utc_timestamp(start_date)
    end_ts = _to_utc_timestamp(end_date)
    if start_ts is not None:
        mask &= timestamps >= start_ts
    if end_ts is not None:
        mask &= timestamps <= end_ts

    sliced = frame.loc[mask].copy()
    sliced_timestamps = timestamps.loc[mask].copy()
    if sliced.empty:
        raise ValueError("Date filter removed all rows")
    return sliced, time_column, sliced_timestamps


def prepare_benchmark_dataset(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    symbols: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    source_path = Path(source_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    for symbol in symbols:
        input_csv = _resolve_csv_path(source_path, symbol)
        frame = pd.read_csv(input_csv)
        sliced, time_column, sliced_timestamps = _slice_frame(
            frame,
            start_date=start_date,
            end_date=end_date,
        )

        output_csv = output_path / f"{symbol}.csv"
        sliced.to_csv(output_csv, index=False)
        files.append(
            {
                "symbol": symbol,
                "source_csv": str(input_csv),
                "output_csv": str(output_csv),
                "rows": int(len(sliced)),
                "time_column": time_column,
                "min_timestamp": str(sliced_timestamps.min()),
                "max_timestamp": str(sliced_timestamps.max()),
            }
        )

    manifest = {
        "source_root": str(source_path),
        "output_dir": str(output_path),
        "symbols": list(symbols),
        "start_date": start_date,
        "end_date": end_date,
        "files": files,
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a local date-sliced benchmark dataset")
    parser.add_argument("--source-root", default="trainingdata")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    symbols = _parse_symbols(args.symbols)
    manifest = prepare_benchmark_dataset(
        source_root=args.source_root,
        output_dir=args.output_dir,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
