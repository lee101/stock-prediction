#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.hourly_data_utils import discover_hourly_symbols


@dataclass(frozen=True)
class CacheAuditRow:
    symbol: str
    horizon_hours: int
    history_rows: int
    cache_rows: int
    latest_history_timestamp: str | None
    latest_cache_timestamp: str | None
    latest_gap_hours: float | None
    missing_file: bool
    missing_timestamps_in_cache_range: int

    @property
    def has_issue(self) -> bool:
        return bool(
            self.missing_file
            or self.missing_timestamps_in_cache_range > 0
            or (self.latest_gap_hours is not None and self.latest_gap_hours > 0.0)
        )


def _parse_symbols(raw: str | None, *, data_root: Path) -> list[str]:
    if raw:
        return [token.strip().upper() for token in str(raw).split(",") if token.strip()]
    return discover_hourly_symbols(Path(data_root))


def _parse_horizons(raw: str) -> list[int]:
    horizons: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        horizons.append(int(token))
    if not horizons:
        raise ValueError("Expected at least one horizon.")
    return horizons


def _load_timestamps_csv(path: Path) -> pd.Series:
    frame = pd.read_csv(path, usecols=["timestamp"])
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    return ts.reset_index(drop=True)


def _load_timestamps_parquet(path: Path) -> pd.Series:
    frame = pd.read_parquet(path, columns=["timestamp"])
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    return ts.reset_index(drop=True)


def audit_cache_pair(*, symbol: str, horizon_hours: int, data_root: Path, forecast_cache_root: Path) -> CacheAuditRow:
    history_path = Path(data_root) / f"{symbol}.csv"
    cache_path = Path(forecast_cache_root) / f"h{int(horizon_hours)}" / f"{symbol}.parquet"

    history_ts = _load_timestamps_csv(history_path)
    if not cache_path.exists():
        latest_history = history_ts.iloc[-1] if not history_ts.empty else None
        return CacheAuditRow(
            symbol=symbol,
            horizon_hours=int(horizon_hours),
            history_rows=int(len(history_ts)),
            cache_rows=0,
            latest_history_timestamp=latest_history.isoformat() if latest_history is not None else None,
            latest_cache_timestamp=None,
            latest_gap_hours=None,
            missing_file=True,
            missing_timestamps_in_cache_range=0,
        )

    cache_ts = _load_timestamps_parquet(cache_path)
    latest_history = history_ts.iloc[-1] if not history_ts.empty else None
    latest_cache = cache_ts.iloc[-1] if not cache_ts.empty else None
    latest_gap_hours: float | None = None
    if latest_history is not None and latest_cache is not None:
        latest_gap_hours = max((latest_history - latest_cache).total_seconds() / 3600.0, 0.0)

    missing_count = 0
    if not history_ts.empty and not cache_ts.empty:
        eligible_history = history_ts[history_ts >= cache_ts.iloc[0]]
        missing_count = int((~eligible_history.isin(set(cache_ts.tolist()))).sum())

    return CacheAuditRow(
        symbol=symbol,
        horizon_hours=int(horizon_hours),
        history_rows=int(len(history_ts)),
        cache_rows=int(len(cache_ts)),
        latest_history_timestamp=latest_history.isoformat() if latest_history is not None else None,
        latest_cache_timestamp=latest_cache.isoformat() if latest_cache is not None else None,
        latest_gap_hours=latest_gap_hours,
        missing_file=False,
        missing_timestamps_in_cache_range=missing_count,
    )


def run_audit(*, symbols: Sequence[str], horizons: Sequence[int], data_root: Path, forecast_cache_root: Path) -> list[CacheAuditRow]:
    rows: list[CacheAuditRow] = []
    for symbol in symbols:
        for horizon in horizons:
            rows.append(
                audit_cache_pair(
                    symbol=str(symbol).upper(),
                    horizon_hours=int(horizon),
                    data_root=Path(data_root),
                    forecast_cache_root=Path(forecast_cache_root),
                )
            )
    return rows


def _summary_payload(rows: Iterable[CacheAuditRow]) -> dict[str, object]:
    rows = list(rows)
    issues = [row for row in rows if row.has_issue]
    return {
        "rows": [asdict(row) | {"has_issue": row.has_issue} for row in rows],
        "summary": {
            "total_pairs": len(rows),
            "issue_pairs": len(issues),
            "missing_files": sum(1 for row in rows if row.missing_file),
            "stale_pairs": sum(1 for row in rows if row.latest_gap_hours is not None and row.latest_gap_hours > 0.0),
            "pairs_with_missing_timestamps": sum(1 for row in rows if row.missing_timestamps_in_cache_range > 0),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit hourly forecast-cache coverage against source hourly bars.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols. Default: discover from data root.")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--forecast-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--horizons", default="1,24")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args(argv)

    symbols = _parse_symbols(args.symbols, data_root=args.data_root)
    horizons = _parse_horizons(args.horizons)
    rows = run_audit(
        symbols=symbols,
        horizons=horizons,
        data_root=Path(args.data_root),
        forecast_cache_root=Path(args.forecast_cache_root),
    )
    payload = _summary_payload(rows)

    for row in rows:
        status = "ISSUE" if row.has_issue else "OK"
        print(
            f"{status} symbol={row.symbol} h={row.horizon_hours} "
            f"cache_rows={row.cache_rows} history_rows={row.history_rows} "
            f"latest_gap_hours={row.latest_gap_hours} "
            f"missing_in_range={row.missing_timestamps_in_cache_range} "
            f"missing_file={row.missing_file}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    if args.fail_on_issues and payload["summary"]["issue_pairs"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
