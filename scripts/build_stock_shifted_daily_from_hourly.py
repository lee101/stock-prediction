#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.bar_aggregation import hourly_to_shifted_session_daily_ohlcv


def _discover_symbols(hourly_root: Path) -> list[str]:
    return sorted(path.stem.upper() for path in hourly_root.glob("*.csv"))


def _parse_offsets(raw: str) -> list[int]:
    offsets: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        offsets.append(int(token))
    if not offsets:
        raise ValueError("At least one offset is required")
    return sorted(set(offsets))


def _read_hourly_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(col).lower() for col in frame.columns]
    return frame


def _filter_hourly_date_range(
    hourly: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if hourly.empty:
        return hourly.copy()
    if "timestamp" not in hourly.columns:
        raise KeyError("Hourly CSV must contain a timestamp column")
    ts = pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce")
    mask = pd.Series(True, index=hourly.index, dtype=bool)
    if start_date:
        start_ts = pd.Timestamp(start_date)
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        mask &= ts >= start_ts.floor("D")
    if end_date:
        end_ts = pd.Timestamp(end_date)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        mask &= ts <= end_ts.floor("D") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered = hourly.loc[mask].copy()
    return filtered.reset_index(drop=True)


def infer_session_bars(hourly: pd.DataFrame, *, min_count: int = 2) -> int:
    if "timestamp" not in hourly.columns:
        raise KeyError("Hourly CSV must contain a timestamp column")
    ts = pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        raise ValueError("No valid timestamps found")
    counts = ts.dt.floor("D").value_counts()
    counts = counts[counts >= int(min_count)]
    if counts.empty:
        raise ValueError("Unable to infer session bars from hourly data")
    mode = counts.mode()
    return int(mode.iloc[-1] if not mode.empty else counts.median())


def _write_daily_csv(frame: pd.DataFrame, output_path: Path, *, force: bool) -> None:
    if output_path.exists() and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def _rebase_block(frame: pd.DataFrame, *, block_start: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    deltas = ts - ts.min()
    out["timestamp"] = (block_start + deltas).astype("datetime64[ns, UTC]")
    return out


def build_shifted_daily_dataset(
    *,
    hourly_root: Path,
    output_root: Path,
    symbols: Sequence[str],
    offsets: Sequence[int],
    mode: str,
    bars_per_session: int | None,
    separator_days: int,
    start_date: str | None = None,
    end_date: str | None = None,
    force: bool,
) -> dict[str, object]:
    symbol_list = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not symbol_list:
        symbol_list = _discover_symbols(hourly_root)
    if not symbol_list:
        raise ValueError(f"No hourly CSVs found under {hourly_root}")

    manifest: dict[str, object] = {
        "hourly_root": str(hourly_root),
        "output_root": str(output_root),
        "mode": mode,
        "requested_offsets": [int(v) for v in offsets],
        "separator_days": int(separator_days),
        "start_date": start_date,
        "end_date": end_date,
        "symbols": {},
    }

    output_root.mkdir(parents=True, exist_ok=True)

    for symbol in symbol_list:
        hourly_path = hourly_root / f"{symbol}.csv"
        if not hourly_path.exists():
            raise FileNotFoundError(f"Missing hourly CSV for {symbol}: {hourly_path}")
        hourly = _read_hourly_csv(hourly_path)
        hourly = _filter_hourly_date_range(hourly, start_date=start_date, end_date=end_date)
        if hourly.empty:
            raise ValueError(f"{symbol}: no hourly rows remain after date filtering")
        inferred_bars = int(bars_per_session or infer_session_bars(hourly))
        active_offsets = [offset for offset in offsets if int(offset) < inferred_bars]
        if not active_offsets:
            raise ValueError(f"{symbol}: no offsets < inferred bars/session ({inferred_bars})")

        per_offset_rows: dict[str, int] = {}
        if mode == "per_offset":
            for offset in active_offsets:
                daily, stats = hourly_to_shifted_session_daily_ohlcv(
                    hourly,
                    offset_bars=int(offset),
                    output_symbol=symbol,
                    require_full_shift=True,
                )
                if daily.empty:
                    continue
                per_offset_rows[str(int(offset))] = int(len(daily))
                variant_root = output_root / f"offset_{int(offset)}"
                _write_daily_csv(daily, variant_root / f"{symbol}.csv", force=force)
                manifest["symbols"][symbol] = {
                    "bars_per_session": inferred_bars,
                    "offset_rows": per_offset_rows,
                }
            continue

        block_start = pd.Timestamp("2000-01-03T00:00:00Z")
        combined_parts: list[pd.DataFrame] = []
        offset_payload: dict[str, object] = {}
        for offset in active_offsets:
            daily, stats = hourly_to_shifted_session_daily_ohlcv(
                hourly,
                offset_bars=int(offset),
                output_symbol=symbol,
                require_full_shift=True,
            )
            if daily.empty:
                offset_payload[str(int(offset))] = {
                    "rows": 0,
                    "dropped_incomplete_days": int(stats.dropped_incomplete_days),
                }
                continue
            rebased = _rebase_block(daily, block_start=block_start)
            combined_parts.append(rebased)
            block_span = pd.to_datetime(rebased["timestamp"], utc=True, errors="coerce").max() - pd.to_datetime(
                rebased["timestamp"], utc=True, errors="coerce"
            ).min()
            block_start = block_start + block_span + pd.Timedelta(days=int(separator_days) + 1)
            offset_payload[str(int(offset))] = {
                "rows": int(len(daily)),
                "dropped_incomplete_days": int(stats.dropped_incomplete_days),
            }
            per_offset_rows[str(int(offset))] = int(len(daily))

        combined = (
            pd.concat(combined_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            if combined_parts
            else pd.DataFrame()
        )
        if not combined.empty:
            _write_daily_csv(combined, output_root / f"{symbol}.csv", force=force)
        manifest["symbols"][symbol] = {
            "bars_per_session": inferred_bars,
            "offset_rows": per_offset_rows,
            "offset_stats": offset_payload,
            "combined_rows": int(len(combined)),
        }

    manifest_path = output_root / "shift_manifest.json"
    manifest_path.write_text(f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build shifted pseudo-daily stock CSVs from hourly stock bars.",
    )
    parser.add_argument(
        "--hourly-root",
        type=Path,
        default=Path("trainingdatahourly/stocks"),
        help="Directory containing hourly stock CSVs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to write shifted daily CSVs.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbol list. Default: all CSVs under --hourly-root.",
    )
    parser.add_argument(
        "--offsets",
        default="0,1,2,3",
        help="Comma-separated session-boundary offsets in market bars.",
    )
    parser.add_argument(
        "--mode",
        choices=("combined", "per_offset"),
        default="combined",
        help="Write a single combined augmented dataset per symbol, or one directory per offset.",
    )
    parser.add_argument(
        "--bars-per-session",
        type=int,
        default=None,
        help="Optional fixed session bar count. Default: infer from mode of bars/day.",
    )
    parser.add_argument(
        "--separator-days",
        type=int,
        default=14,
        help="Calendar-day gap inserted between offset blocks in combined mode.",
    )
    parser.add_argument("--start-date", default=None, help="Optional ISO start date filter for hourly rows.")
    parser.add_argument("--end-date", default=None, help="Optional ISO end date filter for hourly rows.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = build_shifted_daily_dataset(
        hourly_root=Path(args.hourly_root),
        output_root=Path(args.output_root),
        symbols=args.symbols or (),
        offsets=_parse_offsets(args.offsets),
        mode=str(args.mode),
        bars_per_session=args.bars_per_session,
        separator_days=int(args.separator_days),
        start_date=args.start_date,
        end_date=args.end_date,
        force=bool(args.force),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
