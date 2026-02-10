#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
from loguru import logger

# Allow running as `python scripts/...` without needing PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.bar_aggregation import hourly_to_daily_ohlcv
from src.binance_symbol_utils import proxy_symbol_to_usd, normalize_compact_symbol


def _maybe_create_symlink(link_path: Path, target_path: Path) -> None:
    """Best-effort helper to keep `trainingdatadailybinance` paths working."""
    link_path = Path(link_path)
    target_path = Path(target_path)
    if link_path.exists() or link_path.is_symlink():
        return
    try:
        rel = os.path.relpath(target_path, start=link_path.parent)
        link_path.symlink_to(rel)
        logger.info("Created symlink {} -> {}", link_path, rel)
    except OSError as exc:
        logger.warning("Failed creating symlink {} -> {}: {}", link_path, target_path, exc)


def _iter_symbols(items: Optional[Sequence[str]]) -> list[str]:
    symbols: list[str] = []
    for item in items or []:
        normalized = normalize_compact_symbol(str(item))
        if normalized:
            symbols.append(normalized)
    # Stable order for reproducibility.
    return sorted(set(symbols))


def _build_one(
    *,
    symbol: str,
    hourly_root: Path,
    output_dir: Path,
    expected_bars_per_day: int,
    drop_incomplete_days: bool,
    write_proxy_usd: bool,
    force: bool,
) -> list[Path]:
    hourly_path = hourly_root / f"{symbol}.csv"
    if not hourly_path.exists():
        raise FileNotFoundError(f"Missing hourly Binance CSV: {hourly_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / f"{symbol}.csv"

    written: list[Path] = []
    if daily_path.exists() and not force:
        logger.info("Daily CSV already exists for {} (skip; use --force to rebuild): {}", symbol, daily_path)
        written.append(daily_path)
    else:
        hourly = pd.read_csv(hourly_path)
        daily, stats = hourly_to_daily_ohlcv(
            hourly,
            expected_bars_per_day=expected_bars_per_day,
            drop_incomplete_last_day=True,
            drop_incomplete_days=drop_incomplete_days,
            output_symbol=symbol,
        )
        if daily.empty:
            raise RuntimeError(f"No daily rows produced for {symbol} from {hourly_path}")

        # Keep schema compatible with Binance hourly CSVs (timestamp as column, no index).
        daily = daily.drop(columns=["bar_count"], errors="ignore")
        daily.to_csv(daily_path, index=False)
        written.append(daily_path)
        logger.info(
            "Wrote daily bars for {} (rows={}, dropped_incomplete_last_day={}, dropped_incomplete_days={}): {}",
            symbol,
            len(daily),
            stats.dropped_incomplete_last_day,
            stats.dropped_incomplete_days,
            daily_path,
        )

    if write_proxy_usd:
        proxy = proxy_symbol_to_usd(symbol)
        proxy = normalize_compact_symbol(proxy)
        if proxy and proxy != symbol:
            proxy_path = output_dir / f"{proxy}.csv"
            if proxy_path.exists() and not force:
                logger.info("USD proxy daily CSV already exists for {} -> {} (skip): {}", symbol, proxy, proxy_path)
                written.append(proxy_path)
            else:
                df = pd.read_csv(daily_path)
                if "symbol" in df.columns:
                    df["symbol"] = proxy
                df.to_csv(proxy_path, index=False)
                written.append(proxy_path)
                logger.info("Wrote USD proxy daily bars {} -> {}", symbol, proxy_path)

    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build daily Binance spot OHLCV CSVs by aggregating existing hourly CSVs (UTC days).",
    )
    parser.add_argument(
        "--hourly-root",
        type=Path,
        default=Path("trainingdatahourlybinance"),
        help="Root directory containing hourly Binance CSVs (default: trainingdatahourlybinance).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("binance_spot_daily"),
        help="Directory to write daily CSVs (default: binance_spot_daily).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=("SOLFDUSD",),
        help="Symbols to aggregate (default: SOLFDUSD).",
    )
    parser.add_argument(
        "--expected-bars-per-day",
        type=int,
        default=24,
        help="Expected hourly bars per day (default: 24).",
    )
    parser.add_argument(
        "--drop-incomplete-days",
        action="store_true",
        help="Drop all days with fewer than expected bars (default: only drop trailing incomplete day).",
    )
    parser.add_argument(
        "--no-proxy-usd",
        action="store_true",
        help="Do not write USD proxy symbols (e.g., SOLFDUSD -> SOLUSD).",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if they already exist.")
    parser.add_argument(
        "--symlink-alias",
        action="store_true",
        default=True,
        help="Create/update trainingdatadailybinance symlink (default: enabled).",
    )
    parser.add_argument(
        "--no-symlink-alias",
        action="store_false",
        dest="symlink_alias",
        help="Disable creating trainingdatadailybinance symlink.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbols = _iter_symbols(args.symbols)
    if not symbols:
        logger.error("No symbols provided.")
        return 2

    hourly_root = Path(args.hourly_root)
    output_dir = Path(args.output_dir)
    expected = max(1, int(args.expected_bars_per_day))
    drop_incomplete_days = bool(args.drop_incomplete_days)
    write_proxy_usd = not bool(args.no_proxy_usd)
    force = bool(args.force)

    all_written: list[Path] = []
    for symbol in symbols:
        all_written.extend(
            _build_one(
                symbol=symbol,
                hourly_root=hourly_root,
                output_dir=output_dir,
                expected_bars_per_day=expected,
                drop_incomplete_days=drop_incomplete_days,
                write_proxy_usd=write_proxy_usd,
                force=force,
            )
        )

    if args.symlink_alias:
        if output_dir == Path("binance_spot_daily"):
            _maybe_create_symlink(Path("trainingdatadailybinance"), output_dir)
        elif output_dir == Path("trainingdatadailybinance"):
            _maybe_create_symlink(Path("binance_spot_daily"), output_dir)

    logger.info("Done. Wrote/confirmed {} file(s).", len(set(all_written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

