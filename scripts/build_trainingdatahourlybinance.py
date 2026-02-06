#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from src.binance_symbol_utils import (
    DEFAULT_STABLE_QUOTES,
    normalize_compact_symbol,
    stable_quote_aliases_from_usd,
    unique_symbols,
)


def _iter_csv_symbols(folder: Path) -> List[Tuple[str, Path]]:
    if not folder.exists():
        return []
    items: List[Tuple[str, Path]] = []
    for path in sorted(folder.glob("*.csv")):
        stem = path.stem
        if stem.lower() in {"summary", "download_summary"}:
            continue
        items.append((normalize_compact_symbol(stem), path))
    return items


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _same_symlink_target(link_path: Path, target_path: Path) -> bool:
    if not link_path.is_symlink():
        return False
    try:
        resolved = link_path.resolve(strict=True)
    except FileNotFoundError:
        return False
    return resolved == target_path.resolve()


def _ensure_symlink(
    link_path: Path,
    target_path: Path,
    *,
    force: bool,
) -> str:
    """Create/update a symlink; returns status string."""
    if not target_path.exists():
        return "missing_target"

    if link_path.exists() or link_path.is_symlink():
        if _same_symlink_target(link_path, target_path):
            return "exists"
        if not force:
            return "skip_conflict"
        _safe_unlink(link_path)

    link_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer relative symlinks so the repo stays relocatable.
    rel = os.path.relpath(target_path, start=link_path.parent)
    link_path.symlink_to(rel)
    return "linked"


def build_trainingdatahourlybinance(
    *,
    output_dir: Path,
    crypto_source_dir: Path,
    stock_source_dir: Path,
    stable_quotes: Sequence[str] = DEFAULT_STABLE_QUOTES,
    force: bool = False,
    include_usd_files: bool = True,
) -> Dict[str, int]:
    output_dir = Path(output_dir)
    crypto_source_dir = Path(crypto_source_dir)
    stock_source_dir = Path(stock_source_dir)

    crypto_items = _iter_csv_symbols(crypto_source_dir)
    stock_items = _iter_csv_symbols(stock_source_dir)

    counts: Dict[str, int] = {
        "linked": 0,
        "exists": 0,
        "missing_target": 0,
        "skip_conflict": 0,
    }

    # Stocks: link exact ticker name (e.g., NVDA.csv)
    for symbol, src_path in stock_items:
        status = _ensure_symlink(output_dir / f"{symbol}.csv", src_path, force=force)
        counts[status] = counts.get(status, 0) + 1

    # Crypto: link USD symbol + stable-quote aliases (e.g., BTCUSD.csv -> BTCUSDT.csv)
    for symbol, src_path in crypto_items:
        to_link: List[str] = []
        if include_usd_files:
            to_link.append(symbol)
        to_link.extend(stable_quote_aliases_from_usd(symbol, stable_quotes=stable_quotes))
        for alias in unique_symbols(to_link):
            status = _ensure_symlink(output_dir / f"{alias}.csv", src_path, force=force)
            counts[status] = counts.get(status, 0) + 1

    logger.info(
        "trainingdatahourlybinance: {} linked, {} existed, {} missing targets, {} conflicts skipped.",
        counts.get("linked", 0),
        counts.get("exists", 0),
        counts.get("missing_target", 0),
        counts.get("skip_conflict", 0),
    )
    return counts


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build trainingdatahourlybinance/ as symlinks to Alpaca hourly CSVs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("trainingdatahourlybinance"))
    parser.add_argument("--crypto-source-dir", type=Path, default=Path("trainingdatahourly") / "crypto")
    parser.add_argument("--stock-source-dir", type=Path, default=Path("trainingdatahourly") / "stocks")
    parser.add_argument(
        "--stable-quote",
        action="append",
        default=None,
        help="Stable quote suffix to create aliases for (repeatable). Defaults to a curated list.",
    )
    parser.add_argument("--force", action="store_true", help="Replace conflicting files/links.")
    parser.add_argument(
        "--no-usd-files",
        action="store_true",
        help="Skip linking the USD symbols themselves; only create stable-quote aliases.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    stable_quotes = args.stable_quote or list(DEFAULT_STABLE_QUOTES)

    try:
        build_trainingdatahourlybinance(
            output_dir=args.output_dir,
            crypto_source_dir=args.crypto_source_dir,
            stock_source_dir=args.stock_source_dir,
            stable_quotes=stable_quotes,
            force=args.force,
            include_usd_files=not args.no_usd_files,
        )
    except Exception as exc:
        logger.exception("Failed building trainingdatahourlybinance: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
