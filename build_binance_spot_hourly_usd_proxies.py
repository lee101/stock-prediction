#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.binance_symbol_utils import normalize_compact_symbol, proxy_symbol_to_usd


def _iter_symbols(items: Optional[Sequence[str]]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        for token in str(item).split(","):
            symbol = normalize_compact_symbol(token)
            if not symbol or symbol in seen:
                continue
            resolved.append(symbol)
            seen.add(symbol)
    return resolved


def _build_one(*, hourly_root: Path, symbol: str, force: bool) -> Optional[Path]:
    source_path = hourly_root / f"{symbol}.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing hourly Binance CSV: {source_path}")

    proxy_symbol = proxy_symbol_to_usd(symbol)
    if proxy_symbol == symbol:
        logger.info("Symbol {} already uses USD proxy naming; skipping.", symbol)
        return None

    proxy_path = hourly_root / f"{proxy_symbol}.csv"
    if proxy_path.exists() and not force:
        logger.info("Hourly USD proxy already exists for {} -> {} (skip; use --force to rebuild).", symbol, proxy_path)
        return proxy_path

    frame = pd.read_csv(source_path)
    if "symbol" in frame.columns:
        frame["symbol"] = proxy_symbol
    frame.to_csv(proxy_path, index=False)
    logger.info("Wrote hourly USD proxy {} -> {}", symbol, proxy_path)
    return proxy_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build hourly USD proxy CSVs from Binance stable-quote spot CSVs (e.g. SUIUSDT -> SUIUSD).",
    )
    parser.add_argument(
        "--hourly-root",
        type=Path,
        default=Path("trainingdatahourly/crypto"),
        help="Directory containing hourly Binance CSVs.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Stable-quote compact symbols to proxy (e.g. SUIUSDT BTCFDUSD).",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild proxy CSVs even if they already exist.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbols = _iter_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No symbols provided.")

    hourly_root = Path(args.hourly_root)
    written = 0
    for symbol in symbols:
        result = _build_one(hourly_root=hourly_root, symbol=symbol, force=bool(args.force))
        if result is not None:
            written += 1

    logger.info("Done. Wrote/confirmed {} hourly USD proxy file(s).", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
