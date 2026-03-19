from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from loguru import logger

# Allow running as `python scripts/...` without needing PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import binance_data_wrapper


DEFAULT_ZERO_FEE_PAIRS: Sequence[str] = (
    "BTC/FDUSD",
    "ETH/FDUSD",
    "SOL/FDUSD",
    "BNB/FDUSD",
    # Stablecoin conversion for manual USDT -> FDUSD moves.
    "FDUSD/USDT",
    # Optional: useful for EUR conversions / checks.
    "AEUR/USDT",
)

DEFAULT_U_ZERO_FEE_PAIRS: Sequence[str] = (
    "BTC/U",
    "ETH/U",
    "SOL/U",
    "BNB/U",
    # Stablecoin conversion for manual USDT -> U moves.
    "U/USDT",
)


def _maybe_create_symlink(link_path: Path, target_path: Path) -> None:
    """Best-effort helper to keep `trainingdatahourlybinance` paths working."""
    link_path = Path(link_path)
    target_path = Path(target_path)
    if link_path.exists() or link_path.is_symlink():
        return
    try:
        rel = os.path.relpath(target_path, start=link_path.parent)
        link_path.symlink_to(rel)
        logger.info(f"Created symlink {link_path} -> {rel}")
    except OSError as exc:
        logger.warning(f"Failed creating symlink {link_path} -> {target_path}: {exc}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download hourly Binance spot klines for curated zero-fee pairs (FDUSD/U) "
            "into binancetrainingdatahourly/."
        )
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help=(
            "Pairs to download (default: BTC/FDUSD ETH/FDUSD SOL/FDUSD BNB/FDUSD "
            "FDUSD/USDT AEUR/USDT)."
        ),
    )
    parser.add_argument(
        "--all-fdusd",
        action="store_true",
        help=(
            "Download the curated FDUSD list from binance_data_wrapper "
            "(plus FDUSD/USDT and AEUR/USDT). Ignored when --pairs is provided."
        ),
    )
    parser.add_argument(
        "--all-u",
        action="store_true",
        help=(
            "Download the curated U list from binance_data_wrapper "
            "(plus U/USDT). Ignored when --pairs is provided."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("binancetrainingdatahourly"),
        help="Directory for Binance hourly CSVs (default: binancetrainingdatahourly).",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=binance_data_wrapper.DEFAULT_HISTORY_YEARS,
        help=f"Years of history to attempt (default: {binance_data_wrapper.DEFAULT_HISTORY_YEARS}).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=binance_data_wrapper.DEFAULT_SLEEP_SECONDS,
        help=f"Seconds to sleep between pairs (default: {binance_data_wrapper.DEFAULT_SLEEP_SECONDS}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local files appear up to date.",
    )
    parser.add_argument(
        "--fallback-quote",
        action="append",
        default=None,
        help=(
            "Fallback quote asset(s) if a requested pair is unavailable. "
            "Omit to disable fallbacks (recommended for zero-fee FDUSD training)."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.pairs:
        pairs = list(args.pairs)
    elif args.all_fdusd:
        pairs = list(binance_data_wrapper.DEFAULT_BINANCE_FDUSD_PAIRS)
        for extra in ("FDUSD/USDT", "AEUR/USDT"):
            if extra not in pairs:
                pairs.append(extra)
    elif args.all_u:
        pairs = list(binance_data_wrapper.DEFAULT_BINANCE_U_PAIRS)
    else:
        pairs = list(DEFAULT_ZERO_FEE_PAIRS)
    output_dir = Path(args.output_dir)
    history_years = int(args.history_years)
    sleep_seconds = float(args.sleep)
    skip_if_exists = not bool(args.force)

    # Ensure the output directory exists up-front. If we are using one of the two
    # common local names, create a best-effort alias symlink so older scripts
    # that default to `trainingdatahourlybinance` keep working.
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir == Path("binancetrainingdatahourly"):
        _maybe_create_symlink(Path("trainingdatahourlybinance"), output_dir)
    elif output_dir == Path("trainingdatahourlybinance"):
        _maybe_create_symlink(Path("binancetrainingdatahourly"), output_dir)

    fallback_quotes: list[str]
    if args.fallback_quote:
        fallback_quotes = [str(q).strip().upper() for q in args.fallback_quote if str(q).strip()]
    else:
        fallback_quotes = []

    logger.info(f"Downloading Binance hourly data for {len(pairs)} pair(s) into {output_dir}")
    if not fallback_quotes:
        logger.info("Fallback quotes disabled (strict pair download).")
    else:
        logger.info(f"Fallback quotes enabled: {','.join(fallback_quotes)}")

    try:
        results = binance_data_wrapper.download_all_pairs(
            pairs=pairs,
            output_dir=output_dir,
            history_years=history_years,
            sleep_seconds=sleep_seconds,
            skip_if_exists=skip_if_exists,
            fallback_quotes=fallback_quotes,
        )
    except Exception as exc:
        logger.error(f"Zero-fee Binance download failed: {exc}")
        sys.exit(1)

    missing = [r for r in results if r.status in {"invalid", "unavailable", "no_data"}]
    if missing:
        logger.warning("Some pairs were not downloaded cleanly:")
        for item in missing:
            logger.warning(
                "pair={} resolved={} status={} error={}",
                item.symbol,
                item.resolved_symbol,
                item.status,
                item.error,
            )


if __name__ == "__main__":
    main()
