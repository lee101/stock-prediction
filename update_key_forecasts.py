#!/usr/bin/env python3
"""
Update Chronos forecasts for key symbols that already have cached history.

This module exposes ``collect_forecasts`` so other automation (e.g., the daily
refresh pipeline) can import it without shelling out to a subprocess.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_DATA_DIR = Path("trainingdata/train")
DEFAULT_CACHE_DIR = Path("strategytraining/forecast_cache")


def _discover_cached_symbols(cache_dir: Path) -> List[str]:
    forecast_files = sorted(cache_dir.glob("*.parquet"))
    return [f.stem for f in forecast_files]


def _canonicalize_symbols(symbols: Iterable[str]) -> List[str]:
    canonical: List[str] = []
    for symbol in symbols:
        safe = symbol.upper().replace("/", "-")
        canonical.append(safe)
    return canonical


def collect_forecasts(
    symbols: Optional[Sequence[str]] = None,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    context_length: int = 512,
    batch_size: int = 32,
    device_map: str = "cuda",
    extra_args: Optional[Sequence[str]] = None,
) -> int:
    """Run Chronos collection for the provided symbols."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if symbols is None:
        symbols = _discover_cached_symbols(cache_dir)
        if not symbols:
            raise RuntimeError(
                f"No cached forecasts found under {cache_dir}; "
                "provide --symbol arguments to bootstrap the cache."
            )
    symbol_list = _canonicalize_symbols(symbols)
    print(f"Updating Chronos forecasts for {len(symbol_list)} symbols: {symbol_list}")

    cmd = [
        sys.executable,
        "-m",
        "strategytrainingneural.collect_forecasts",
        "--data-dir",
        str(data_dir),
        "--cache-dir",
        str(cache_dir),
        "--context-length",
        str(context_length),
        "--batch-size",
        str(batch_size),
        "--device-map",
        device_map,
    ]
    for symbol in symbol_list:
        cmd.extend(["--symbol", symbol])
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "." if not existing_pythonpath else f".{os.pathsep}{existing_pythonpath}"
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Chronos forecast cache.")
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Explicit symbol to refresh (can be specified multiple times).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Training data directory (default: trainingdata/train).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Forecast cache directory (default: strategytraining/forecast_cache).",
    )
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument(
        "--extra-arg",
        action="append",
        dest="extra_args",
        help="Additional argument to pass through to the Chronos collector.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    symbols = args.symbols
    try:
        returncode = collect_forecasts(
            symbols,
            data_dir=Path(args.data_dir),
            cache_dir=Path(args.cache_dir),
            context_length=args.context_length,
            batch_size=args.batch_size,
            device_map=args.device_map,
            extra_args=args.extra_args,
        )
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"\n❌ Forecast update failed: {exc}")
        return 1

    if returncode == 0:
        print("\n✅ Chronos forecast update complete!")
    else:
        print(f"\n❌ Forecast update failed with code {returncode}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
