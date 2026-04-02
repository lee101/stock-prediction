#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from src.symbol_file_utils import load_symbols_from_file

REPO = Path(__file__).resolve().parents[1]


def _load_symbols(path: Path) -> list[str]:
    return load_symbols_from_file(path)


def _chunk(symbols: list[str], batch_size: int) -> list[list[str]]:
    return [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stock forecast caches in batches using scripts/build_hourly_forecast_caches.py",
    )
    parser.add_argument("--symbols-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--end-batch", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--forecast-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--horizons", default="1,24")
    parser.add_argument("--lookback-hours", type=float, default=24.0 * 365.0)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/forecast_cache_batches"))
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = _load_symbols(args.symbols_file)
    if not symbols:
        raise SystemExit(f"No symbols found in {args.symbols_file}")

    batches = _chunk(symbols, max(1, int(args.batch_size)))
    start_batch = max(1, int(args.start_batch))
    end_batch = int(args.end_batch) if int(args.end_batch) > 0 else len(batches)
    if start_batch > end_batch or start_batch > len(batches):
        raise SystemExit(
            f"Invalid batch range start={start_batch} end={end_batch} total_batches={len(batches)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(start_batch, min(end_batch, len(batches)) + 1):
        batch_symbols = batches[index - 1]
        batch_file = args.output_dir / f"batch_{index:03d}_symbols.txt"
        summary_file = args.output_dir / f"batch_{index:03d}_mae.json"
        batch_file.write_text("\n".join(batch_symbols) + "\n")
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "build_hourly_forecast_caches.py"),
            "--symbols-file",
            str(batch_file),
            "--data-root",
            str(args.data_root),
            "--forecast-cache-root",
            str(args.forecast_cache_root),
            "--horizons",
            str(args.horizons),
            "--lookback-hours",
            str(float(args.lookback_hours)),
            "--output-json",
            str(summary_file),
        ]
        if args.force_rebuild:
            cmd.append("--force-rebuild")
        print(f"[batch {index}/{len(batches)}] {len(batch_symbols)} symbols")
        print(" ".join(shlex.quote(part) for part in cmd))
        if args.execute:
            subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
