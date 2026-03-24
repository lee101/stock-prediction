#!/usr/bin/env python3
"""Generate missing forecast caches for all symbols.

Usage:
    python -m sharpnessadjustedproximalpolicy.generate_caches
    python -m sharpnessadjustedproximalpolicy.generate_caches --symbols BTCUSD ETHUSD --horizons 1 4 24
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_available_symbols():
    data_root = Path("trainingdatahourly") / "crypto"
    return sorted([p.stem for p in data_root.glob("*USD.csv")])


def check_cache(symbol: str, horizon: int, cache_root: Path) -> bool:
    return (cache_root / f"h{horizon}" / f"{symbol}.parquet").exists()


def generate_cache(symbol: str, horizon: int):
    from binanceneural.forecasts import build_forecast_bundle
    print(f"Generating h{horizon} cache for {symbol}...", flush=True)
    t0 = time.time()
    build_forecast_bundle(
        symbol=symbol,
        data_root=Path("trainingdatahourly") / "crypto",
        cache_root=Path("binanceneural") / "forecast_cache",
        horizons=(horizon,),
        context_hours=24 * 14,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=128,
        cache_only=False,
    )
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 24])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = args.symbols or get_available_symbols()
    cache_root = Path("binanceneural") / "forecast_cache"

    missing = []
    for sym in symbols:
        for h in args.horizons:
            if not check_cache(sym, h, cache_root):
                missing.append((sym, h))

    if not missing:
        print("All caches present.")
        return

    print(f"Missing {len(missing)} caches:")
    for sym, h in missing:
        print(f"  {sym} h{h}")

    if args.dry_run:
        return

    for i, (sym, h) in enumerate(missing):
        print(f"\n[{i+1}/{len(missing)}]", flush=True)
        try:
            generate_cache(sym, h)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
