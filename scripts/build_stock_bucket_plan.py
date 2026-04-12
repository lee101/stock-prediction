#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alpaca_stock_expansion import (  # noqa: E402
    build_candidate_sector_buckets,
    build_hourly_return_correlation_cohorts,
    default_stock_expansion_candidates,
)
from src.daily_stock_defaults import DEFAULT_SYMBOLS as DEFAULT_DAILY_STOCK_SYMBOLS  # noqa: E402
from src.hourly_data_utils import resolve_hourly_symbol_path  # noqa: E402
from src.remote_training_pipeline import build_remote_large_universe_stock_plan  # noqa: E402


def _load_symbols_file(path: Path) -> list[str]:
    symbols: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        symbol = stripped.split("#", 1)[0].strip().upper()
        if symbol:
            symbols.append(symbol)
    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        deduped.append(symbol)
        seen.add(symbol)
    return deduped


def _existing_hourly_symbols(symbols: list[str], *, data_root: Path) -> list[str]:
    return [
        symbol for symbol in symbols
        if resolve_hourly_symbol_path(symbol, data_root) is not None
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a bucketed stock-research plan from a wide symbol list.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=Path("symbol_lists/expanded_research_stocks_v1.txt"),
        help="Research symbol list to load.",
    )
    parser.add_argument(
        "--hourly-data-root",
        type=Path,
        default=Path("trainingdatahourly"),
        help="Hourly data root used to validate coverage and compute correlation cohorts.",
    )
    parser.add_argument(
        "--daily-data-root",
        type=Path,
        default=Path("trainingdata"),
        help="Daily data root used for the remote large-universe plan summary.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24 * 120,
        help="Lookback window used for correlation cohorts.",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=24 * 5,
        help="Minimum overlap hours required for correlation estimation.",
    )
    parser.add_argument(
        "--max-cohort-size",
        type=int,
        default=4,
        help="Maximum peers to keep per symbol in the correlation cohort map.",
    )
    parser.add_argument(
        "--min-abs-corr",
        type=float,
        default=0.25,
        help="Minimum absolute hourly return correlation to retain a peer.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("analysis/stock_bucket_plan_v1.json"),
        help="Where to write the bucket plan JSON summary.",
    )
    parser.add_argument(
        "--train-hours",
        type=int,
        default=24 * 30,
        help="Training window for the remote large-universe plan.",
    )
    parser.add_argument(
        "--val-hours",
        type=int,
        default=24 * 15,
        help="Validation window for the remote large-universe plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = _load_symbols_file(args.symbols_file)
    if not symbols:
        raise SystemExit(f"No symbols found in {args.symbols_file}")

    hourly_ready = _existing_hourly_symbols(symbols, data_root=args.hourly_data_root)
    missing_hourly = [symbol for symbol in symbols if symbol not in set(hourly_ready)]

    sector_buckets = build_candidate_sector_buckets(
        default_stock_expansion_candidates(),
        include_symbols=symbols,
    )
    if DEFAULT_DAILY_STOCK_SYMBOLS:
        sector_buckets["base_live_daily"] = tuple(DEFAULT_DAILY_STOCK_SYMBOLS)

    correlation_cohorts = build_hourly_return_correlation_cohorts(
        hourly_ready,
        data_root=args.hourly_data_root,
        lookback_hours=args.lookback_hours,
        min_periods=args.min_periods,
        max_size=args.max_cohort_size,
        min_abs_corr=args.min_abs_corr,
    ) if hourly_ready else {}

    remote_plan = None
    remote_plan_error = None
    if hourly_ready:
        try:
            plan = build_remote_large_universe_stock_plan(
                run_id="stocks_large_universe_v1",
                symbols=hourly_ready,
                local_hourly_data_root=args.hourly_data_root,
                remote_hourly_data_root="trainingdatahourly/stocks",
                local_daily_data_root=args.daily_data_root,
                remote_daily_data_root="trainingdata/stocks",
                train_hours=args.train_hours,
                val_hours=args.val_hours,
                gap_hours=24,
            )
            remote_plan = {
                "run_id": plan.run_id,
                "symbol_count": len(plan.symbols),
                "symbols": list(plan.symbols),
                "remote_run_dir": plan.remote_run_dir,
                "hourly_train_data_path": plan.hourly_train_data_path,
                "hourly_val_data_path": plan.hourly_val_data_path,
                "daily_train_data_path": plan.daily_train_data_path,
                "daily_val_data_path": plan.daily_val_data_path,
            }
        except Exception as exc:  # pragma: no cover - summary path only
            remote_plan_error = f"{type(exc).__name__}: {exc}"

    payload = {
        "source_symbols_file": str(args.symbols_file),
        "symbol_count": len(symbols),
        "symbols": symbols,
        "hourly_ready_count": len(hourly_ready),
        "hourly_ready_symbols": hourly_ready,
        "missing_hourly_symbols": missing_hourly,
        "sector_buckets": {key: list(value) for key, value in sector_buckets.items()},
        "correlation_cohorts": {key: list(value) for key, value in correlation_cohorts.items()},
        "remote_large_universe_plan": remote_plan,
        "remote_large_universe_plan_error": remote_plan_error,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote bucket plan to {args.json_out}")
    print(f"Loaded {len(symbols)} symbols; {len(hourly_ready)} have hourly coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
