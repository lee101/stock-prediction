#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stock_universe_builder import rank_stock_universe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a ranked stock universe from local daily CSVs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("trainingdata/train"),
        help="Directory containing daily CSVs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1000,
        help="Maximum number of ranked symbols to emit.",
    )
    parser.add_argument(
        "--lookback-rows",
        type=int,
        default=60,
        help="Rows used to estimate recent dollar volume.",
    )
    parser.add_argument(
        "--min-history-rows",
        type=int,
        default=252,
        help="Minimum history rows required to keep a symbol.",
    )
    parser.add_argument(
        "--min-last-close",
        type=float,
        default=3.0,
        help="Minimum latest close price.",
    )
    parser.add_argument(
        "--min-median-dollar-volume",
        type=float,
        default=2_000_000.0,
        help="Minimum trailing median dollar volume.",
    )
    parser.add_argument(
        "--min-last-date",
        type=str,
        default="",
        help="Optional ISO date cutoff; keep symbols whose latest bar is on/after this date.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("symbol_lists/stocks_1000_v1.txt"),
        help="Newline-delimited symbol output.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("analysis/stocks_1000_v1.json"),
        help="JSON summary output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    csv_paths = sorted(args.data_dir.glob("*.csv"))
    ranked = rank_stock_universe(
        csv_paths,
        lookback_rows=args.lookback_rows,
        min_history_rows=args.min_history_rows,
        min_last_close=args.min_last_close,
        min_median_dollar_volume=args.min_median_dollar_volume,
        min_last_timestamp=(f"{args.min_last_date}T00:00:00+00:00" if args.min_last_date else None),
        top_n=args.top_n,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        "".join(f"{candidate.symbol}\n" for candidate in ranked),
        encoding="utf-8",
    )

    payload = {
        "data_dir": str(args.data_dir),
        "top_n": args.top_n,
        "lookback_rows": args.lookback_rows,
        "min_history_rows": args.min_history_rows,
        "min_last_close": args.min_last_close,
        "min_median_dollar_volume": args.min_median_dollar_volume,
        "min_last_date": args.min_last_date,
        "selected_count": len(ranked),
        "symbols": [candidate.symbol for candidate in ranked],
        "candidates": [candidate.to_dict() for candidate in ranked],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(ranked)} symbols to {args.output_file}")
    print(f"Wrote universe summary to {args.json_out}")
    if ranked:
        print(f"Top 10: {', '.join(candidate.symbol for candidate in ranked[:10])}")
    else:
        print("Top 10: <empty>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
