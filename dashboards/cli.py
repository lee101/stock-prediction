from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

if __name__ == "__main__" and __package__ is None:  # pragma: no cover - support direct execution
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dashboards.config import load_config
    from dashboards.db import DashboardDatabase, MetricEntry, ShelfSnapshot
else:
    from .config import load_config
    from .db import DashboardDatabase, MetricEntry, ShelfSnapshot


def _downsample_points(points: Sequence[Tuple[datetime, float]], width: int) -> List[Tuple[datetime, float]]:
    if len(points) <= width:
        return list(points)
    step = max(1, int(len(points) / width))
    sampled: List[Tuple[datetime, float]] = []
    for idx in range(0, len(points), step):
        sampled.append(points[idx])
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _render_ascii_chart(points: Sequence[Tuple[datetime, float]], width: int = 80, height: int = 10) -> str:
    if not points:
        return "No data available for chart."

    sampled = _downsample_points(points, width)
    values = [value for _, value in sampled]
    min_val = min(values)
    max_val = max(values)
    if abs(max_val - min_val) < 1e-6:
        max_val += 1.0
        min_val -= 1.0

    span = max_val - min_val
    normalized = [
        0 if span == 0 else int(round((val - min_val) / span * (height - 1)))
        for val in values
    ]

    grid = [[" " for _ in range(len(sampled))] for _ in range(height)]
    for idx, level in enumerate(normalized):
        row_idx = height - 1 - level
        grid[row_idx][idx] = "*"

    labels = []
    for row_idx, row in enumerate(grid):
        label_val = max_val - (span * row_idx / max(1, height - 1))
        labels.append(f"{label_val:>10.2f} |{''.join(row)}")

    axis = " " * 10 + "+" + "-" * len(sampled)
    labels.append(axis)

    start_ts = sampled[0][0].strftime("%Y-%m-%d %H:%M")
    end_ts = sampled[-1][0].strftime("%Y-%m-%d %H:%M")
    labels.append(f"{start_ts:<21}{end_ts:>21}")
    return "\n".join(labels)


def _format_metric_value(value: Optional[float]) -> str:
    if value is None:
        return "—"
    abs_val = abs(value)
    if abs_val >= 1000:
        return f"{value:,.2f}"
    if abs_val >= 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def handle_metrics(args: argparse.Namespace) -> int:
    config = load_config()
    symbol = args.symbol.upper() if args.symbol else None
    with DashboardDatabase(config) as db:
        rows = list(
            db.iter_metrics(
                metric=args.metric,
                symbol=symbol,
                source=args.source,
                limit=args.limit,
            )
        )
    if not rows:
        scope = f" for {symbol}" if symbol else ""
        source_part = f" [{args.source}]" if args.source else ""
        print(f"No metrics stored for '{args.metric}'{scope}{source_part}.")
        return 1

    rows = list(reversed(rows))
    print(
        f"Latest {len(rows)} samples for metric '{args.metric}'"
        + (f" (source={args.source})" if args.source else "")
        + (f" (symbol={symbol})" if symbol else "")
        + ":"
    )
    header = f"{'Timestamp (UTC)':<25}{'Source':>14}{'Symbol':>10}{'Value':>14}"
    print(header)
    print("-" * len(header))
    for entry in rows[-args.table_rows :]:
        ts = entry.recorded_at.strftime("%Y-%m-%d %H:%M:%S")
        source = entry.source
        sym = entry.symbol or "—"
        value = _format_metric_value(entry.value)
        print(f"{ts:<25}{source:>14}{sym:>10}{value:>14}")

    if args.chart:
        chart_points = [(entry.recorded_at, entry.value) for entry in rows if entry.value is not None]
        if chart_points:
            print()
            print("Metric chart:")
            print(_render_ascii_chart(chart_points, width=args.chart_width, height=args.chart_height))
        else:
            print("\nNo numeric values available to chart for this metric.")

    if args.show_message:
        latest = rows[-1]
        if latest.message:
            print()
            print("Most recent log message:")
            print(latest.message)

    return 0


def handle_spreads(args: argparse.Namespace) -> int:
    config = load_config()
    symbol = args.symbol.upper()
    with DashboardDatabase(config) as db:
        observations = list(db.iter_spreads(symbol, limit=args.limit))
    if not observations:
        print(f"No spread observations stored for {symbol}.")
        return 1

    observations = list(reversed(observations))
    print(f"Latest {len(observations)} spread points for {symbol}:")
    header = f"{'Timestamp (UTC)':<25}{'Bid':>12}{'Ask':>12}{'Spread(bps)':>14}{'Spread(%)':>12}"
    print(header)
    print("-" * len(header))
    for obs in observations[-args.table_rows :]:
        bid = f"{obs.bid:.4f}" if obs.bid is not None else "—"
        ask = f"{obs.ask:.4f}" if obs.ask is not None else "—"
        spread_bps = obs.spread_bps
        spread_pct = (obs.spread_ratio - 1.0) * 100
        timestamp = obs.recorded_at.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp:<25}{bid:>12}{ask:>12}{spread_bps:>14.2f}{spread_pct:>12.4f}")

    if args.chart:
        points = [(obs.recorded_at, obs.spread_bps) for obs in observations]
        print()
        print("Spread (bps) chart:")
        print(_render_ascii_chart(points, width=args.chart_width, height=args.chart_height))
    return 0


def _load_snapshot_json(snapshot: ShelfSnapshot) -> Optional[dict]:
    try:
        return json.loads(snapshot.data)
    except json.JSONDecodeError:
        return None


def handle_shelves(args: argparse.Namespace) -> int:
    config = load_config()
    if args.file:
        shelf_path = Path(args.file).expanduser().resolve()
    else:
        if not config.shelf_files:
            print("No shelf files configured. Use --file to specify one.")
            return 1
        shelf_path = config.shelf_files[0]

    with DashboardDatabase(config) as db:
        snapshots = list(db.iter_latest_snapshots(shelf_path, limit=args.limit))
    if not snapshots:
        print(f"No snapshots recorded for {shelf_path}.")
        return 1

    print(f"Stored snapshots for {shelf_path}:")
    print(f"{'Timestamp (UTC)':<25}{'Bytes':>10}{'SHA256':>18}")
    print("-" * 55)
    for snapshot in snapshots:
        ts = snapshot.recorded_at.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts:<25}{snapshot.bytes:>10}{snapshot.sha256[:16]:>18}")

    latest = snapshots[0]
    if args.summary:
        payload = _load_snapshot_json(latest)
        if isinstance(payload, dict):
            total_entries = len(payload)
            strategy_counter = Counter(payload.values())
            top_strategies = strategy_counter.most_common(5)
            print()
            print(f"Latest snapshot summary ({latest.recorded_at.isoformat()}):")
            print(f"  Total entries: {total_entries}")
            print("  Top strategies:")
            for strategy, count in top_strategies:
                print(f"    - {strategy}: {count}")
        else:
            print("Unable to parse latest snapshot JSON for summary.")

    if args.show_json:
        print()
        print(f"Latest snapshot JSON ({latest.recorded_at.isoformat()}):")
        print(latest.data)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dashboards CLI for vanity metrics and spreads.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    spreads_parser = subparsers.add_parser("spreads", help="Inspect spread history for a symbol.")
    spreads_parser.add_argument("--symbol", required=True, help="Symbol to inspect (e.g. AAPL, BTCUSD).")
    spreads_parser.add_argument("--limit", type=int, default=200, help="Maximum points to load.")
    spreads_parser.add_argument(
        "--table-rows",
        type=int,
        default=20,
        help="Number of rows to display in the summary table.",
    )
    spreads_parser.add_argument(
        "--chart",
        action="store_true",
        help="Render an ASCII chart for the selected symbol.",
    )
    spreads_parser.add_argument("--chart-width", type=int, default=80, help="Character width for chart output.")
    spreads_parser.add_argument("--chart-height", type=int, default=12, help="Row height for chart output.")
    spreads_parser.set_defaults(func=handle_spreads)

    shelves_parser = subparsers.add_parser("shelves", help="Inspect stored shelf snapshots.")
    shelves_parser.add_argument("--file", help="Shelf file to inspect. Defaults to first configured shelf.")
    shelves_parser.add_argument("--limit", type=int, default=10, help="Number of snapshots to display.")
    shelves_parser.add_argument(
        "--summary",
        action="store_true",
        help="Display a parsed summary of the latest snapshot (if JSON).",
    )
    shelves_parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the full JSON content for the latest snapshot.",
    )
    shelves_parser.set_defaults(func=handle_shelves)

    metrics_parser = subparsers.add_parser("metrics", help="Inspect stored metrics from log ingestion.")
    metrics_parser.add_argument("--metric", required=True, help="Metric name to inspect (e.g. current_qty).")
    metrics_parser.add_argument("--symbol", help="Filter metric by symbol (if applicable).")
    metrics_parser.add_argument("--source", help="Filter metric by source (e.g. trade_stock_e2e, alpaca_cli).")
    metrics_parser.add_argument("--limit", type=int, default=200, help="Maximum records to fetch.")
    metrics_parser.add_argument(
        "--table-rows",
        type=int,
        default=20,
        help="Number of rows to display from the loaded records.",
    )
    metrics_parser.add_argument("--chart", action="store_true", help="Render an ASCII chart for this metric.")
    metrics_parser.add_argument("--chart-width", type=int, default=80, help="Character width for chart output.")
    metrics_parser.add_argument("--chart-height", type=int, default=12, help="Row height for chart output.")
    metrics_parser.add_argument(
        "--show-message",
        action="store_true",
        help="Show the most recent log message associated with the metric.",
    )
    metrics_parser.set_defaults(func=handle_metrics)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
