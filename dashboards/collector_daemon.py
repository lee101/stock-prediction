from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

if __name__ == "__main__" and __package__ is None:  # pragma: no cover - runtime convenience
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dashboards.collectors import CollectionStats, collect_log_metrics, collect_shelf_snapshots, collect_spreads
    from dashboards.config import DashboardConfig, load_config
    from dashboards.db import DashboardDatabase
    from dashboards.spread_fetcher import SpreadFetcher
else:
    from .collectors import CollectionStats, collect_log_metrics, collect_shelf_snapshots, collect_spreads
    from .config import DashboardConfig, load_config
    from .db import DashboardDatabase
    from .spread_fetcher import SpreadFetcher


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def _apply_overrides(config: DashboardConfig, args: argparse.Namespace) -> DashboardConfig:
    if args.interval:
        config.collection_interval_seconds = int(args.interval)
    if args.shelf_files:
        config.shelf_files = [Path(item).expanduser().resolve() for item in args.shelf_files]
    if args.symbols:
        config.spread_symbols = [symbol.upper() for symbol in args.symbols]
    return config


def _run_iteration(
    config: DashboardConfig,
    db: DashboardDatabase,
    fetcher: SpreadFetcher,
) -> CollectionStats:
    iteration_stats = CollectionStats()
    iteration_stats += collect_shelf_snapshots(config, db)
    iteration_stats += collect_spreads(config, db, fetcher)
    iteration_stats += collect_log_metrics(config, db)
    return iteration_stats


def _sleep_until_next(start_time: float, interval: int) -> None:
    elapsed = time.time() - start_time
    sleep_for = max(0.0, interval - elapsed)
    if sleep_for > 0:
        time.sleep(sleep_for)


def run_daemon(args: argparse.Namespace) -> None:
    _setup_logging(args.log_level)
    config = load_config()
    config = _apply_overrides(config, args)

    logging.getLogger(__name__).info(
        "Dashboards collector starting; interval=%ss shelves=%s symbols=%s logs=%s",
        config.collection_interval_seconds,
        [str(path) for path in config.shelf_files],
        config.spread_symbols,
        {name: str(path) for name, path in config.log_files.items()},
    )

    fetcher = SpreadFetcher()
    with DashboardDatabase(config) as db:
        iteration = 0
        while True:
            iteration += 1
            started = time.time()
            stats = _run_iteration(config, db, fetcher)
            logging.getLogger(__name__).info(
                "Iteration %d completed: %d shelf snapshots, %d spread observations, %d metrics",
                iteration,
                stats.shelf_snapshots,
                stats.spread_observations,
                stats.metrics,
            )
            if args.once:
                break
            _sleep_until_next(started, config.collection_interval_seconds)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect vanity metrics and spreads into SQLite.")
    parser.add_argument("--interval", type=int, help="Polling interval in seconds (overrides config).")
    parser.add_argument("--once", action="store_true", help="Run a single collection pass and exit.")
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        help="Symbol to track (repeat for multiple). Overrides config.",
    )
    parser.add_argument(
        "--shelf",
        dest="shelf_files",
        action="append",
        help="Shelf file path to snapshot. Overrides config.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    try:
        run_daemon(args)
    except KeyboardInterrupt:  # pragma: no cover - redundant safety net
        logging.getLogger(__name__).info("Collector interrupted by user")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
