from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

from .current_symbols import load_current_symbols
from .forecast_cache import ChronosForecastGenerator, ForecastGenerationConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chronos2 next-day forecasts and cache them.")
    parser.add_argument(
        "--data-dir",
        default="trainingdata/train",
        help="Directory containing per-symbol OHLC CSV files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="strategytraining/forecast_cache",
        help="Directory where forecast parquet files will be stored.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Chronos2 context length (number of historical rows) to feed per prediction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for Chronos2 predictions.",
    )
    parser.add_argument(
        "--quantile",
        action="append",
        type=float,
        dest="quantiles",
        help="Quantile level to request (default: 0.1,0.5,0.9). Can be specified multiple times.",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Explicit symbol to forecast. Can be passed multiple times.",
    )
    parser.add_argument(
        "--symbols-from-trade-script",
        action="store_true",
        help="Load the current symbol universe from trade_stock_e2e.py.",
    )
    parser.add_argument(
        "--trade-script-path",
        default="trade_stock_e2e.py",
        help="Path to trade_stock_e2e.py when using --symbols-from-trade-script.",
    )
    parser.add_argument(
        "--start-date",
        help="Optional inclusive start date (YYYY-MM-DD) for generated forecasts.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional inclusive end date (YYYY-MM-DD) for generated forecasts.",
    )
    parser.add_argument(
        "--device-map",
        default="cuda",
        help="Device map for Chronos2 (e.g., 'cuda', 'cpu', 'auto').",
    )
    parser.add_argument(
        "--model-id",
        default="amazon/chronos-2",
        help="Chronos2 model identifier or local path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    symbols: List[str] = []
    if args.symbols:
        symbols.extend(args.symbols)
    if args.symbols_from_trade_script:
        symbols.extend(load_current_symbols(args.trade_script_path))
    unique = sorted({sym.upper() for sym in symbols})
    if not unique:
        raise SystemExit("No symbols provided. Use --symbol or --symbols-from-trade-script.")
    return unique


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    symbols = _resolve_symbols(args)
    quantiles: Sequence[float]
    if args.quantiles:
        quantiles = tuple(sorted(set(args.quantiles)))
    else:
        quantiles = (0.1, 0.5, 0.9)

    config = ForecastGenerationConfig(
        context_length=args.context_length,
        prediction_length=1,
        quantile_levels=quantiles,
        batch_size=args.batch_size,
    )
    wrapper_kwargs = {
        "model_id": args.model_id,
        "device_map": args.device_map,
        "default_context_length": args.context_length,
        "default_batch_size": args.batch_size,
        "quantile_levels": quantiles,
    }
    generator = ChronosForecastGenerator(
        data_dir=Path(args.data_dir),
        cache_dir=Path(args.cache_dir),
        config=config,
        wrapper_kwargs=wrapper_kwargs,
    )
    logger.info("Generating forecasts for %d symbols (context=%d).", len(symbols), args.context_length)
    generator.generate(symbols, start_date=args.start_date, end_date=args.end_date)
    logger.info("Forecast generation completed.")


if __name__ == "__main__":
    main()
