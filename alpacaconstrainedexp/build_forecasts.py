from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger
import pandas as pd

from binanceneural.forecasts import ChronosForecastManager, ForecastConfig
from src.hourly_data_utils import resolve_hourly_symbol_path
from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.symbol_utils import is_crypto_symbol
from src.torch_device_utils import require_cuda as require_cuda_device

from .symbols import build_longable_symbols, build_shortable_symbols, normalize_symbols


def _parse_symbols(raw: str | None) -> List[str]:
    if raw is None:
        return []
    symbols = normalize_symbols([token for token in raw.split(",") if token.strip()])
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _resolve_data_root(
    symbol: str,
    *,
    data_root: Optional[Path],
    crypto_root: Optional[Path],
    stock_root: Optional[Path],
) -> Path:
    if data_root is not None:
        candidate = resolve_hourly_symbol_path(symbol, data_root)
        if candidate is not None:
            return candidate.parent
    if is_crypto_symbol(symbol) and crypto_root is not None:
        candidate = resolve_hourly_symbol_path(symbol, crypto_root)
        if candidate is not None:
            return candidate.parent
    if not is_crypto_symbol(symbol) and stock_root is not None:
        candidate = resolve_hourly_symbol_path(symbol, stock_root)
        if candidate is not None:
            return candidate.parent
    fallback_root = Path("trainingdatahourly")
    candidate = resolve_hourly_symbol_path(symbol, fallback_root)
    if candidate is None:
        raise FileNotFoundError(f"Hourly data for {symbol} not found under {fallback_root}")
    return candidate.parent


def _resolve_window(
    *,
    start: Optional[str],
    end: Optional[str],
    lookback_hours: Optional[float],
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if lookback_hours is not None and lookback_hours > 0:
        end_ts = pd.Timestamp(datetime.now(timezone.utc))
        start_ts = end_ts - pd.Timedelta(hours=float(lookback_hours))
        return start_ts, end_ts
    start_ts = pd.to_datetime(start, utc=True, errors="coerce") if start else None
    end_ts = pd.to_datetime(end, utc=True, errors="coerce") if end else None
    if start and start_ts is None:
        raise ValueError(f"Invalid --start timestamp: {start}")
    if end and end_ts is None:
        raise ValueError(f"Invalid --end timestamp: {end}")
    return start_ts, end_ts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forecast caches using a fine-tuned Chronos2 model.")
    parser.add_argument("--symbols", default=None)
    parser.add_argument(
        "--long-stocks",
        default=None,
        help="Comma-separated longable stock symbols (default: NVDA,GOOG,MSFT).",
    )
    parser.add_argument(
        "--short-stocks",
        default=None,
        help="Comma-separated shortable stock symbols (default: YELP,EBAY,TRIP,MTCH,KIND,ANGI,Z,EXPE,BKNG,NWSA,NYT).",
    )
    parser.add_argument(
        "--crypto",
        default=None,
        help="Comma-separated crypto symbols to include (default: BTCUSD,ETHUSD,SOLUSD).",
    )
    parser.add_argument("--finetuned-model", required=True, help="Path to fine-tuned Chronos2 model dir.")
    parser.add_argument("--forecast-cache-root", default="alpacaconstrainedexp/forecast_cache")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--crypto-data-root", default=None)
    parser.add_argument("--stock-data-root", default=None)
    parser.add_argument("--horizons", default="1,24")
    parser.add_argument("--context-hours", type=int, default=24 * 14)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quantiles", default="0.1,0.5,0.9")
    parser.add_argument("--predict-batches-jointly", action="store_true")
    parser.add_argument("--start", default=None, help="ISO timestamp for start of forecast window (UTC).")
    parser.add_argument("--end", default=None, help="ISO timestamp for end of forecast window (UTC).")
    parser.add_argument("--lookback-hours", type=float, default=None)
    args = parser.parse_args()

    require_cuda_device("chronos2 forecast generation", allow_fallback=False)

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        longable = build_longable_symbols(
            crypto_symbols=_parse_symbols(args.crypto) if args.crypto else None,
            stock_symbols=_parse_symbols(args.long_stocks) if args.long_stocks else None,
        )
        shortable = build_shortable_symbols(
            stock_symbols=_parse_symbols(args.short_stocks) if args.short_stocks else None,
        )
        symbols = normalize_symbols(longable + shortable)
        if not symbols:
            raise ValueError("At least one symbol is required.")
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()]
    cache_root = Path(args.forecast_cache_root)

    logger.info("Loading fine-tuned Chronos2 model from %s", args.finetuned_model)
    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id=str(Path(args.finetuned_model)),
        device_map="cuda",
        default_context_length=args.context_hours,
        default_batch_size=args.batch_size,
        quantile_levels=quantiles,
    )

    def _factory():
        return wrapper

    start_ts, end_ts = _resolve_window(
        start=args.start,
        end=args.end,
        lookback_hours=args.lookback_hours,
    )

    for symbol in symbols:
        data_root = _resolve_data_root(
            symbol,
            data_root=Path(args.data_root) if args.data_root else None,
            crypto_root=Path(args.crypto_data_root) if args.crypto_data_root else None,
            stock_root=Path(args.stock_data_root) if args.stock_data_root else None,
        )

        for horizon in horizons:
            horizon_dir = cache_root / f"h{int(horizon)}"
            cfg = ForecastConfig(
                symbol=symbol,
                data_root=data_root,
                context_hours=args.context_hours,
                prediction_horizon_hours=int(horizon),
                quantile_levels=tuple(quantiles),
                batch_size=args.batch_size,
                cache_dir=horizon_dir,
            )
            manager = ChronosForecastManager(cfg, wrapper_factory=_factory)
            if args.predict_batches_jointly:
                manager._predict_kwargs = {"predict_batches_jointly": True}
            logger.info("Generating forecasts for %s horizon=%dh", symbol, horizon)
            manager.ensure_latest(start=start_ts, end=end_ts, cache_only=False)


if __name__ == "__main__":
    main()
