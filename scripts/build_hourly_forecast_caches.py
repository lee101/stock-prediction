#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binanceneural.forecasts import ChronosForecastManager, ForecastConfig
from src.chronos2_params import resolve_chronos2_params
from src.forecast_cache_metrics import ForecastMAE, compute_forecast_cache_mae_for_paths
from src.hourly_data_utils import discover_hourly_symbols, resolve_hourly_symbol_path
from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.torch_device_utils import require_cuda as require_cuda_device


DEFAULT_SYMBOLS = "BTCFDUSD,ETHFDUSD,SOLFDUSD,BNBFDUSD"


def _parse_csv_list(raw: str, *, cast) -> List[Any]:
    values: List[Any] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(cast(token))
    return values


def _parse_symbols(raw: Optional[str], *, data_root: Path) -> List[str]:
    if raw:
        return [tok.strip().upper() for tok in str(raw).split(",") if tok.strip()]
    return discover_hourly_symbols(Path(data_root))


def _resolve_window(
    *,
    start: Optional[str],
    end: Optional[str],
    lookback_hours: Optional[float],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
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


def _forecast_parquet_path(cache_dir: Path, symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / f"{safe.upper()}.parquet"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build hourly Chronos2 forecast caches (h1/h4/h24) using per-symbol hyperparams/chronos2/hourly.",
    )
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS, help="Comma-separated symbols to process.")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourlybinance"))
    parser.add_argument("--forecast-cache-root", type=Path, default=Path("binancecrosslearning/forecast_cache_fdusd_pslora"))
    parser.add_argument("--horizons", default="1,4,24", help="Comma-separated forecast horizons (hours).")
    parser.add_argument("--quantiles", default=None, help="Override quantiles (comma-separated).")
    parser.add_argument("--context-hours", type=int, default=None, help="Override context length in hours.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--start", default=None, help="ISO timestamp for start of forecast window (UTC).")
    parser.add_argument("--end", default=None, help="ISO timestamp for end of forecast window (UTC).")
    parser.add_argument("--lookback-hours", type=float, default=5000.0)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-symlink-data", action="store_true")
    parser.add_argument("--no-compute-mae", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON path to write MAE summary.")
    args = parser.parse_args(argv)

    require_cuda_device("chronos2 hourly forecast caching", allow_fallback=False)

    symbols = _parse_symbols(args.symbols, data_root=args.data_root)
    if not symbols:
        raise SystemExit("No symbols provided.")

    horizons = [int(x) for x in _parse_csv_list(args.horizons, cast=int)]
    if not horizons:
        raise SystemExit("No horizons provided.")

    start_ts, end_ts = _resolve_window(start=args.start, end=args.end, lookback_hours=args.lookback_hours)

    cache_root = Path(args.forecast_cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, Any]] = []

    for symbol in symbols:
        symbol = symbol.upper().strip()

        csv_path = resolve_hourly_symbol_path(symbol, Path(args.data_root))
        if csv_path is None:
            raise FileNotFoundError(f"Hourly data for {symbol} not found under {args.data_root}")
        if args.skip_symlink_data and csv_path.is_symlink():
            logger.warning("Skipping {} (data path is a symlink: {})", symbol, csv_path)
            continue
        data_root = csv_path.parent

        params = resolve_chronos2_params(symbol, frequency="hourly")
        model_id = str(params.get("model_id") or "amazon/chronos-2")
        device_map = str(params.get("device_map") or "cuda")

        ctx_hours = int(args.context_hours) if args.context_hours is not None else int(params.get("context_length") or 512)
        batch_size = int(args.batch_size) if args.batch_size is not None else int(params.get("batch_size") or 128)

        if args.quantiles:
            quantiles = tuple(float(x) for x in _parse_csv_list(args.quantiles, cast=float))
        else:
            quantiles = tuple(float(x) for x in (params.get("quantile_levels") or (0.1, 0.5, 0.9)))
        predict_kwargs = dict(params.get("predict_kwargs") or {})

        logger.info(
            "Loading Chronos2 model for {}: model_id={} ctx={} batch={} horizons={}",
            symbol,
            model_id,
            ctx_hours,
            batch_size,
            horizons,
        )

        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id=model_id,
            device_map=device_map,
            default_context_length=ctx_hours,
            default_batch_size=batch_size,
            quantile_levels=quantiles,
        )

        def _factory():
            return wrapper

        for horizon in horizons:
            horizon_dir = cache_root / f"h{int(horizon)}"
            cfg = ForecastConfig(
                symbol=symbol,
                data_root=data_root,
                context_hours=ctx_hours,
                prediction_horizon_hours=int(horizon),
                quantile_levels=quantiles,
                batch_size=batch_size,
                cache_dir=horizon_dir,
            )
            manager = ChronosForecastManager(cfg, wrapper_factory=_factory)
            manager._predict_kwargs = predict_kwargs
            logger.info("Generating forecasts for {} horizon={}h", symbol, horizon)
            manager.ensure_latest(start=start_ts, end=end_ts, cache_only=False, force_rebuild=args.force_rebuild)

            if args.no_compute_mae:
                continue

            try:
                mae = compute_forecast_cache_mae_for_paths(
                    symbol=symbol,
                    horizon_hours=int(horizon),
                    history_csv=csv_path,
                    forecast_parquet=_forecast_parquet_path(horizon_dir, symbol),
                )
            except Exception as exc:
                logger.warning("MAE compute failed for {} h{}: {}", symbol, horizon, exc)
                continue

            payload = {
                "symbol": mae.symbol,
                "horizon_hours": mae.horizon_hours,
                "count": mae.count,
                "mae": mae.mae,
                "mae_percent": mae.mae_percent,
                "start_timestamp": mae.start_timestamp,
                "end_timestamp": mae.end_timestamp,
                "model_id": model_id,
                "context_hours": ctx_hours,
                "batch_size": batch_size,
                "cache_dir": str(horizon_dir),
            }
            all_metrics.append(payload)
            logger.info(
                "MAE% {} h{}: {:.4f} (count={}, window={}→{})",
                symbol,
                horizon,
                mae.mae_percent,
                mae.count,
                mae.start_timestamp,
                mae.end_timestamp,
            )

    if args.no_compute_mae:
        return 0

    out_path = args.output_json
    if out_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out_path = cache_root / f"mae_summary_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"metrics": all_metrics}, indent=2, default=_json_default) + "\n")
    logger.info("Wrote MAE summary: {}", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
