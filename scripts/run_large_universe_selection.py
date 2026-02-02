#!/usr/bin/env python3
"""Rank a large symbol universe by Chronos2 daily forecasts and pick top N.

Examples:
  # Use trainingdata/train symbols, auto date, pick top 5
  python scripts/run_large_universe_selection.py --top-n 5 --date auto

  # Use explicit date and custom symbol list
  python scripts/run_large_universe_selection.py --date 2025-12-20 --symbols-file symbolsofinterest.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marketsimlong.config import DataConfigLong, ForecastConfigLong
from marketsimlong.data import DailyDataLoader, is_crypto_symbol
from marketsimlong.forecaster import Chronos2Forecaster


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


_EXCLUDED_SYMBOLS = {"CORRELATION_MATRIX", "DATA_SUMMARY", "VOLATILITY_METRICS"}


def _clean_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not cleaned:
        return ""
    if cleaned in _EXCLUDED_SYMBOLS:
        return ""
    if not cleaned.replace("-", "").replace("_", "").isalnum():
        return ""
    return cleaned


def _load_symbols_from_dir(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbol directory not found: {path}")
    symbols = sorted({_clean_symbol(p.stem) for p in path.glob("*.csv")})
    return [s for s in symbols if s]


def _load_symbols_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            candidates = data.get("available_symbols") or data.get("symbols") or []
        else:
            candidates = data
        cleaned = [_clean_symbol(str(s)) for s in candidates if str(s).strip()]
        return [s for s in cleaned if s]
    symbols: List[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned = _clean_symbol(stripped.strip("',\" "))
        if cleaned:
            symbols.append(cleaned)
    return symbols


def _filter_symbols_with_history(
    loader: DailyDataLoader,
    symbols: Sequence[str],
    min_history: int,
) -> List[str]:
    filtered = []
    for symbol in symbols:
        frame = loader._data_cache.get(symbol)
        if frame is None or frame.empty:
            continue
        if len(frame) < min_history:
            continue
        filtered.append(symbol)
    return filtered


def _resolve_auto_date(
    loader: DailyDataLoader,
    symbols: Sequence[str],
    min_history: int,
) -> date:
    last_dates: List[date] = []
    for symbol in symbols:
        frame = loader._data_cache.get(symbol)
        if frame is None or frame.empty:
            continue
        if len(frame) < min_history:
            continue
        last_dates.append(frame["date"].max())
    if not last_dates:
        raise RuntimeError("No symbols with sufficient history to determine auto date.")
    return min(last_dates)


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank a large daily symbol universe by Chronos2 forecasts and pick top N.",
    )
    parser.add_argument("--date", type=str, default="auto", help="Target date (YYYY-MM-DD) or 'auto'")
    parser.add_argument("--top-n", type=int, default=5, help="Number of symbols to output")
    parser.add_argument("--metric", type=str, default="predicted_return", help="Ranking metric on SymbolForecast")
    parser.add_argument("--symbols-dir", type=str, default="trainingdata/train", help="Directory of CSVs")
    parser.add_argument("--symbols-file", type=str, default="", help="Optional symbols file (txt/json)")
    parser.add_argument("--data-root", type=str, default="trainingdata", help="Data root for DailyDataLoader")
    parser.add_argument("--context-length", type=int, default=512, help="Context length (days)")
    parser.add_argument("--prediction-length", type=int, default=1, help="Prediction length (days)")
    parser.add_argument("--batch-size", type=int, default=128, help="Chronos2 batch size")
    parser.add_argument("--chunk-size", type=int, default=200, help="Cross-learning chunk size")
    parser.add_argument("--min-history", type=int, default=60, help="Minimum history rows per symbol")
    parser.add_argument("--max-symbols", type=int, default=0, help="Limit number of symbols (0 = no limit)")
    parser.add_argument("--device", type=str, default="cuda", help="Device map for Chronos2 (cuda/cpu)")
    parser.add_argument("--output-csv", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--output-json", type=str, default="", help="Optional JSON output path")
    parser.add_argument("--log-gpu-mem", action="store_true", help="Log peak GPU memory for forecasting")
    args = parser.parse_args()

    symbols: List[str]
    if args.symbols_file:
        symbols = _load_symbols_from_file(Path(args.symbols_file))
    else:
        symbols = _load_symbols_from_dir(Path(args.symbols_dir))

    if not symbols:
        raise RuntimeError("No symbols found for ranking.")

    # Split into stock/crypto for data config
    stock_symbols = tuple(sym for sym in symbols if not is_crypto_symbol(sym))
    crypto_symbols = tuple(sym for sym in symbols if is_crypto_symbol(sym))

    data_config = DataConfigLong(
        stock_symbols=stock_symbols,
        crypto_symbols=crypto_symbols,
        data_root=Path(args.data_root),
        context_days=args.context_length,
    )
    loader = DailyDataLoader(data_config)
    loader.load_all_symbols()

    # Filter to symbols with enough history
    symbols_with_history = _filter_symbols_with_history(loader, symbols, args.min_history)
    if not symbols_with_history:
        raise RuntimeError("No symbols meet the minimum history requirement.")

    if args.max_symbols and args.max_symbols > 0:
        symbols_with_history = symbols_with_history[: args.max_symbols]

    if args.date.lower() == "auto":
        target_date = _resolve_auto_date(loader, symbols_with_history, args.min_history)
    else:
        target_date = _parse_date(args.date)

    # Filter tradable symbols for that date
    tradable = loader.get_tradable_symbols_on_date(target_date)
    tradable = [sym for sym in symbols_with_history if sym in tradable]
    if not tradable:
        raise RuntimeError(f"No tradable symbols found on {target_date}.")

    forecast_config = ForecastConfigLong(
        model_id="amazon/chronos-2",
        device_map=args.device,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=args.batch_size,
        use_multivariate=True,
        use_cross_learning=True,
        cross_learning_min_batch=2,
        cross_learning_group_by_asset_type=True,
        cross_learning_chunk_size=args.chunk_size,
    )

    forecaster = Chronos2Forecaster(loader, forecast_config)
    if args.log_gpu_mem:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    forecasts = forecaster.forecast_all_symbols(target_date, tradable)

    if args.log_gpu_mem:
        try:
            import torch

            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_gb = peak_bytes / (1024 ** 3)
                print(f"\nPeak GPU memory allocated: {peak_gb:.2f} GB")
        except Exception:
            pass
    ranked = forecasts.get_ranked_symbols(metric=args.metric, ascending=False)
    top_ranked = ranked[: args.top_n]

    rows = []
    for symbol, score in top_ranked:
        fc = forecasts.forecasts[symbol]
        range_value = fc.predicted_high - fc.predicted_low
        range_pct = range_value / fc.current_close if fc.current_close else 0.0
        rows.append(
            {
                "symbol": symbol,
                "metric": args.metric,
                "score": float(score),
                "current_close": fc.current_close,
                "predicted_close": fc.predicted_close,
                "predicted_high": fc.predicted_high,
                "predicted_low": fc.predicted_low,
                "predicted_close_p10": fc.predicted_close_p10,
                "predicted_close_p90": fc.predicted_close_p90,
                "expected_range": range_value,
                "expected_range_pct": range_pct,
            }
        )

    print()
    print(f"Target date: {target_date}")
    print(f"Universe size: {len(tradable)} (history-filtered)")
    print(f"Ranking metric: {args.metric}")
    print()
    for idx, row in enumerate(rows, 1):
        print(
            f"{idx:2d}. {row['symbol']:8} "
            f"score={_format_float(row['score'], 4)} "
            f"close={_format_float(row['current_close'], 2)} "
            f"pred_close={_format_float(row['predicted_close'], 2)} "
            f"high={_format_float(row['predicted_high'], 2)} "
            f"low={_format_float(row['predicted_low'], 2)} "
            f"range%={_format_float(row['expected_range_pct'] * 100, 2)}"
        )

    if args.output_csv:
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nWrote CSV: {args.output_csv}")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(rows, indent=2))
        print(f"Wrote JSON: {args.output_json}")

    forecaster.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
