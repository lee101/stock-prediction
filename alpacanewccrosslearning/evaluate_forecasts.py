from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .data import load_symbol_frame


def _parse_symbols(raw: str) -> List[str]:
    symbols = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _load_forecasts(cache_root: Path, symbol: str, horizon: int) -> pd.DataFrame:
    path = cache_root / f"h{int(horizon)}" / f"{symbol.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _maybe_filter_window(frame: pd.DataFrame, *, eval_days: Optional[float], eval_hours: Optional[float]) -> pd.DataFrame:
    if frame.empty:
        return frame
    hours = max(0.0, float(eval_days or 0) * 24.0, float(eval_hours or 0))
    if hours <= 0:
        return frame
    ts_end = pd.to_datetime(frame["timestamp"], utc=True).max()
    ts_start = ts_end - pd.Timedelta(hours=hours)
    return frame[pd.to_datetime(frame["timestamp"], utc=True) >= ts_start].reset_index(drop=True)


def _mae(a: Iterable[float], b: Iterable[float]) -> float:
    a_arr = np.asarray(list(a), dtype=np.float64)
    b_arr = np.asarray(list(b), dtype=np.float64)
    if a_arr.size == 0 or b_arr.size == 0:
        return float("nan")
    if a_arr.size != b_arr.size:
        raise ValueError("MAE inputs must be the same length.")
    return float(np.mean(np.abs(a_arr - b_arr)))


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.size == 0:
        return float("nan")
    denom = np.where(actual == 0.0, np.nan, np.abs(actual))
    frac = np.abs(predicted - actual) / denom
    return float(np.nanmean(frac))


def _evaluate_symbol(
    *,
    symbol: str,
    horizons: Iterable[int],
    cache_root: Path,
    data_root: Optional[Path],
    crypto_root: Optional[Path],
    stock_root: Optional[Path],
    eval_days: Optional[float],
    eval_hours: Optional[float],
) -> List[Dict[str, object]]:
    price = load_symbol_frame(
        symbol,
        data_root=data_root,
        crypto_root=crypto_root,
        stock_root=stock_root,
    )
    price = price[["timestamp", "close", "high", "low"]].copy()
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True, errors="coerce")
    price = price.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    rows: List[Dict[str, object]] = []
    for horizon in horizons:
        forecast = _load_forecasts(cache_root, symbol, horizon)
        if forecast.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "horizon": int(horizon),
                    "n": 0,
                    "mae_close": float("nan"),
                    "mape_close": float("nan"),
                    "mae_high": float("nan"),
                    "mae_low": float("nan"),
                    "notes": "missing_forecasts",
                }
            )
            continue

        merged = price.merge(
            forecast[
                [
                    "timestamp",
                    "predicted_close_p50",
                    "predicted_high_p50",
                    "predicted_low_p50",
                ]
            ],
            on="timestamp",
            how="inner",
        )
        merged = _maybe_filter_window(merged, eval_days=eval_days, eval_hours=eval_hours)
        merged = merged.dropna(
            subset=[
                "close",
                "high",
                "low",
                "predicted_close_p50",
                "predicted_high_p50",
                "predicted_low_p50",
            ]
        )

        actual_close = merged["close"].astype(float).to_numpy()
        pred_close = merged["predicted_close_p50"].astype(float).to_numpy()
        actual_high = merged["high"].astype(float).to_numpy()
        pred_high = merged["predicted_high_p50"].astype(float).to_numpy()
        actual_low = merged["low"].astype(float).to_numpy()
        pred_low = merged["predicted_low_p50"].astype(float).to_numpy()

        rows.append(
            {
                "symbol": symbol,
                "horizon": int(horizon),
                "n": int(len(merged)),
                "mae_close": _mae(actual_close, pred_close),
                "mape_close": _mape(actual_close, pred_close),
                "mae_high": _mae(actual_high, pred_high),
                "mae_low": _mae(actual_low, pred_low),
                "notes": "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forecast MAE for a forecast cache root.")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--forecast-cache-root", required=True)
    parser.add_argument("--horizons", default="1,24")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--crypto-data-root", default=None)
    parser.add_argument("--stock-data-root", default=None)
    parser.add_argument("--eval-days", type=float, default=None)
    parser.add_argument("--eval-hours", type=float, default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    cache_root = Path(args.forecast_cache_root)
    data_root = Path(args.data_root) if args.data_root else None
    crypto_root = Path(args.crypto_data_root) if args.crypto_data_root else None
    stock_root = Path(args.stock_data_root) if args.stock_data_root else None

    all_rows: List[Dict[str, object]] = []
    for symbol in symbols:
        all_rows.extend(
            _evaluate_symbol(
                symbol=symbol,
                horizons=horizons,
                cache_root=cache_root,
                data_root=data_root,
                crypto_root=crypto_root,
                stock_root=stock_root,
                eval_days=args.eval_days,
                eval_hours=args.eval_hours,
            )
        )

    out = pd.DataFrame(all_rows)
    if out.empty:
        raise RuntimeError("No evaluation rows produced.")
    out = out.sort_values(["symbol", "horizon"]).reset_index(drop=True)

    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 200):
        print(out)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

