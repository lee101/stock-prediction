#!/usr/bin/env python3
"""
Evaluate the blended stockagentcombined forecaster against the production Toto forecaster.

The script walks forward through the most recent portion of each symbol's training dataset,
computing 1-step-ahead price/return errors for both models. Results are logged per symbol and
aggregated at the end. Inspired by ``test_ourtoto_vs_toto.py`` but adapted for the combined agent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure the combined generator does not silently downshift to "fast" mode.
os.environ.setdefault("FAST_TESTING", "0")

from backtest_test3_inline import (  # type: ignore
    _compute_toto_forecast,
    pre_process_data,
    release_model_resources,
    resolve_toto_params,
)
from hyperparamstore.store import HyperparamStore
from stockagentcombined.forecaster import CombinedForecastGenerator


DEFAULT_DATA_ROOT = Path("trainingdata")
DEFAULT_HYPERPARAM_ROOT = Path("hyperparams")


@dataclass
class SymbolEvaluation:
    symbol: str
    points: int
    combined_price_mae: float
    baseline_price_mae: float
    combined_pct_return_mae: float
    baseline_pct_return_mae: float
    combined_latency_s: float
    baseline_latency_s: float
    price_improved: bool
    return_improved: bool
    skipped: int


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def _list_symbols(data_root: Path, symbols: Optional[Sequence[str]]) -> List[str]:
    if symbols:
        return sorted({symbol.upper(): None for symbol in symbols}.keys())
    discovered = sorted(p.stem.upper() for p in data_root.glob("*.csv") if p.is_file())
    return discovered


def _load_symbol_frame(symbol: str, data_root: Path) -> pd.DataFrame:
    path = data_root / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training data for symbol {symbol} not found at {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Dataset {path} missing 'timestamp' column.")
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Dataset {path} missing required columns: {sorted(missing)}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _prepare_baseline_price_frame(history_cap: pd.DataFrame) -> pd.DataFrame:
    renamed = history_cap.rename(
        columns={
            "timestamp": "Timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    data = pre_process_data(renamed, "Close")
    price = data[["Close", "High", "Low", "Open"]].copy()
    price = price.rename(columns={"Date": "time_idx"})
    price["ds"] = pd.date_range(start="1949-01-01", periods=len(price), freq="D").values
    price["y"] = price["Close"].shift(-1)
    price["trade_weight"] = (price["y"] > 0) * 2 - 1
    price = price.iloc[:-1]
    price["id"] = price.index
    price["unique_id"] = 1
    price = price.dropna()
    return price


def _toto_forecast_next_step(price_frame: pd.DataFrame, last_price: float, params: Dict[str, int]) -> Tuple[float, float]:
    predictions, _, predicted_abs = _compute_toto_forecast(price_frame, last_price, params)
    if predictions.numel() == 0:
        raise RuntimeError("Toto forecast returned no predictions.")
    predicted_pct = float(predictions[-1].item())
    predicted_abs = float(predicted_abs)
    return predicted_abs, predicted_pct


def _evaluate_symbol(
    symbol: str,
    frame: pd.DataFrame,
    generator: CombinedForecastGenerator,
    eval_points: int,
    min_history: int,
    prediction_length: int,
) -> SymbolEvaluation:
    toto_params = resolve_toto_params(symbol)
    price_errors_combined: List[float] = []
    price_errors_baseline: List[float] = []
    return_errors_combined: List[float] = []
    return_errors_baseline: List[float] = []
    latency_combined: List[float] = []
    latency_baseline: List[float] = []

    start_idx = max(min_history, len(frame) - eval_points)
    skipped = 0

    for idx in range(start_idx, len(frame)):
        history = frame.iloc[:idx].copy()
        if history.empty or len(history) < min_history:
            skipped += 1
            continue

        baseline_history = history

        try:
            price_frame = _prepare_baseline_price_frame(baseline_history)
        except Exception:
            skipped += 1
            continue
        if price_frame.empty or len(price_frame) < prediction_length + 1:
            skipped += 1
            continue

        last_price = float(baseline_history["close"].iloc[-1])
        actual_price = float(frame["close"].iloc[idx])
        if last_price == 0.0:
            skipped += 1
            continue

        actual_return = (actual_price - last_price) / last_price

        baseline_start = time.perf_counter()
        try:
            baseline_abs, baseline_pct = _toto_forecast_next_step(price_frame, last_price, toto_params)
        except Exception:
            skipped += 1
            continue
        latency_baseline.append(time.perf_counter() - baseline_start)

        combined_start = time.perf_counter()
        try:
            combined = generator.generate_for_symbol(
                symbol,
                prediction_length=prediction_length,
                historical_frame=history,
            )
        except Exception:
            skipped += 1
            continue
        latency_combined.append(time.perf_counter() - combined_start)

        combined_abs = float(combined.combined.get("close", float("nan")))
        if math.isnan(combined_abs):
            skipped += 1
            continue

        combined_return = (combined_abs - last_price) / last_price

        price_errors_baseline.append(abs(baseline_abs - actual_price))
        price_errors_combined.append(abs(combined_abs - actual_price))
        return_errors_baseline.append(abs(baseline_pct - actual_return))
        return_errors_combined.append(abs(combined_return - actual_return))

    points = len(price_errors_baseline)
    if points == 0:
        return SymbolEvaluation(
            symbol=symbol,
            points=0,
            combined_price_mae=float("nan"),
            baseline_price_mae=float("nan"),
            combined_pct_return_mae=float("nan"),
            baseline_pct_return_mae=float("nan"),
            combined_latency_s=float("nan"),
            baseline_latency_s=float("nan"),
            price_improved=False,
            return_improved=False,
            skipped=skipped,
        )

    combined_price_mae = float(np.mean(price_errors_combined))
    baseline_price_mae = float(np.mean(price_errors_baseline))
    combined_pct_return_mae = float(np.mean(return_errors_combined))
    baseline_pct_return_mae = float(np.mean(return_errors_baseline))
    combined_latency = float(np.mean(latency_combined)) if latency_combined else float("nan")
    baseline_latency = float(np.mean(latency_baseline)) if latency_baseline else float("nan")

    return SymbolEvaluation(
        symbol=symbol,
        points=points,
        combined_price_mae=combined_price_mae,
        baseline_price_mae=baseline_price_mae,
        combined_pct_return_mae=combined_pct_return_mae,
        baseline_pct_return_mae=baseline_pct_return_mae,
        combined_latency_s=combined_latency,
        baseline_latency_s=baseline_latency,
        price_improved=combined_price_mae < baseline_price_mae,
        return_improved=combined_pct_return_mae < baseline_pct_return_mae,
        skipped=skipped,
    )


def _summarize(symbol_results: List[SymbolEvaluation]) -> Dict[str, float]:
    total_points = sum(result.points for result in symbol_results if result.points)
    if total_points == 0:
        return {
            "total_points": 0,
            "combined_price_mae": float("nan"),
            "baseline_price_mae": float("nan"),
            "combined_pct_return_mae": float("nan"),
            "baseline_pct_return_mae": float("nan"),
            "price_improved_symbols": 0,
            "return_improved_symbols": 0,
            "evaluated_symbols": 0,
        }

    def weighted_average(values: Iterable[Tuple[int, float]]) -> float:
        acc = 0.0
        weight = 0
        for count, value in values:
            if not math.isnan(value):
                acc += count * value
                weight += count
        if weight == 0:
            return float("nan")
        return acc / weight

    price_mae_combined = weighted_average((res.points, res.combined_price_mae) for res in symbol_results)
    price_mae_baseline = weighted_average((res.points, res.baseline_price_mae) for res in symbol_results)
    pct_return_mae_combined = weighted_average((res.points, res.combined_pct_return_mae) for res in symbol_results)
    pct_return_mae_baseline = weighted_average((res.points, res.baseline_pct_return_mae) for res in symbol_results)

    return {
        "total_points": total_points,
        "evaluated_symbols": sum(1 for res in symbol_results if res.points),
        "combined_price_mae": price_mae_combined,
        "baseline_price_mae": price_mae_baseline,
        "combined_pct_return_mae": pct_return_mae_combined,
        "baseline_pct_return_mae": pct_return_mae_baseline,
        "price_improved_symbols": sum(res.price_improved for res in symbol_results if res.points),
        "return_improved_symbols": sum(res.return_improved for res in symbol_results if res.points),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to evaluate (default: all trainingdata CSVs).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory for training CSVs.")
    parser.add_argument(
        "--hyperparam-root",
        type=Path,
        default=DEFAULT_HYPERPARAM_ROOT,
        help="Root directory containing hyperparameter JSONs.",
    )
    parser.add_argument("--eval-points", type=int, default=64, help="Number of most-recent points to evaluate.")
    parser.add_argument("--min-history", type=int, default=256, help="Minimum history length required per forecast.")
    parser.add_argument("--prediction-length", type=int, default=1, help="Forecast horizon in steps.")
    parser.add_argument("--json-out", type=Path, help="Optional path to write detailed JSON results.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    data_root = args.data_root
    hyper_root = args.hyperparam_root

    symbols = _list_symbols(data_root, args.symbols)
    if not symbols:
        raise SystemExit("No symbols discovered for evaluation.")

    store = HyperparamStore(hyper_root)
    generator = CombinedForecastGenerator(
        data_root=data_root,
        hyperparam_root=hyper_root,
        prediction_columns=("close",),
        hyperparam_store=store,
    )

    symbol_results: List[SymbolEvaluation] = []

    for symbol in symbols:
        try:
            frame = _load_symbol_frame(symbol, data_root)
        except Exception as exc:
            print(f"[{symbol}] Skipping due to dataset error: {exc}", file=sys.stderr)
            continue

        result = _evaluate_symbol(
            symbol=symbol,
            frame=frame,
            generator=generator,
            eval_points=args.eval_points,
            min_history=args.min_history,
            prediction_length=args.prediction_length,
        )
        symbol_results.append(result)
        status = "improved" if result.price_improved else "worse"
        print(
            f"[{symbol}] points={result.points} combined_price_mae={_format_float(result.combined_price_mae)} "
            f"baseline_price_mae={_format_float(result.baseline_price_mae)} ({status}) "
            f"combined_pct_return_mae={_format_float(result.combined_pct_return_mae)} "
            f"baseline_pct_return_mae={_format_float(result.baseline_pct_return_mae)} "
            f"combined_latency={_format_float(result.combined_latency_s)}s "
            f"baseline_latency={_format_float(result.baseline_latency_s)}s "
            f"skipped={result.skipped}"
        )

    summary = _summarize(symbol_results)
    print("\n=== Aggregate Summary ===")
    print(f"Symbols evaluated: {summary['evaluated_symbols']} (total points: {summary['total_points']})")
    print(
        f"Price MAE -> combined={_format_float(summary['combined_price_mae'])} "
        f"baseline={_format_float(summary['baseline_price_mae'])}"
    )
    print(
        f"Return MAE -> combined={_format_float(summary['combined_pct_return_mae'])} "
        f"baseline={_format_float(summary['baseline_pct_return_mae'])}"
    )
    print(
        f"Improved symbols: price={summary['price_improved_symbols']} "
        f"return={summary['return_improved_symbols']}"
    )

    if args.json_out:
        payload = {
            "summary": summary,
            "symbols": [asdict(result) for result in symbol_results],
            "config": {
                "data_root": str(data_root),
                "hyperparam_root": str(hyper_root),
                "eval_points": args.eval_points,
                "min_history": args.min_history,
                "prediction_length": args.prediction_length,
            },
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))

    release_model_resources()


if __name__ == "__main__":
    main()
