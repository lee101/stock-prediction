#!/usr/bin/env python3
"""
Generate side-by-side Kronos vs. Toto forecast plots using the stored best hyperparameters.

This script is a lightweight wrapper around ``test_kronos_vs_toto`` that:
  * loads the best Kronos/Toto configuration for each requested symbol,
  * runs the sequential evaluation used during hyperparameter selection,
  * writes a comparison plot (actual vs. forecast) to ``testresults/``,
  * emits a JSON summary with the key metrics per symbol.

Example
-------
.. code-block:: bash

    uv run python test_toto_vs_kronos_graphical.py --symbols AAPL,BTCUSD

The command above writes ``PNG`` plots and per-symbol metric JSON files under
``testresults/toto_vs_kronos``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from test_kronos_vs_toto import (  # type: ignore
    FORECAST_HORIZON,
    KronosRunConfig,
    ModelEvaluation,
    TotoRunConfig,
    _evaluate_kronos_sequential,
    _evaluate_toto_sequential,
    _load_best_config_from_store,
    _plot_forecast_comparison,
    _load_toto_pipeline,
)


def _available_symbols() -> List[str]:
    """Return the intersection of symbols with both Kronos and Toto hyperparams."""
    root = Path("hyperparams")
    kronos_root = root / "kronos"
    toto_root = root / "toto"
    if not kronos_root.exists() or not toto_root.exists():
        return []
    kronos_symbols = {path.stem for path in kronos_root.glob("*.json")}
    toto_symbols = {path.stem for path in toto_root.glob("*.json")}
    return sorted(kronos_symbols & toto_symbols)


def _load_dataset(symbol: str, data_path: Optional[Path] = None, *, data_root: Optional[Path] = None) -> pd.DataFrame:
    """Load the historical price series for ``symbol``."""
    if data_path is None:
        repo_root = Path(__file__).resolve().parent
        candidates = [
            repo_root / "trainingdata" / f"{symbol}.csv",
            Path("trainingdata") / f"{symbol}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                data_path = candidate
                break
        if data_path is None and data_root is not None:
            candidate = data_root / f"{symbol}.csv"
            if candidate.exists():
                data_path = candidate
    if data_path is None or not data_path.exists():
        raise FileNotFoundError(f"Dataset for '{symbol}' not found (looked in trainingdata/{symbol}.csv).")

    df = pd.read_csv(data_path).copy()
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError(f"Dataset for {symbol} must include 'timestamp' and 'close' columns.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _build_eval_window(prices: np.ndarray, test_window: int) -> List[int]:
    """Build sequential evaluation indices matching the hyperparameter window."""
    if prices.size < 2:
        raise ValueError("Need at least two price points for sequential evaluation.")
    window = max(1, int(test_window))
    if window >= len(prices):
        window = len(prices) - 1
    start = len(prices) - window
    if start <= 0:
        start = 1
    return list(range(start, len(prices)))


def _compute_actual_returns(series: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Compute step returns aligned with ``indices``."""
    returns: List[float] = []
    prev_price = float(series[indices[0] - 1])
    for idx in indices:
        price = float(series[idx])
        if prev_price == 0.0:
            returns.append(0.0)
        else:
            returns.append((price - prev_price) / prev_price)
        prev_price = price
    return np.asarray(returns, dtype=np.float64)


def _evaluate_symbol(symbol: str, output_dir: Path, *, data_root: Optional[Path] = None) -> Optional[Path]:
    kronos_cfg, kronos_meta, kronos_windows = _load_best_config_from_store("kronos", symbol)
    toto_cfg, toto_meta, toto_windows = _load_best_config_from_store("toto", symbol)

    if kronos_cfg is None and toto_cfg is None:
        print(f"[WARN] No hyperparameters found for {symbol}; skipping.")
        return None

    df = _load_dataset(symbol, data_root=data_root)
    prices = df["close"].to_numpy(dtype=np.float64)
    if prices.size <= FORECAST_HORIZON:
        raise ValueError(f"Dataset for {symbol} must exceed the forecast horizon ({FORECAST_HORIZON}).")

    windows: Dict[str, int] = {}
    for payload in (kronos_windows, toto_windows):
        if payload:
            windows.update({key: int(value) for key, value in payload.items() if isinstance(value, (int, float))})
    test_window = int(windows.get("test_window", 20))
    eval_indices = _build_eval_window(prices, test_window)
    actual_prices = prices[eval_indices]
    actual_returns = _compute_actual_returns(prices, eval_indices)

    kronos_eval: Optional[ModelEvaluation] = None
    if isinstance(kronos_cfg, KronosRunConfig):
        kronos_eval = _evaluate_kronos_sequential(
            df,
            eval_indices,
            kronos_cfg,
            extra_metadata=kronos_meta or None,
        )

    toto_eval: Optional[ModelEvaluation] = None
    if isinstance(toto_cfg, TotoRunConfig):
        _load_toto_pipeline()  # ensure pipeline is initialised once
        toto_eval = _evaluate_toto_sequential(
            prices,
            eval_indices,
            toto_cfg,
            extra_metadata=toto_meta or None,
        )

    timestamps = pd.to_datetime(df["timestamp"].iloc[eval_indices])
    plot_path = _plot_forecast_comparison(
        timestamps,
        actual_prices,
        kronos_eval,
        toto_eval,
        symbol=symbol,
        output_dir=output_dir,
    )

    summary = {
        "symbol": symbol,
        "test_window": test_window,
        "forecast_horizon": FORECAST_HORIZON,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }
    if kronos_eval is not None:
        summary["kronos"] = {
            "config": kronos_eval.config,
            "price_mae": kronos_eval.price_mae,
            "pct_return_mae": kronos_eval.pct_return_mae,
            "latency_s": kronos_eval.latency_s,
        }
    if toto_eval is not None:
        summary["toto"] = {
            "config": toto_eval.config,
            "price_mae": toto_eval.price_mae,
            "pct_return_mae": toto_eval.pct_return_mae,
            "latency_s": toto_eval.latency_s,
        }
    if plot_path:
        summary["plot"] = str(plot_path)

    json_path = output_dir / f"{symbol}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] {symbol}: wrote summary -> {json_path}")
    if plot_path:
        print(f"[INFO] {symbol}: wrote plot -> {plot_path}")
    return plot_path


def _parse_symbols(value: str) -> List[str]:
    items = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one symbol.")
    return items


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Kronos vs Toto forecast plots.")
    parser.add_argument(
        "--symbols",
        type=_parse_symbols,
        help="Comma-separated list of symbols (default: intersection of stored hyperparams).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("testresults") / "toto_vs_kronos",
        help="Directory to write plots and summaries (default: %(default)s).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional directory containing <symbol>.csv data files.",
    )
    args = parser.parse_args(argv)

    symbols = args.symbols or _available_symbols()
    if not symbols:
        print("No symbols requested and no overlapping hyperparameters were found.")
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating symbols: {', '.join(symbols)}")
    print(f"Writing artefacts to: {output_dir}")

    for symbol in symbols:
        try:
            _evaluate_symbol(symbol, output_dir, data_root=args.data_root)
        except Exception as exc:
            print(f"[ERROR] Failed to evaluate {symbol}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
