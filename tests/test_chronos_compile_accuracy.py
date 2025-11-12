"""Chronos2 torch.compile regression test using real trainingdata samples."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

try:  # pragma: no cover - optional instrumentation for CUDA graph debugging
    import torch._inductor.config as inductor_config  # type: ignore
except Exception:  # pragma: no cover - torch nightly variations
    inductor_config = None  # type: ignore
else:
    if os.getenv("CHRONOS_INDUCTOR_DEBUG") == "1":
        inductor_config.debug = True

try:  # pragma: no cover - optional helper
    import chronos_compile_config
except Exception:  # pragma: no cover - envs without Chronos extras
    chronos_compile_config = None  # type: ignore

from src.models.chronos2_wrapper import Chronos2OHLCWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chronos_compile_test")

SYMBOLS: Tuple[str, ...] = ("BTCUSD", "ETHUSD")
CONTEXT_LENGTH = 1024
PREDICTION_LENGTH = 32
MAE_TOLERANCE = 5e-3
BASELINE_PATH = Path(__file__).parent / "chronos_mae_baseline.txt"


def _load_symbol_frames(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = Path("trainingdata") / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found for {symbol}: {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError(f"{symbol} CSV missing required columns (timestamp, close)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    required = CONTEXT_LENGTH + PREDICTION_LENGTH
    if len(df) < required:
        raise ValueError(f"{symbol} needs at least {required} rows, found {len(df)}")

    context = df.iloc[-required:-PREDICTION_LENGTH].copy()
    holdout = df.iloc[-PREDICTION_LENGTH:].copy()
    return context, holdout


def _prepare_wrapper(compiled: bool) -> Chronos2OHLCWrapper:
    if compiled and chronos_compile_config is not None:
        chronos_compile_config.apply(verbose=False)
    elif not compiled:
        os.environ["CHRONOS_COMPILE"] = "0"

    return Chronos2OHLCWrapper.from_pretrained(
        model_id="amazon/chronos-2",
        torch_compile=compiled,
        compile_mode="reduce-overhead" if compiled else None,
        compile_backend="inductor" if compiled else None,
        default_context_length=CONTEXT_LENGTH,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )


def _run_inference(symbol: str, compiled: bool) -> Dict[str, float]:
    context, holdout = _load_symbol_frames(symbol)
    wrapper = _prepare_wrapper(compiled)

    start = time.perf_counter()
    batch = wrapper.predict_ohlc(
        context_df=context,
        symbol=symbol,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        evaluation_df=holdout,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0

    median = batch.median
    target_index = pd.to_datetime(holdout["timestamp"], utc=True)
    preds = median.loc[target_index, "close"].to_numpy(dtype=np.float64)
    actual = holdout["close"].to_numpy(dtype=np.float64)
    mae = float(np.mean(np.abs(preds - actual)))

    wrapper.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"mae": mae, "latency_ms": latency_ms}


def _write_baseline(rows: Tuple[Dict[str, float], ...]) -> None:
    with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
        handle.write("# Chronos2 MAE Baseline\n")
        handle.write(f"# Generated: {pd.Timestamp.now()}\n")
        handle.write(f"# PyTorch: {torch.__version__}\n\n")
        handle.write("Symbol,Uncompiled_MAE,Compiled_MAE,Uncompiled_ms,Compiled_ms\n")
        for row in rows:
            handle.write(
                f"{row['symbol']},{row['mae_uncompiled']:.6f},{row['mae_compiled']:.6f},"
                f"{row['latency_uncompiled_ms']:.2f},{row['latency_compiled_ms']:.2f}\n"
            )


def test_chronos_compile_matches_baseline() -> bool:
    logger.info("Chronos2 compile accuracy test (context=%s, horizon=%s)", CONTEXT_LENGTH, PREDICTION_LENGTH)

    summary_rows = []
    for symbol in SYMBOLS:
        logger.info("\n=== %s ===", symbol)
        eager = _run_inference(symbol, compiled=False)
        compiled = _run_inference(symbol, compiled=True)

        diff = abs(eager["mae"] - compiled["mae"])
        logger.info(
            "Uncompiled MAE=%.6f (%.2f ms), compiled MAE=%.6f (%.2f ms), diff=%.6g",
            eager["mae"],
            eager["latency_ms"],
            compiled["mae"],
            compiled["latency_ms"],
            diff,
        )
        assert diff <= MAE_TOLERANCE, f"MAE drift {diff} exceeds tolerance for {symbol}"

        summary_rows.append(
            {
                "symbol": symbol,
                "mae_uncompiled": eager["mae"],
                "mae_compiled": compiled["mae"],
                "latency_uncompiled_ms": eager["latency_ms"],
                "latency_compiled_ms": compiled["latency_ms"],
            }
        )

    _write_baseline(tuple(summary_rows))
    logger.info("\nBaseline written to %s", BASELINE_PATH)
    return True


if __name__ == "__main__":
    try:
        ok = test_chronos_compile_matches_baseline()
        sys.exit(0 if ok else 1)
    except Exception as exc:  # pragma: no cover - manual runs
        logger.exception("Chronos compile test failed: %s", exc)
        sys.exit(1)
