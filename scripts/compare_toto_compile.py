#!/usr/bin/env python3
"""
Utility script to sanity-check Toto predictions before and after torch.compile.

Runs the Toto pipeline twice (compiled vs. eager) on a realistic stock series
and reports the delta in absolute percentage error for the final step forecast.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.toto_wrapper import TotoPipeline


def load_series(data_path: Path) -> np.ndarray:
    df = pd.read_csv(data_path)
    if "Close" not in df.columns:
        raise ValueError(f"'Close' column missing in {data_path}")
    return df["Close"].to_numpy(dtype=np.float32)


def evaluate_pipeline(
    context: np.ndarray,
    target: float,
    *,
    compile_model: bool,
    device: str,
) -> dict[str, float]:
    torch.manual_seed(42)
    pipeline = TotoPipeline.from_pretrained(
        "Datadog/Toto-Open-Base-1.0",
        device_map=device,
        compile_model=compile_model,
    )

    forecast = pipeline.predict(
        context=context,
        prediction_length=1,
        num_samples=2048,
    )[0].numpy()

    mean_prediction = float(np.mean(forecast))
    abs_error = abs(mean_prediction - target)
    ape = abs_error / abs(target) if not math.isclose(target, 0.0) else abs_error

    return {
        "mean_prediction": mean_prediction,
        "absolute_error": abs_error,
        "absolute_percentage_error": ape * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("/home/lee/code/stock/data/2023-07-08 01:30:11/AAPL-2023-07-08.csv"),
        help="CSV file with a Close column to evaluate against.",
    )
    parser.add_argument("--device", default="cuda", help="Device identifier (e.g. cuda, cuda:0, cpu).")
    args = parser.parse_args()

    series = load_series(args.data_file)
    if series.size < 2:
        raise ValueError("Need at least two points to form context and target.")

    context, target = series[:-1], float(series[-1])

    eager_metrics = evaluate_pipeline(context, target, compile_model=False, device=args.device)
    compiled_metrics = evaluate_pipeline(context, target, compile_model=True, device=args.device)

    delta = compiled_metrics["absolute_percentage_error"] - eager_metrics["absolute_percentage_error"]

    print("Eager metrics:", eager_metrics)
    print("Compiled metrics:", compiled_metrics)
    print(f"∆ absolute percentage error: {delta:.4f} pp")

    tolerance = 0.25  # percentage points
    if abs(delta) > tolerance:
        print(
            "WARNING: compile introduced a noticeable drift. "
            "Consider re-running with a newer torch nightly or filing a bug."
        )


if __name__ == "__main__":
    main()
