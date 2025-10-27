#!/usr/bin/env python3
"""
Generate a BTCUSD close-price overlay chart with Kronos and Toto forecasts.

The script loads the last ``window`` bars from ``trainingdata/<symbol>.csv``,
evaluates several Kronos/Toto variants strictly on GPU, and writes a PNG plot
plus a JSON metrics payload under ``testresults/``.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Sequence

import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAININGDATA_ROOT = REPO_ROOT / "trainingdata"
sys.path.insert(0, str(REPO_ROOT))
import test_kronos_vs_toto as kvs


@dataclass(frozen=True)
class ForecastVariant:
    label: str
    model_type: Literal["kronos", "toto"]
    config: kvs.KronosRunConfig | kvs.TotoRunConfig
    env_overrides: Dict[str, Optional[str]]
    description: str = ""


@dataclass
class ForecastRunResult:
    variant: ForecastVariant
    evaluation: kvs.ModelEvaluation


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available; GPU execution is required for this script.")


def load_price_history(symbol: str) -> pd.DataFrame:
    path = TRAININGDATA_ROOT / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path).copy()
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError(f"Dataset {path} must contain 'timestamp' and 'close' columns.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_eval_indices(length: int, window: int) -> List[int]:
    if length <= window:
        raise ValueError(
            f"Window {window} exceeds dataset length {length}; need sufficient history for sequential evaluation."
        )
    start = max(1, length - window)
    return list(range(start, length))


def clone_kronos_config(base: kvs.KronosRunConfig, *, name: str, **overrides: object) -> kvs.KronosRunConfig:
    payload = asdict(base)
    payload.update(overrides)
    payload["name"] = name
    return kvs.KronosRunConfig(**payload)


def clone_toto_config(base: kvs.TotoRunConfig, *, name: str, **overrides: object) -> kvs.TotoRunConfig:
    payload = asdict(base)
    payload.update(overrides)
    payload["name"] = name
    return kvs.TotoRunConfig(**payload)


def build_variants(symbol: str) -> List[ForecastVariant]:
    kronos_cfg, _, _ = kvs._load_best_config_from_store("kronos", symbol)
    if kronos_cfg is None:
        raise RuntimeError(f"No stored Kronos hyperparameters for {symbol}.")

    kronos_variants: List[ForecastVariant] = [
        ForecastVariant(
            label="kronos_best",
            model_type="kronos",
            config=kronos_cfg,
            env_overrides={},
            description="Stored best Kronos configuration.",
        )
    ]

    # Use a higher-sample Kronos sweep configuration for contrast.
    if kvs.KRONOS_SWEEP:
        kronos_variants.append(
            ForecastVariant(
                label="kronos_high_samples",
                model_type="kronos",
                config=clone_kronos_config(
                    kvs.KRONOS_SWEEP[min(5, len(kvs.KRONOS_SWEEP) - 1)],
                    name="kronos_high_samples",
                ),
                env_overrides={},
                description="Representative Kronos sweep entry with larger sample count.",
            )
        )

    toto_cfg, _, _ = kvs._load_best_config_from_store("toto", symbol)
    if toto_cfg is None:
        raise RuntimeError(f"No stored Toto hyperparameters for {symbol}.")

    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    toto_variants: List[ForecastVariant] = [
        ForecastVariant(
            label="toto_best",
            model_type="toto",
            config=toto_cfg,
            env_overrides={},
            description="Stored best Toto configuration without compilation.",
        ),
        ForecastVariant(
            label="toto_compiled_fp32",
            model_type="toto",
            config=clone_toto_config(
                toto_cfg,
                name="toto_compiled_fp32",
                aggregate="median",
                samples_per_batch=max(64, min(256, toto_cfg.samples_per_batch)),
            ),
            env_overrides={
                "TOTO_TORCH_COMPILE": "1",
                "TOTO_TORCH_DTYPE": "float32",
                "TOTO_COMPILE_MODE": "max-autotune",
                "TOTO_COMPILE_BACKEND": "inductor",
            },
            description="torch.compile with FP32 execution.",
        ),
    ]

    if bf16_supported:
        toto_variants.append(
            ForecastVariant(
                label="toto_compiled_bf16",
                model_type="toto",
                config=clone_toto_config(
                    toto_cfg,
                    name="toto_compiled_bf16",
                    aggregate="trimmed_mean_0.10",
                    samples_per_batch=max(64, min(192, toto_cfg.samples_per_batch)),
                ),
                env_overrides={
                    "TOTO_TORCH_COMPILE": "1",
                    "TOTO_TORCH_DTYPE": "bfloat16",
                    "TOTO_COMPILE_MODE": "max-autotune",
                    "TOTO_COMPILE_BACKEND": "inductor",
                },
                description="torch.compile with BF16 execution and trimmed-mean aggregation.",
            )
        )
    else:
        print("[WARN] CUDA BF16 not supported on this device; skipping compiled BF16 Toto variant.")

    return kronos_variants + toto_variants


def _reset_toto_pipeline() -> None:
    pipeline = getattr(kvs, "_toto_pipeline", None)
    if pipeline is not None:
        try:
            pipeline.unload()
        except Exception as exc:  # pragma: no cover - cleanup best effort
            print(f"[WARN] Failed to unload Toto pipeline: {exc}")
    kvs._toto_pipeline = None


@contextmanager
def temporary_environment(overrides: Dict[str, Optional[str]]) -> Iterator[None]:
    originals: Dict[str, Optional[str]] = {}
    try:
        for key, value in overrides.items():
            originals[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_variant(
    variant: ForecastVariant,
    df: pd.DataFrame,
    prices: np.ndarray,
    eval_indices: Sequence[int],
) -> ForecastRunResult:
    print(f"[INFO] Running variant: {variant.label}")
    if variant.model_type == "kronos":
        evaluation = kvs._evaluate_kronos_sequential(
            df,
            eval_indices,
            variant.config,  # type: ignore[arg-type]
            extra_metadata={"variant": variant.label},
        )
        wrapper = kvs._kronos_wrapper or kvs._load_kronos_wrapper()
        device = getattr(wrapper, "_device", "unknown")
        if not str(device).startswith("cuda"):
            raise RuntimeError(f"Kronos variant '{variant.label}' executed on non-CUDA device '{device}'.")
        metadata = dict(evaluation.metadata or {})
        metadata.setdefault("device", device)
        evaluation.metadata = metadata
        return ForecastRunResult(variant=variant, evaluation=evaluation)

    if variant.model_type == "toto":
        with temporary_environment(variant.env_overrides):
            _reset_toto_pipeline()
            try:
                evaluation = kvs._evaluate_toto_sequential(
                    prices,
                    eval_indices,
                    variant.config,  # type: ignore[arg-type]
                    extra_metadata={"variant": variant.label},
                )
                pipeline = kvs._toto_pipeline
                if pipeline is None:
                    pipeline = kvs._load_toto_pipeline()
                device = getattr(pipeline, "device", "unknown")
                if not str(device).startswith("cuda"):
                    raise RuntimeError(f"Toto variant '{variant.label}' executed on non-CUDA device '{device}'.")
                metadata = dict(evaluation.metadata or {})
                metadata.setdefault("device", device)
                evaluation.metadata = metadata
                return ForecastRunResult(variant=variant, evaluation=evaluation)
            finally:
                _reset_toto_pipeline()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    raise ValueError(f"Unsupported model type '{variant.model_type}'.")


def _to_serialisable(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serialisable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(item) for item in value]
    return value


def save_summary(
    symbol: str,
    window: int,
    timestamps: Sequence[pd.Timestamp],
    actual_prices: Sequence[float],
    runs: Sequence[ForecastRunResult],
    output_path: Path,
) -> None:
    payload = {
        "symbol": symbol,
        "window": window,
        "timestamps": [ts.isoformat() for ts in timestamps],
        "actual_close": [float(price) for price in actual_prices],
        "variants": [],
    }
    for run in runs:
        evaluation = run.evaluation
        payload["variants"].append(
            {
                "label": run.variant.label,
                "model_type": run.variant.model_type,
                "description": run.variant.description,
                "config": _to_serialisable(asdict(run.variant.config)),
                "env_overrides": {key: value for key, value in run.variant.env_overrides.items()},
                "price_mae": float(evaluation.price_mae),
                "pct_return_mae": float(evaluation.pct_return_mae),
                "latency_s": float(evaluation.latency_s),
                "predicted_prices": _to_serialisable(evaluation.predicted_prices),
                "predicted_returns": _to_serialisable(evaluation.predicted_returns),
                "metadata": _to_serialisable(evaluation.metadata or {}),
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_overlay(
    timestamps: Sequence[pd.Timestamp],
    actual_prices: Sequence[float],
    runs: Sequence[ForecastRunResult],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timestamps, actual_prices, label="Actual close", color="#111827", linewidth=2.2)

    palette = plt.get_cmap("tab10")
    for idx, run in enumerate(runs):
        evaluation = run.evaluation
        predicted = np.asarray(evaluation.predicted_prices, dtype=np.float64)
        color = palette(idx % palette.N)
        linestyle = "--" if run.variant.model_type == "toto" else "-"
        ax.plot(
            timestamps,
            predicted,
            label=f"{run.variant.label}",
            color=color,
            linewidth=1.8,
            linestyle=linestyle,
        )
        ax.scatter(
            timestamps,
            predicted,
            color=color,
            s=30,
            marker="o" if run.variant.model_type == "kronos" else "s",
            alpha=0.85,
        )

    ax.set_title("BTCUSD Close vs. Kronos/Toto Forecast Variants")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Close Price (USD)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kronos/Toto BTC forecast overlay.")
    parser.add_argument("--symbol", default="BTCUSD", help="Target symbol (default: %(default)s).")
    parser.add_argument("--window", type=int, default=20, help="Number of trailing bars to evaluate (default: %(default)s).")
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "testresults" / "btc_kronos_toto_overlay",
        help="Directory to store artefacts (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    ensure_cuda_available()
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    symbol = args.symbol.upper()
    window = int(args.window)

    print(f"[INFO] Loading dataset for {symbol}")
    df = load_price_history(symbol)
    prices = df["close"].to_numpy(dtype=np.float64)
    eval_indices = build_eval_indices(len(df), window)
    timestamps = pd.to_datetime(df.loc[eval_indices, "timestamp"])
    actual_prices = prices[eval_indices]

    variants = build_variants(symbol)
    runs: List[ForecastRunResult] = []
    for variant in variants:
        run = run_variant(variant, df, prices, eval_indices)
        runs.append(run)
        print(
            f"[INFO] {variant.label}: price_mae={run.evaluation.price_mae:.6f}, "
            f"pct_return_mae={run.evaluation.pct_return_mae:.6f}, latency_s={run.evaluation.latency_s:.2f}"
        )

    plot_path = output_dir / f"{symbol.lower()}_overlay.png"
    print(f"[INFO] Writing overlay plot -> {plot_path}")
    plot_overlay(timestamps, actual_prices, runs, plot_path)

    summary_path = output_dir / f"{symbol.lower()}_overlay_summary.json"
    print(f"[INFO] Writing summary -> {summary_path}")
    save_summary(symbol, window, timestamps, actual_prices, runs, summary_path)

    print("[INFO] Completed Kronos/Toto overlay generation.")


if __name__ == "__main__":
    main()
