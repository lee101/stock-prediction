#!/usr/bin/env python3
"""
Compare the public Toto baseline, its calibrated variant, and an optional
fine-tuned checkpoint using identical evaluation settings.

Outputs price / return MAE & RMSE statistics plus an optional JSON report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.toto_aggregation import aggregate_quantile_plus_std
from src.models.toto_wrapper import TotoPipeline, Toto

DEFAULT_DATA_PATH = Path("trainingdata") / "BTCUSD.csv"
DEFAULT_CALIBRATION_FILE = Path("tototraining") / "artifacts" / "calibrated_toto.json"
DEFAULT_CHECKPOINT_DIR = Path("tototraining") / "checkpoints" / "gpu_run"

BASELINE_MODEL_ID = "Datadog/Toto-Open-Base-1.0"
DEFAULT_EVAL_POINTS = 64
DEFAULT_NUM_SAMPLES = 2048
DEFAULT_SAMPLES_PER_BATCH = 256
DEFAULT_QUANTILE = 0.15
DEFAULT_STD_SCALE = 0.15
MIN_CONTEXT = 192


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset at {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError("Dataset must include 'timestamp' and 'close' columns.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_calibration(path: Path) -> Optional[Tuple[float, float]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return float(payload.get("scale", 1.0)), float(payload.get("bias", 0.0))


def _load_checkpoint_config(checkpoint_path: Path) -> Tuple[Dict, Dict[str, torch.Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise KeyError("Checkpoint missing serialized TrainerConfig ('config').")
    state_dict = checkpoint["model_state_dict"]
    return config, state_dict


def _build_pipeline_from_checkpoint(
    checkpoint_path: Path,
    device: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    max_oom_retries: int = 2,
    min_samples_per_batch: int = 32,
    min_num_samples: int = 256,
) -> TotoPipeline:
    config, state_dict = _load_checkpoint_config(checkpoint_path)
    pretrained_model_id = config.get("pretrained_model_id") or BASELINE_MODEL_ID
    base_model = Toto.from_pretrained(pretrained_model_id, map_location="cpu")
    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing parameters when loading checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected parameters in checkpoint: {unexpected}")
    return TotoPipeline(
        model=base_model,
        device=device,
        torch_dtype=torch_dtype,
        max_oom_retries=max_oom_retries,
        min_samples_per_batch=min_samples_per_batch,
        min_num_samples=min_num_samples,
    )


def _collect_predictions(
    pipeline: TotoPipeline,
    prices: np.ndarray,
    eval_points: int,
    *,
    num_samples: int,
    samples_per_batch: int,
    quantile: float,
    std_scale: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    preds: list[float] = []
    actuals: list[float] = []
    start = max(MIN_CONTEXT, len(prices) - eval_points)

    patch_size = getattr(getattr(pipeline, "model", None), "patch_size", None)
    if patch_size is None:
        patch_size = getattr(getattr(getattr(pipeline, "model", None), "model", None), "patch_embed", None)
        patch_size = getattr(patch_size, "patch_size", 1)
    patch_size = int(patch_size or 1)

    first_idx: Optional[int] = None
    for idx in range(start, len(prices)):
        context = prices[:idx].astype(np.float32)
        if patch_size > 1 and context.shape[0] >= patch_size:
            remainder = context.shape[0] % patch_size
            if remainder:
                context = context[remainder:]
        if context.shape[0] < patch_size:
            continue

        forecast = pipeline.predict(
            context=context,
            prediction_length=1,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
        )
        samples = forecast[0].samples if hasattr(forecast[0], "samples") else forecast[0]
        aggregated = aggregate_quantile_plus_std(samples, quantile=quantile, std_scale=std_scale)
        preds.append(float(np.atleast_1d(aggregated)[0]))
        actuals.append(float(prices[idx]))
        if first_idx is None:
            first_idx = idx

    if first_idx is None:
        raise RuntimeError("No evaluation points collected; consider reducing --eval-points.")

    prev_index = max(start - 1, first_idx - 1)
    prev_price = float(prices[prev_index])
    return np.asarray(preds, dtype=np.float64), np.asarray(actuals, dtype=np.float64), prev_price


def _compute_return_metrics(preds: np.ndarray, actuals: np.ndarray, prev_price: float) -> Tuple[float, float]:
    prev = prev_price
    abs_errors = []
    sq_errors = []
    eps = 1e-8
    for pred, actual in zip(preds, actuals):
        denom = prev if abs(prev) > eps else (eps if prev >= 0 else -eps)
        pred_r = (pred - prev) / denom
        actual_r = (actual - prev) / denom
        diff = pred_r - actual_r
        abs_errors.append(abs(diff))
        sq_errors.append(diff * diff)
        prev = actual
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    return mae, rmse


def _summarise(preds: np.ndarray, actuals: np.ndarray, prev_price: float) -> Dict[str, float]:
    errors = actuals - preds
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    return_mae, return_rmse = _compute_return_metrics(preds, actuals, prev_price)
    return {
        "price_mae": mae,
        "price_mse": mse,
        "price_rmse": rmse,
        "return_mae": return_mae,
        "return_rmse": return_rmse,
    }


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return choice


def _resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Toto baseline, calibrated baseline, and retrained checkpoints.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV with timestamp/close columns")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_FILE, help="Calibration JSON (scale/bias)")
    parser.add_argument("--checkpoint", type=Path, help="Optional fine-tuned Toto checkpoint (.pt)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16", None], default=None)
    parser.add_argument("--eval-points", type=int, default=DEFAULT_EVAL_POINTS)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--samples-per-batch", type=int, default=DEFAULT_SAMPLES_PER_BATCH)
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--std-scale", type=float, default=DEFAULT_STD_SCALE)
    parser.add_argument("--max-oom-retries", type=int, default=2)
    parser.add_argument("--min-samples-per-batch", type=int, default=32)
    parser.add_argument("--min-num-samples", type=int, default=256)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--skip-calibration", action="store_true", help="Ignore calibration even if file exists")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Directory for checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    torch_dtype = _resolve_dtype(args.torch_dtype)

    df = _load_dataset(args.data)
    prices = df["close"].to_numpy(dtype=np.float64)

    print("Loading Toto baseline…")
    baseline_pipeline = TotoPipeline.from_pretrained(
        model_id=BASELINE_MODEL_ID,
        device_map=device,
        torch_dtype=torch_dtype,
        max_oom_retries=args.max_oom_retries,
        min_samples_per_batch=args.min_samples_per_batch,
        min_num_samples=args.min_num_samples,
    )
    base_preds, actuals, prev_price = _collect_predictions(
        baseline_pipeline,
        prices,
        args.eval_points,
        num_samples=args.num_samples,
        samples_per_batch=args.samples_per_batch,
        quantile=args.quantile,
        std_scale=args.std_scale,
    )
    base_metrics = _summarise(base_preds, actuals, prev_price)
    del baseline_pipeline
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    calibration = None if args.skip_calibration else _load_calibration(args.calibration)
    if calibration is not None:
        scale, bias = calibration
        calib_preds = scale * base_preds + bias
        calib_metrics = _summarise(calib_preds, actuals, prev_price)
    else:
        calib_metrics = None

    retrained_metrics = None
    retrained_checkpoint = args.checkpoint
    if retrained_checkpoint is None:
        best_dir = args.checkpoint_dir / "best"
        if best_dir.exists():
            ranked = sorted(best_dir.glob("rank*_val*.pt"))
            if ranked:
                retrained_checkpoint = ranked[0]
        elif (args.checkpoint_dir / "latest.pt").exists():
            retrained_checkpoint = args.checkpoint_dir / "latest.pt"

    if retrained_checkpoint is not None and retrained_checkpoint.exists():
        print(f"Loading retrained checkpoint: {retrained_checkpoint}")
        retrained_pipeline = _build_pipeline_from_checkpoint(
            retrained_checkpoint,
            device=device,
            torch_dtype=torch_dtype,
            max_oom_retries=args.max_oom_retries,
            min_samples_per_batch=args.min_samples_per_batch,
            min_num_samples=args.min_num_samples,
        )
        retrained_preds, _, _ = _collect_predictions(
            retrained_pipeline,
            prices,
            args.eval_points,
            num_samples=args.num_samples,
            samples_per_batch=args.samples_per_batch,
            quantile=args.quantile,
            std_scale=args.std_scale,
        )
        retrained_metrics = _summarise(retrained_preds, actuals, prev_price)
        del retrained_pipeline
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        if retrained_checkpoint is not None:
            print(f"Warning: checkpoint {retrained_checkpoint} not found; skipping retrained comparison.")
        else:
            print("No retrained checkpoint provided or discovered; skipping retrained comparison.")

    def _format(metrics: Dict[str, float]) -> str:
        return (
            f"price MAE={metrics['price_mae']:.6f}, "
            f"price RMSE={metrics['price_rmse']:.6f}, "
            f"return MAE={metrics['return_mae']:.6f}, "
            f"return RMSE={metrics['return_rmse']:.6f}"
        )

    print("\n=== Toto Model Comparison (horizon=1) ===")
    print(f"Evaluation points: {len(actuals)} (prev close = {prev_price:.2f})")
    print(f"Baseline ({BASELINE_MODEL_ID}): {_format(base_metrics)}")

    if calib_metrics is not None:
        print(
            f"Calibrated (scale={scale:.6f}, bias={bias:.6f}): {_format(calib_metrics)} "
            f"ΔpriceMAE={calib_metrics['price_mae'] - base_metrics['price_mae']:+.6f}"
        )

    if retrained_metrics is not None:
        print(
            f"Retrained ({retrained_checkpoint.name}): {_format(retrained_metrics)} "
            f"ΔpriceMAE={retrained_metrics['price_mae'] - base_metrics['price_mae']:+.6f}"
        )

    summary = {
        "data_path": str(args.data),
        "device": device,
        "torch_dtype": args.torch_dtype,
        "eval_points": args.eval_points,
        "num_samples": args.num_samples,
        "samples_per_batch": args.samples_per_batch,
        "quantile": args.quantile,
        "std_scale": args.std_scale,
        "baseline": base_metrics,
        "calibrated": calib_metrics,
        "retrained_checkpoint": str(retrained_checkpoint) if retrained_checkpoint else None,
        "retrained": retrained_metrics,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved JSON report to {args.output}")


if __name__ == "__main__":
    main()
