#!/usr/bin/env python3
"""
Compare the newly trained Toto checkpoint against the public Toto baseline.

Run this script after generating a checkpoint via ``tototraining/toto_trainer.py``.
It reports absolute-price MAE and return MAE for both models over the most recent
window of the BTCUSD training series.
"""
from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.toto_aggregation import aggregate_quantile_plus_std
from src.models.toto_wrapper import TotoPipeline, Toto


DATA_PATH = Path("trainingdata") / "BTCUSD.csv"
DEFAULT_CHECKPOINT_PATH = Path("tototraining") / "checkpoints" / "our_run" / "latest.pt"
BASE_MODEL_ID = "Datadog/Toto-Open-Base-1.0"

EVAL_POINTS = 64
MIN_CONTEXT = 192
NUM_SAMPLES = 4096
SAMPLES_PER_BATCH = 512
QUANTILE = 0.15
STD_SCALE = 0.15


def _load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset at {DATA_PATH}. Run data preparation first."
        )
    df = pd.read_csv(DATA_PATH)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError("Dataset must include 'timestamp' and 'close' columns.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_checkpoint_config(checkpoint_path: Path) -> Tuple[Dict, Dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise KeyError("Checkpoint is missing the serialized TrainerConfig.")
    state_dict = checkpoint["model_state_dict"]
    return config, state_dict


def _extract_model_kwargs(config: Dict) -> Dict:
    """Project TrainerConfig down to Toto constructor arguments."""
    model_kwargs = {
        "patch_size": config["patch_size"],
        "stride": config["stride"],
        "embed_dim": config["embed_dim"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "mlp_hidden_dim": config["mlp_hidden_dim"],
        "dropout": config["dropout"],
        "spacewise_every_n_layers": config.get("spacewise_every_n_layers", 2),
        "scaler_cls": config["scaler_cls"],
        "output_distribution_classes": config["output_distribution_classes"],
        "use_memory_efficient_attention": config.get("memory_efficient_attention", True),
    }
    # Some checkpoints may include extra knobs that Toto accepts.
    if "stabilize_with_global" in config:
        model_kwargs["stabilize_with_global"] = config["stabilize_with_global"]
    if "scale_factor_exponent" in config:
        model_kwargs["scale_factor_exponent"] = config["scale_factor_exponent"]
    return model_kwargs


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

    pretrained_model_id = config.get("pretrained_model_id") or "Datadog/Toto-Open-Base-1.0"
    base_model = Toto.from_pretrained(pretrained_model_id, map_location="cpu")
    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing parameters in state_dict: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected parameters in state_dict: {unexpected}")
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
    preds = []
    actuals = []
    start = max(MIN_CONTEXT, len(prices) - eval_points)

    patch_size = getattr(getattr(pipeline, "model", None), "patch_size", None)
    if patch_size is None:
        patch_size = getattr(getattr(getattr(pipeline, "model", None), "model", None), "patch_embed", None)
        patch_size = getattr(patch_size, "patch_size", 1)

    first_idx = None
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
        aggregated = aggregate_quantile_plus_std(
            samples,
            quantile=quantile,
            std_scale=std_scale,
        )
        preds.append(float(np.atleast_1d(aggregated)[0]))
        actuals.append(float(prices[idx]))
        if first_idx is None:
            first_idx = idx

    if not actuals:
        raise RuntimeError("No evaluation points were collected; reduce MIN_CONTEXT or EVAL_POINTS.")

    prev_idx = max(start - 1, (first_idx - 1) if first_idx else start - 1)
    prev_price = float(prices[prev_idx])
    return np.asarray(preds, dtype=np.float64), np.asarray(actuals, dtype=np.float64), prev_price


def _compute_return_mae(preds: np.ndarray, actuals: np.ndarray, prev_price: float) -> float:
    prev = prev_price
    errors = []
    for pred, actual in zip(preds, actuals):
        pred_return = (pred - prev) / prev if prev != 0 else 0.0
        actual_return = (actual - prev) / prev if prev != 0 else 0.0
        errors.append(abs(pred_return - actual_return))
        prev = actual
    return float(np.mean(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Toto checkpoints.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("TOTO_CHECKPOINT_PATH"),
        help="Path to the checkpoint (.pt) file for the trained Toto model.",
    )
    parser.add_argument(
        "--eval-points",
        type=int,
        default=EVAL_POINTS,
        help="Number of evaluation points from the end of the series.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="Number of Monte Carlo samples per forecast.",
    )
    parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=SAMPLES_PER_BATCH,
        help="Samples processed per batch to control GPU memory.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=QUANTILE,
        help="Quantile used in the quantile+std aggregator (0-1).",
    )
    parser.add_argument(
        "--std-scale",
        type=float,
        default=STD_SCALE,
        help="Standard deviation multiplier in the aggregator.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float32", "float16", "bfloat16", None],
        default=None,
        help="Optional torch dtype override for both models when running on GPU.",
    )
    parser.add_argument(
        "--max-oom-retries",
        type=int,
        default=2,
        help="Number of automatic OOM retries inside TotoPipeline.",
    )
    parser.add_argument(
        "--min-samples-per-batch",
        type=int,
        default=32,
        help="Minimum samples per batch when autotuning after OOM.",
    )
    parser.add_argument(
        "--min-num-samples",
        type=int,
        default=256,
        help="Minimum total samples when autotuning after OOM.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device to use for inference.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        device = args.device
    df = _load_dataset()
    prices = df["close"].to_numpy(dtype=np.float64)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(args.torch_dtype) if args.torch_dtype else None

    print("Loading Toto baselines...")
    base_pipeline = TotoPipeline.from_pretrained(
        model_id=BASE_MODEL_ID,
        device_map=device,
        torch_dtype=torch_dtype,
        max_oom_retries=args.max_oom_retries,
        min_samples_per_batch=args.min_samples_per_batch,
        min_num_samples=args.min_num_samples,
    )
    our_pipeline = _build_pipeline_from_checkpoint(
        checkpoint_path,
        device=device,
        torch_dtype=torch_dtype,
        max_oom_retries=args.max_oom_retries,
        min_samples_per_batch=args.min_samples_per_batch,
        min_num_samples=args.min_num_samples,
    )

    print("Collecting forecasts...")
    eval_points = args.eval_points
    base_preds, actuals, prev_price = _collect_predictions(
        base_pipeline,
        prices,
        eval_points,
        num_samples=args.num_samples,
        samples_per_batch=args.samples_per_batch,
        quantile=args.quantile,
        std_scale=args.std_scale,
    )
    our_preds, _, _ = _collect_predictions(
        our_pipeline,
        prices,
        eval_points,
        num_samples=args.num_samples,
        samples_per_batch=args.samples_per_batch,
        quantile=args.quantile,
        std_scale=args.std_scale,
    )

    base_mae = float(np.mean(np.abs(actuals - base_preds)))
    our_mae = float(np.mean(np.abs(actuals - our_preds)))

    base_return_mae = _compute_return_mae(base_preds, actuals, prev_price)
    our_return_mae = _compute_return_mae(our_preds, actuals, prev_price)

    summary = {
        "evaluation_points": len(actuals),
        "base_price_mae": base_mae,
        "our_price_mae": our_mae,
        "price_mae_delta": our_mae - base_mae,
        "base_return_mae": base_return_mae,
        "our_return_mae": our_return_mae,
        "return_mae_delta": our_return_mae - base_return_mae,
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "num_samples": args.num_samples,
        "samples_per_batch": args.samples_per_batch,
        "quantile": args.quantile,
        "std_scale": args.std_scale,
        "torch_dtype": args.torch_dtype,
    }

    print("\n=== Toto Baseline vs Our Trained Toto ===")
    print(f"Evaluation points: {summary['evaluation_points']}")
    print(f"Base Toto price MAE: {base_mae:.6f}")
    print(f"Our Toto price MAE:  {our_mae:.6f} (Δ {summary['price_mae_delta']:+.6f})")
    print(f"Base Toto return MAE: {base_return_mae:.6f}")
    print(f"Our Toto return MAE:  {our_return_mae:.6f} (Δ {summary['return_mae_delta']:+.6f})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("\nJSON summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
