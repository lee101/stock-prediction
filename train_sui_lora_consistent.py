#!/usr/bin/env python3
"""Train SUI LoRA with MAE consistency metrics and preaug/dilation support."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preaug import get_augmentation
from preaug.inference_strategies import get_inference_strategy, BaseInferenceStrategy

DEFAULT_DATA_PATH = Path("trainingdatahourlybinance/SUIUSDT.csv")
DEFAULT_OUTPUT_ROOT = Path("chronos2_finetuned")
DEFAULT_RESULTS_DIR = Path("hyperparams/sui_lora_consistent")


@dataclass
class TrainConfig:
    context_length: int = 128
    prediction_length: int = 24
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_steps: int = 100
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    preaug: str = "percent_change"
    inference_strategy: str = "single"
    val_hours: int = 168
    test_hours: int = 168


@dataclass
class ConsistencyMetrics:
    mae_mean: float
    mae_std: float
    mae_max: float
    mae_p90: float
    mae_p95: float
    mae_percent_mean: float
    mae_percent_std: float
    mae_percent_max: float
    count: int

    def consistency_score(self) -> float:
        return self.mae_percent_mean + 0.5 * self.mae_percent_std + 0.3 * (self.mae_percent_max - self.mae_percent_mean)


def load_hourly_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, val_hours: int, test_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = len(df) - (val_hours + test_hours)
    val_end = len(df) - test_hours
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def compute_consistency_metrics(
    pipeline: Any,
    df: pd.DataFrame,
    context_length: int,
    prediction_length: int,
    start_idx: int,
    end_idx: int,
    preaug_name: str = "baseline",
    inference_strategy: Optional[BaseInferenceStrategy] = None,
) -> ConsistencyMetrics:
    if start_idx < context_length:
        start_idx = context_length
    if start_idx >= end_idx - prediction_length:
        return ConsistencyMetrics(
            mae_mean=float("inf"), mae_std=0, mae_max=float("inf"), mae_p90=float("inf"), mae_p95=float("inf"),
            mae_percent_mean=float("inf"), mae_percent_std=0, mae_percent_max=float("inf"), count=0
        )

    target_cols = ["open", "high", "low", "close"]
    close_idx = target_cols.index("close")

    quantiles = getattr(pipeline, "quantiles", None)
    q_index = int(np.argmin([abs(float(q) - 0.5) for q in quantiles])) if quantiles else 0

    window_maes: List[float] = []
    window_mae_pcts: List[float] = []

    step = max(1, prediction_length // 2)
    for idx in range(start_idx, end_idx - prediction_length, step):
        context = df.iloc[idx - context_length : idx]
        future = df.iloc[idx : idx + prediction_length]
        if len(context) < context_length or len(future) < prediction_length:
            continue

        ctx_data = context.copy()
        aug = get_augmentation(preaug_name) if preaug_name != "baseline" else None
        if aug:
            ctx_data = aug.transform_dataframe(ctx_data)

        inputs = ctx_data[target_cols].to_numpy(dtype=np.float32).T

        try:
            if inference_strategy:
                pred_vals = inference_strategy.predict(pipeline, inputs, prediction_length, q_index).T
            else:
                preds = pipeline.predict([inputs], prediction_length=prediction_length, batch_size=1)
                if not preds:
                    continue
                pred_tensor = preds[0].detach().cpu().numpy()
                pred_vals = pred_tensor[:, q_index, :].T
        except Exception as e:
            logger.warning("Prediction failed: {}", e)
            continue

        if aug:
            pred_vals = aug.inverse_transform_predictions(pred_vals, context, columns=target_cols)

        actual = future[target_cols].to_numpy(dtype=np.float32)
        actual_close = actual[:, close_idx]
        pred_close = pred_vals[:, close_idx]

        mae = float(np.mean(np.abs(actual_close - pred_close)))
        mae_pct = float(np.mean(np.abs(actual_close - pred_close) / np.clip(np.abs(actual_close), 1e-8, None)) * 100)

        window_maes.append(mae)
        window_mae_pcts.append(mae_pct)

    if not window_maes:
        return ConsistencyMetrics(
            mae_mean=float("inf"), mae_std=0, mae_max=float("inf"), mae_p90=float("inf"), mae_p95=float("inf"),
            mae_percent_mean=float("inf"), mae_percent_std=0, mae_percent_max=float("inf"), count=0
        )

    return ConsistencyMetrics(
        mae_mean=float(np.mean(window_maes)),
        mae_std=float(np.std(window_maes)),
        mae_max=float(np.max(window_maes)),
        mae_p90=float(np.percentile(window_maes, 90)),
        mae_p95=float(np.percentile(window_maes, 95)),
        mae_percent_mean=float(np.mean(window_mae_pcts)),
        mae_percent_std=float(np.std(window_mae_pcts)),
        mae_percent_max=float(np.max(window_mae_pcts)),
        count=len(window_maes),
    )


def train_lora(
    config: TrainConfig,
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Any, Dict[str, Any]]:
    from chronos import ChronosPipeline
    from transformers import TrainingArguments
    from chronos.scripts.training.train import ChronosConfig, train as chronos_train

    target_cols = ["open", "high", "low", "close"]
    train_df, val_df, test_df = split_data(df, config.val_hours, config.test_hours)

    aug = get_augmentation(config.preaug) if config.preaug != "baseline" else None
    if aug:
        train_df = aug.transform_dataframe(train_df)

    train_data = train_df[target_cols].to_numpy(dtype=np.float32).T.tolist()
    output_dir.mkdir(parents=True, exist_ok=True)

    chronos_cfg = ChronosConfig(
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        use_lora=True,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        max_steps=config.num_steps,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        save_strategy="steps",
        save_steps=config.num_steps,
        logging_steps=50,
        report_to=[],
    )

    logger.info("Training SUI LoRA: ctx={} lr={} preaug={}", config.context_length, config.learning_rate, config.preaug)
    t0 = time.time()
    chronos_train(
        train_datasets=train_data,
        model_id="amazon/chronos-t5-small",
        config=chronos_cfg,
        training_args=training_args,
    )
    train_time = time.time() - t0

    merged_dir = output_dir / "finetuned-ckpt"
    lora_dir = output_dir / "lora-adapter"

    from peft import PeftModel
    base = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cuda", torch_dtype="float32")

    ckpt_dir = list(output_dir.glob("checkpoint-*"))
    if ckpt_dir:
        adapter_path = max(ckpt_dir, key=lambda p: int(p.name.split("-")[1]))
        peft_model = PeftModel.from_pretrained(base.model.model, adapter_path)
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        base.model.model.config.save_pretrained(merged_dir)
        base.model.tokenizer.save_pretrained(merged_dir)
        peft_model.save_pretrained(lora_dir)

    pipeline = ChronosPipeline.from_pretrained(str(merged_dir), device_map="cuda", torch_dtype="float32")

    inference_strat = None
    if config.inference_strategy != "single":
        inference_strat = get_inference_strategy(config.inference_strategy, target_points=config.context_length)

    val_start = len(df) - config.val_hours - config.test_hours
    val_end = len(df) - config.test_hours
    val_metrics = compute_consistency_metrics(
        pipeline, df, config.context_length, config.prediction_length,
        val_start, val_end, config.preaug, inference_strat
    )

    test_start = len(df) - config.test_hours
    test_metrics = compute_consistency_metrics(
        pipeline, df, config.context_length, config.prediction_length,
        test_start, len(df), config.preaug, inference_strat
    )

    results = {
        "config": asdict(config),
        "val_metrics": asdict(val_metrics),
        "val_consistency_score": val_metrics.consistency_score(),
        "test_metrics": asdict(test_metrics),
        "test_consistency_score": test_metrics.consistency_score(),
        "train_time_sec": train_time,
        "output_dir": str(output_dir),
    }

    logger.info("Val MAE: {:.4f}% (std={:.4f}%) consistency={:.4f}",
                val_metrics.mae_percent_mean, val_metrics.mae_percent_std, val_metrics.consistency_score())
    logger.info("Test MAE: {:.4f}% consistency={:.4f}",
                test_metrics.mae_percent_mean, test_metrics.consistency_score())

    return pipeline, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--preaug", default="percent_change")
    parser.add_argument("--inference-strategy", default="single", choices=["single", "dilation"])
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--test-hours", type=int, default=168)
    args = parser.parse_args()

    df = load_hourly_frame(args.data_path)
    logger.info("Loaded {} rows from {}", len(df), args.data_path)

    config = TrainConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        preaug=args.preaug,
        inference_strategy=args.inference_strategy,
        val_hours=args.val_hours,
        test_hours=args.test_hours,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"SUI_lora_{config.preaug}_ctx{config.context_length}_lr{config.learning_rate}_r{config.lora_r}_{timestamp}"
    output_dir = args.output_root / run_name

    pipeline, results = train_lora(config, df, output_dir)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / f"{run_name}.json"
    result_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to {}", result_path)

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps({
        "context_length": config.context_length,
        "prediction_length": config.prediction_length,
        "learning_rate": config.learning_rate,
        "lora_r": config.lora_r,
        "preaug": config.preaug,
        "inference_strategy": config.inference_strategy,
        "final_metrics": {
            "val_mae_percent": results["val_metrics"]["mae_percent_mean"],
            "consistency_score": results["val_consistency_score"],
            "test_mae_percent": results["test_metrics"]["mae_percent_mean"],
        }
    }, indent=2))


if __name__ == "__main__":
    main()
