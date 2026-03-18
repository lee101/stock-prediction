#!/usr/bin/env python3
"""Chronos2 LoRA fine-tuning trainer for hourly crypto data.

Supports full fine-tuning, LoRA adapters, validation/test MAE reporting,
and simple hyperparameter sweeps.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.preaug import PreAugmentationChoice, PreAugmentationSelector, candidate_preaug_symbols

DEFAULT_MODEL_ID = "amazon/chronos-2"
DEFAULT_TARGET_COLS = ("open", "high", "low", "close")
DEFAULT_DATA_ROOT = Path("trainingdata/csv")
DEFAULT_OUTPUT_ROOT = Path("chronos2_finetuned")
DEFAULT_HYPERPARAM_ROOT = Path("hyperparams") / "chronos2"
DEFAULT_HYPERPARAM_HOURLY_DIR = DEFAULT_HYPERPARAM_ROOT / "hourly_finetune"
DEFAULT_HYPERPARAM_LORA_DIR = DEFAULT_HYPERPARAM_ROOT / "hourly_lora"


def compute_mae_percent(forecasts, actuals):
    forecasts = np.asarray(forecasts, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    return float((np.abs(forecasts - actuals) / np.clip(np.abs(actuals), 1e-8, None)).mean() * 100)


def _resolve_preaug_choice(
    symbol: str,
    preferred_choice: Optional[PreAugmentationChoice],
    best_dirs: Sequence[Path],
) -> tuple[Optional[PreAugmentationChoice], Optional[str]]:
    """Resolve a pre-augmentation strategy choice for the requested symbol.

    Args:
        symbol: The requested trading symbol.
        preferred_choice: Optional already-resolved choice (returned as-is).
        best_dirs: Ordered directories containing per-symbol JSON choice payloads.

    Returns:
        (choice, source_path). Source path is a string for compatibility with older
        trainer codepaths/tests.
    """

    if preferred_choice is not None:
        return preferred_choice, str(getattr(preferred_choice, "source_path", "")) or None

    selector = PreAugmentationSelector(best_dirs)
    for candidate in candidate_preaug_symbols(symbol):
        choice = selector.get_choice(candidate)
        if choice is not None:
            return choice, str(choice.source_path)

    return None, None


@dataclass
class TrainerConfig:
    symbol: str
    data_root: Optional[Path]
    output_root: Path
    model_id: str = DEFAULT_MODEL_ID
    device_map: str = "cuda"
    torch_dtype: Optional[str] = None
    target_cols: Tuple[str, ...] = DEFAULT_TARGET_COLS
    prediction_length: int = 1
    context_length: int = 1024
    batch_size: int = 64
    learning_rate: float = 1e-5
    num_steps: int = 1000
    val_hours: int = 168
    test_hours: int = 168
    finetune_mode: str = "full"  # full|lora|none
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Tuple[str, ...] = ("q", "k", "v", "o")
    merge_lora: bool = True
    seed: int = 1337
    save_name: Optional[str] = None
    covariate_symbols: Tuple[str, ...] = ()
    covariate_cols: Tuple[str, ...] = ("close",)


@dataclass
class WindowMetrics:
    mae: float
    rmse: float
    mae_percent: float
    pct_return_mae: float
    count: int


@dataclass
class FineTuneReport:
    symbol: str
    finetune_mode: str
    config: Dict[str, Any]
    train_end: str
    val_end: str
    test_end: str
    val_metrics: WindowMetrics
    test_metrics: WindowMetrics
    output_dir: str


def _parse_torch_dtype(value: Optional[str]):
    if value is None:
        return None
    import torch
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    normalized = value.strip().lower()
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype '{value}'.")
    return mapping[normalized]


def _load_hourly_frame(csv_path: Path, target_cols: Sequence[str]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Hourly data not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise KeyError(f"{csv_path} missing 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    for col in target_cols:
        if col not in df.columns:
            raise KeyError(f"{csv_path} missing required column '{col}'")
    df = df.set_index("timestamp")
    if len(df.index) > 1:
        full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
        df = df.reindex(full_index)
        df = df.ffill().bfill()
    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df


def _resolve_data_path(symbol: str, data_root: Optional[Path]) -> Path:
    root = data_root or DEFAULT_DATA_ROOT
    candidate = root / f"{symbol}.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No data for {symbol} under {root}")


def _split_windows(df: pd.DataFrame, val_hours: int, test_hours: int):
    if val_hours <= 0 or test_hours <= 0:
        raise ValueError("val_hours and test_hours must be positive")
    if len(df) <= val_hours + test_hours:
        raise ValueError(f"Not enough rows ({len(df)}) for val={val_hours} test={test_hours}")
    train_end = len(df) - (val_hours + test_hours)
    val_end = len(df) - test_hours
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def _prepare_inputs(df: pd.DataFrame, target_cols: Sequence[str]) -> List[Dict[str, Any]]:
    values = df[list(target_cols)].to_numpy(dtype=np.float32).T
    return [{"target": values}]


def _prepare_inputs_multivariate(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    covariate_dfs: List[pd.DataFrame],
    covariate_cols: Sequence[str],
) -> List[Dict[str, Any]]:
    target_values = df[list(target_cols)].to_numpy(dtype=np.float32).T
    all_channels = [target_values]
    for cov_df in covariate_dfs:
        aligned = cov_df.reindex(df.index)
        aligned = aligned.ffill().bfill()
        cov_values = aligned[list(covariate_cols)].to_numpy(dtype=np.float32).T
        all_channels.append(cov_values)
    stacked = np.concatenate(all_channels, axis=0)
    return [{"target": stacked}]


def _median_quantile_index(quantiles: Sequence[float]) -> int:
    if not quantiles:
        return 0
    distances = [abs(float(q) - 0.5) for q in quantiles]
    return int(np.argmin(distances))


def _evaluate_pipeline(
    pipeline: Any,
    df: pd.DataFrame,
    target_cols: Sequence[str],
    context_length: int,
    prediction_length: int,
    start_idx: int,
    end_idx: int,
    preaug_choice: Optional[PreAugmentationChoice] = None,
) -> WindowMetrics:
    if start_idx < context_length:
        start_idx = context_length
    if end_idx <= start_idx:
        raise ValueError("Invalid evaluation window")

    close_col = "close" if "close" in target_cols else target_cols[0]

    actual_values: List[float] = []
    predicted_values: List[float] = []
    actual_returns: List[float] = []
    predicted_returns: List[float] = []

    quantiles = getattr(pipeline, "quantiles", None)
    if quantiles is None:
        raise RuntimeError("Pipeline missing quantiles attribute")
    q_index = _median_quantile_index(list(quantiles))

    for idx in range(start_idx, end_idx):
        context = df.iloc[idx - context_length : idx]
        future = df.iloc[idx : idx + prediction_length]
        if len(context) < context_length or len(future) < prediction_length:
            continue

        inputs = context[list(target_cols)].to_numpy(dtype=np.float32).T
        predictions = pipeline.predict([inputs], prediction_length=prediction_length, batch_size=1)
        if not predictions:
            continue
        pred_tensor = predictions[0].detach().cpu().numpy()
        pred_vals = pred_tensor[:, q_index, :]
        pred_matrix = pred_vals.T

        actual_matrix = future[list(target_cols)].to_numpy(dtype=np.float32)

        close_idx = list(target_cols).index(close_col)
        actual_close = actual_matrix[:, close_idx]
        pred_close = pred_matrix[:, close_idx]
        actual_values.extend(actual_close.tolist())
        predicted_values.extend(pred_close.tolist())

        base = float(context[close_col].iloc[-1]) if close_col in context.columns else float(actual_close[0])
        if base != 0:
            actual_returns.extend(((actual_close - base) / base).tolist())
            predicted_returns.extend(((pred_close - base) / base).tolist())

    if not actual_values:
        raise RuntimeError("No evaluation windows produced metrics")

    actual_arr = np.array(actual_values, dtype=np.float32)
    pred_arr = np.array(predicted_values, dtype=np.float32)
    abs_errors = np.abs(actual_arr - pred_arr)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))
    mae_pct = compute_mae_percent(pred_arr, actual_arr)

    if actual_returns and predicted_returns:
        pct_return_mae = float(np.mean(np.abs(
            np.array(actual_returns, dtype=np.float32) - np.array(predicted_returns, dtype=np.float32)
        )))
    else:
        pct_return_mae = float("nan")

    return WindowMetrics(mae=mae, rmse=rmse, mae_percent=mae_pct, pct_return_mae=pct_return_mae, count=len(actual_values))


def _load_pipeline(model_id: str, device_map: str, torch_dtype: Optional[str]):
    try:
        from chronos import Chronos2Pipeline
    except Exception as exc:
        raise RuntimeError("Chronos2Pipeline import failed; install chronos-forecasting>=2.0") from exc
    dtype = _parse_torch_dtype(torch_dtype)
    logger.info("Loading Chronos2 model {}", model_id)
    return Chronos2Pipeline.from_pretrained(model_id, device_map=device_map, torch_dtype=dtype)


def _apply_lora(model: Any, config: TrainerConfig):
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise RuntimeError("peft is required for LoRA finetuning. Install peft>=0.11") from exc
    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        bias="none",
    )
    return get_peft_model(model, lora_cfg)


def _fit_pipeline(
    pipeline: Any,
    train_inputs: List[Dict[str, Any]],
    val_inputs: Optional[List[Dict[str, Any]]],
    config: TrainerConfig,
    output_dir: Path,
    *,
    resume_from_checkpoint: Optional[str] = None,
) -> Any:
    from copy import deepcopy

    try:
        from chronos.chronos2 import Chronos2Model
        from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
        from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback
        from transformers.training_args import TrainingArguments
    except Exception as exc:
        raise RuntimeError("Chronos2 training dependencies missing.") from exc

    base_model = pipeline.model
    model_config = deepcopy(base_model.config)
    model = Chronos2Model(model_config).to(base_model.device)
    model.load_state_dict(base_model.state_dict())

    use_lora = config.finetune_mode == "lora"
    if use_lora:
        model = _apply_lora(model, config)

    context_length = config.context_length
    prediction_length = config.prediction_length

    train_dataset = Chronos2Dataset.convert_inputs(
        inputs=train_inputs,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=config.batch_size,
        output_patch_size=pipeline.model_output_patch_size,
        min_past=prediction_length,
        mode=DatasetMode.TRAIN,
    )

    eval_dataset = None
    callbacks: List[Any] = []
    training_kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=100,
        disable_tqdm=False,
        report_to="none",
        max_steps=config.num_steps,
        gradient_accumulation_steps=1,
        dataloader_num_workers=1,
        tf32=False,
        bf16=False,
        save_only_model=True,
        prediction_loss_only=True,
        save_total_limit=1,
        save_strategy="no",
        eval_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model=None,
        use_cpu=str(base_model.device) == "cpu",
    )

    if val_inputs is not None:
        eval_dataset = Chronos2Dataset.convert_inputs(
            inputs=val_inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=config.batch_size,
            output_patch_size=pipeline.model_output_patch_size,
            mode=DatasetMode.VALIDATION,
        )
        training_kwargs.update({
            "save_strategy": "steps",
            "save_steps": 100,
            "eval_strategy": "steps",
            "eval_steps": 100,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "label_names": ["future_target"],
        })
        callbacks.append(EvaluateAndSaveFinalStepCallback())

    training_args = TrainingArguments(**training_kwargs)

    trainer = Chronos2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    resume_path: Optional[str] = None
    if resume_from_checkpoint:
        token = str(resume_from_checkpoint).strip()
        if token.lower() in {"1", "true", "yes", "y", "auto", "last"}:
            resume_path = _find_last_checkpoint(output_dir)
        else:
            candidate = Path(token)
            if not candidate.is_dir():
                candidate = output_dir / token
            if candidate.is_dir():
                resume_path = str(candidate)
    if resume_path:
        logger.info("Resuming from checkpoint: {}", resume_path)
    trainer.train(resume_from_checkpoint=resume_path)

    if use_lora:
        adapter_dir = Path(output_dir) / "lora-adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(adapter_dir)
        except Exception as exc:
            logger.warning("Failed to save LoRA adapter: {}", exc)
        if config.merge_lora:
            try:
                model = model.merge_and_unload()
            except Exception as exc:
                logger.warning("Failed to merge LoRA weights: {}", exc)

    model.chronos_config.max_output_patches = max(
        model.chronos_config.max_output_patches,
        math.ceil(prediction_length / pipeline.model_output_patch_size),
    )

    return pipeline.__class__(model=model)


def _save_pipeline(pipeline: Any, output_dir: Path, name: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / name
    pipeline.save_pretrained(save_path)
    return save_path


def _find_last_checkpoint(output_dir: Path) -> Optional[str]:
    if not output_dir.exists():
        return None
    best_step = -1
    best_path: Optional[Path] = None
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.split("-", 1)[1].strip())
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_path = child
    return str(best_path) if best_path is not None else None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _write_report(report: FineTuneReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_slug = Path(report.output_dir).name
    report_path = output_dir / f"{report.symbol}_{report.finetune_mode}_{run_slug}.json"
    payload = asdict(report)
    payload["val_metrics"] = asdict(report.val_metrics)
    payload["test_metrics"] = asdict(report.test_metrics)
    report_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return report_path


def run_finetune(config: TrainerConfig) -> FineTuneReport:
    symbol = config.symbol.upper()
    data_path = _resolve_data_path(symbol, config.data_root)
    df = _load_hourly_frame(data_path, config.target_cols)
    train_df, val_df, test_df = _split_windows(df, config.val_hours, config.test_hours)

    covariate_dfs: List[pd.DataFrame] = []
    if config.covariate_symbols:
        for cov_sym in config.covariate_symbols:
            cov_path = _resolve_data_path(cov_sym.upper(), config.data_root)
            cov_cols = list(config.covariate_cols) if config.covariate_cols else list(config.target_cols)
            cov_df = _load_hourly_frame(cov_path, tuple(cov_cols))
            cov_df = cov_df.set_index("timestamp") if "timestamp" in cov_df.columns else cov_df
            covariate_dfs.append(cov_df)
            logger.info("Loaded covariate {} ({} rows, cols={})", cov_sym, len(cov_df), cov_cols)

    if config.finetune_mode == "lora" and config.learning_rate <= 1e-5:
        logger.warning(
            "LoRA often benefits from higher LR (e.g. 1e-4). Current: {:.2e}",
            config.learning_rate,
        )

    pipeline = _load_pipeline(config.model_id, config.device_map, config.torch_dtype)

    if covariate_dfs:
        train_df_indexed = train_df.set_index("timestamp") if "timestamp" in train_df.columns else train_df
        val_df_indexed = val_df.set_index("timestamp") if "timestamp" in val_df.columns else val_df
        train_inputs = _prepare_inputs_multivariate(
            train_df_indexed, config.target_cols, covariate_dfs, config.covariate_cols)
        val_inputs = _prepare_inputs_multivariate(
            val_df_indexed, config.target_cols, covariate_dfs, config.covariate_cols)
    else:
        train_inputs = _prepare_inputs(train_df, config.target_cols)
        val_inputs = _prepare_inputs(val_df, config.target_cols)

    run_name = config.save_name or f"{symbol}_{config.finetune_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = config.output_root / run_name

    finetuned_pipeline = pipeline
    if config.finetune_mode != "none":
        finetuned_pipeline = _fit_pipeline(pipeline, train_inputs, val_inputs, config, output_dir)
        _save_pipeline(finetuned_pipeline, output_dir, "finetuned-ckpt")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    val_start = len(train_df)
    val_end = len(train_df) + len(val_df)
    test_start = val_end
    test_end = len(full_df)

    val_metrics = _evaluate_pipeline(
        finetuned_pipeline, full_df, config.target_cols,
        config.context_length, config.prediction_length, val_start, val_end,
    )
    test_metrics = _evaluate_pipeline(
        finetuned_pipeline, full_df, config.target_cols,
        config.context_length, config.prediction_length, test_start, test_end,
    )

    report = FineTuneReport(
        symbol=symbol,
        finetune_mode=config.finetune_mode,
        config=asdict(config),
        train_end=str(train_df["timestamp"].iloc[-1]),
        val_end=str(val_df["timestamp"].iloc[-1]),
        test_end=str(test_df["timestamp"].iloc[-1]),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        output_dir=str(output_dir),
    )

    hyperparam_dir = DEFAULT_HYPERPARAM_LORA_DIR if config.finetune_mode == "lora" else DEFAULT_HYPERPARAM_HOURLY_DIR
    hyperparam_dir.mkdir(parents=True, exist_ok=True)
    _write_report(report, hyperparam_dir)

    return report


def _load_sweep_grid(path: Optional[Path]) -> Dict[str, List[Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    grid = payload.get("grid") if isinstance(payload, dict) else None
    if not isinstance(grid, dict):
        raise ValueError("Sweep config must contain a 'grid' dict")
    return {str(k): list(v) for k, v in grid.items()}


def _expand_grid(base: TrainerConfig, grid: Dict[str, List[Any]]) -> List[TrainerConfig]:
    if not grid:
        return [base]
    keys = sorted(grid.keys())
    configs: List[TrainerConfig] = []
    for values in product(*[grid[k] for k in keys]):
        params = {k: v for k, v in zip(keys, values)}
        cfg_dict = asdict(base)
        cfg_dict.update(params)
        configs.append(TrainerConfig(**cfg_dict))
    return configs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chronos2 fine-tuning trainer")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--test-hours", type=int, default=168)
    parser.add_argument("--finetune-mode", type=str, default="full", choices=["full", "lora", "none"])
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", type=str, default="q,k,v,o")
    parser.add_argument("--merge-lora", action="store_true", default=True)
    parser.add_argument("--no-merge-lora", action="store_false", dest="merge_lora")
    parser.add_argument("--sweep-config", type=Path, default=None)
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument(
        "--target-cols", type=str, default=",".join(DEFAULT_TARGET_COLS),
        help="Comma-separated target columns",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    target_cols = tuple(col.strip() for col in args.target_cols.split(",") if col.strip())

    config = TrainerConfig(
        symbol=args.symbol,
        data_root=args.data_root,
        output_root=args.output_root,
        model_id=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        target_cols=target_cols,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        val_hours=args.val_hours,
        test_hours=args.test_hours,
        finetune_mode=args.finetune_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
        merge_lora=args.merge_lora,
        save_name=args.save_name,
    )

    grid = _load_sweep_grid(args.sweep_config)
    configs = _expand_grid(config, grid)
    logger.info("Running {} configuration(s)", len(configs))

    reports: List[FineTuneReport] = []
    for idx, cfg in enumerate(configs, 1):
        logger.info("[{}/{}] Training {} mode={}", idx, len(configs), cfg.symbol, cfg.finetune_mode)
        report = run_finetune(cfg)
        reports.append(report)
        logger.info("Validation MAE%: {:.4f}", report.val_metrics.mae_percent)

    if len(reports) > 1:
        best = min(reports, key=lambda r: r.val_metrics.mae_percent)
        logger.info("Best config: {} (MAE%={:.4f})", best.output_dir, best.val_metrics.mae_percent)


if __name__ == "__main__":
    main()
