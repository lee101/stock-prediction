from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from loguru import logger

from chronos2_trainer import TrainerConfig, _fit_pipeline, _load_pipeline, _save_pipeline
from src.torch_device_utils import require_cuda as require_cuda_device

from .data import CovariateConfig, build_inputs_for_symbols


def _parse_symbols(raw: str) -> List[str]:
    symbols = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _build_covariate_config(args: argparse.Namespace) -> CovariateConfig:
    return CovariateConfig(
        include_volume=not args.no_volume,
        include_log_volume=not args.no_log_volume,
        include_return_1h=not args.no_return_1h,
        include_range_pct=not args.no_range_pct,
        include_body_pct=not args.no_body_pct,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-symbol Chronos2 fine-tuning (cross-learning).")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g., SOLUSD,LINKUSD,UNIUSD).")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--crypto-data-root", default=None)
    parser.add_argument("--stock-data-root", default=None)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--finetune-mode", choices=["full", "lora"], default="lora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q,k,v,o")
    parser.add_argument("--no-merge-lora", action="store_false", dest="merge_lora")
    parser.add_argument("--output-root", default="alpacanewccrosslearning/chronos_finetuned")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-volume", action="store_true")
    parser.add_argument("--no-log-volume", action="store_true")
    parser.add_argument("--no-return-1h", action="store_true")
    parser.add_argument("--no-range-pct", action="store_true")
    parser.add_argument("--no-body-pct", action="store_true")
    parser.add_argument("--preaug-strategy", default=None, help="Pre-augmentation strategy (baseline, percent_change, log_returns, etc).")
    parser.add_argument("--preaug-params", default=None, help="JSON dict of params for pre-augmentation.")
    args = parser.parse_args()

    require_cuda_device("chronos2 multi-symbol fine-tune", allow_fallback=False)

    symbols = _parse_symbols(args.symbols)
    cov_cfg = _build_covariate_config(args)

    preaug_params = json.loads(args.preaug_params) if args.preaug_params else None
    train_inputs, val_inputs = build_inputs_for_symbols(
        symbols,
        data_root=Path(args.data_root) if args.data_root else None,
        crypto_root=Path(args.crypto_data_root) if args.crypto_data_root else None,
        stock_root=Path(args.stock_data_root) if args.stock_data_root else None,
        covariate_config=cov_cfg,
        val_hours=args.val_hours,
        preaug_strategy=args.preaug_strategy,
        preaug_params=preaug_params,
    )

    run_name = args.run_name or time.strftime("chronos2_multi_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_dir = output_root / run_name

    config = TrainerConfig(
        symbol="multi",
        data_root=None,
        output_root=output_root,
        model_id=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        val_hours=args.val_hours,
        finetune_mode=args.finetune_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=tuple(token.strip() for token in args.lora_targets.split(",") if token.strip()),
        merge_lora=args.merge_lora,
    )

    logger.info("Loading Chronos2 model {}", config.model_id)
    pipeline = _load_pipeline(config.model_id, config.device_map, config.torch_dtype)
    finetuned = _fit_pipeline(
        pipeline=pipeline,
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        config=config,
        output_dir=output_dir,
    )
    save_path = _save_pipeline(finetuned, output_dir, "finetuned")
    logger.info("Saved fine-tuned model to {}", save_path)

    metadata = {
        "symbols": symbols,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "finetune_mode": args.finetune_mode,
        "prediction_length": args.prediction_length,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "val_hours": args.val_hours,
        "preaug": {
            "strategy": args.preaug_strategy or "baseline",
            "params": preaug_params or {},
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": [token.strip() for token in args.lora_targets.split(",") if token.strip()],
            "merge": bool(args.merge_lora),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
