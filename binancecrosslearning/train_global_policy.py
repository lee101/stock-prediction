from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from src.torch_device_utils import require_cuda as require_cuda_device

from newnanoalpacahourlyexp.config import DatasetConfig, ExperimentConfig
from newnanoalpacahourlyexp.data import AlpacaMultiSymbolDataModule
from newnanoalpacahourlyexp.run_experiment import evaluate_model, _load_model


DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,LINKUSDT,ADAUSDT,APTUSDT,AVAXUSDT"


def _parse_symbols(raw: str) -> List[str]:
    symbols = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type != "cuda":
            raise RuntimeError(f"GPU required for training/inference; received device={device_arg!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for training/inference but CUDA is not available.")
        return device
    return require_cuda_device("binance cross-learning training", allow_fallback=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a global policy across multiple Binance symbols.")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--target-symbol", default=None, help="Symbol to use for validation metrics.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--weight-decay-schedule", default="linear_to_zero")
    parser.add_argument("--weight-decay-end", type=float, default=0.0)
    parser.add_argument("--optimizer", default="muon_mix")
    parser.add_argument("--model-arch", default="nano", help="classic or nano")
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--logits-softcap", type=float, default=12.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-momentum-start", type=float, default=0.85)
    parser.add_argument("--muon-momentum-warmup-steps", type=int, default=300)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--causal-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--attention-window", type=int, default=64)
    parser.add_argument("--use-residual-scalars", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual-scale-init", type=float, default=1.0)
    parser.add_argument("--skip-scale-init", type=float, default=0.1)
    parser.add_argument("--use-value-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value-embedding-every", type=int, default=2)
    parser.add_argument("--value-embedding-scale", type=float, default=1.0)
    parser.add_argument("--dry-train-steps", type=int, default=300)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--use-flash-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--maker-fee", type=float, default=0.0)
    parser.add_argument("--periods-per-year", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default="binance_cross_global")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--context-lengths", default="64,96,192")
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", default="1,4,24")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--forecast-cache-root", default="binancecrosslearning/forecast_cache")
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument(
        "--moving-average-windows",
        default=None,
        help="Override MA windows (comma-separated hours).",
    )
    parser.add_argument("--min-history-hours", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-mixed-asset", action="store_true")
    parser.add_argument("--eval-days", type=float, default=None)
    parser.add_argument("--eval-hours", type=float, default=None)
    args = parser.parse_args()

    device = _resolve_device(args.device)

    symbols = _parse_symbols(args.symbols)
    target_symbol = args.target_symbol.upper() if args.target_symbol else symbols[0]

    ctx_lengths = tuple(int(x) for x in args.context_lengths.split(",") if x)
    experiment_cfg = ExperimentConfig(context_lengths=ctx_lengths, trim_ratio=args.trim_ratio)
    forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x)

    if args.moving_average_windows:
        ma_windows = tuple(int(x) for x in args.moving_average_windows.split(",") if x.strip())
    else:
        ma_windows = DatasetConfig().moving_average_windows

    min_history_hours = args.min_history_hours if args.min_history_hours is not None else DatasetConfig().min_history_hours

    data_cfg = DatasetConfig(
        symbol=target_symbol,
        data_root=Path(args.data_root) if args.data_root else None,
        sequence_length=args.sequence_length,
        cache_only=args.cache_only,
        forecast_horizons=forecast_horizons,
        allow_mixed_asset_class=args.allow_mixed_asset,
        forecast_cache_root=Path(args.forecast_cache_root),
        moving_average_windows=ma_windows,
        min_history_hours=min_history_hours,
    )
    data = AlpacaMultiSymbolDataModule(symbols, data_cfg)

    attention_window = None
    if args.attention_window and args.attention_window > 0:
        attention_window = int(args.attention_window)

    periods_per_year = float(args.periods_per_year) if args.periods_per_year is not None else float(
        data.asset_meta.periods_per_year
    )
    maker_fee = float(args.maker_fee) if args.maker_fee is not None else float(data.asset_meta.maker_fee)

    training_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        weight_decay_schedule=args.weight_decay_schedule,
        weight_decay_end=args.weight_decay_end,
        optimizer_name=args.optimizer,
        model_arch=args.model_arch,
        num_kv_heads=args.num_kv_heads,
        mlp_ratio=args.mlp_ratio,
        logits_softcap=args.logits_softcap,
        rope_base=args.rope_base,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_momentum_start=args.muon_momentum_start,
        muon_momentum_warmup_steps=args.muon_momentum_warmup_steps,
        muon_ns_steps=args.muon_ns_steps,
        muon_nesterov=args.muon_nesterov,
        use_qk_norm=args.qk_norm,
        use_causal_attention=args.causal_attn,
        rms_norm_eps=args.rms_norm_eps,
        attention_window=attention_window,
        use_residual_scalars=args.use_residual_scalars,
        residual_scale_init=args.residual_scale_init,
        skip_scale_init=args.skip_scale_init,
        use_value_embedding=args.use_value_embedding,
        value_embedding_every=args.value_embedding_every,
        value_embedding_scale=args.value_embedding_scale,
        maker_fee=maker_fee,
        periods_per_year=periods_per_year,
        dry_train_steps=args.dry_train_steps,
        use_compile=not args.no_compile,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        use_flash_attention=args.use_flash_attention,
        device=str(device),
        run_name=args.run_name,
        log_dir=Path("tensorboard_logs") / "binancecrosslearning",
        checkpoint_root=Path("binancecrosslearning") / "checkpoints",
    )

    trainer = BinanceHourlyTrainer(training_cfg, data)
    artifacts = trainer.train()
    if artifacts.best_checkpoint is None:
        raise RuntimeError("No checkpoint saved during training")

    checkpoint_path = artifacts.best_checkpoint
    model = _load_model(checkpoint_path, len(data.feature_columns), args.sequence_length)

    result = evaluate_model(
        model=model,
        data=data,
        horizon=args.horizon,
        aggregate=args.aggregate,
        experiment_cfg=experiment_cfg,
        maker_fee=maker_fee,
        eval_days=args.eval_days,
        eval_hours=args.eval_hours,
        device=device,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    if args.output_dir:
        history_payload = [entry.__dict__ for entry in artifacts.history]
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "training_history.json").write_text(json.dumps(history_payload, indent=2))

    print(f"Checkpoint: {checkpoint_path}")
    print(f"total_return: {result.total_return:.4f}")
    print(f"sortino: {result.sortino:.4f}")


if __name__ == "__main__":
    main()
