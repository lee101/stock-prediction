from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import torch

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .config import DatasetConfig, ExperimentConfig
    from .data import AlpacaHourlyDataModule, AlpacaMultiSymbolDataModule
    from .run_experiment import (
        _load_model,
        _resolve_device,
        evaluate_model,
        train_model,
    )
except ImportError:  # pragma: no cover - direct script execution
    from newnanoalpacahourlyexp.config import DatasetConfig, ExperimentConfig
    from newnanoalpacahourlyexp.data import AlpacaHourlyDataModule, AlpacaMultiSymbolDataModule
    from newnanoalpacahourlyexp.run_experiment import (
        _load_model,
        _resolve_device,
        evaluate_model,
        train_model,
    )


def _discover_symbols(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.stem.upper() for path in root.glob("*.csv") if path.is_file())


def discover_crossasset_symbols(
    *,
    crypto_root: Path = Path("trainingdatahourly/crypto"),
    stock_root: Path = Path("trainingdatahourly/stocks"),
) -> list[str]:
    seen: set[str] = set()
    symbols: list[str] = []
    for root in (crypto_root, stock_root):
        for symbol in _discover_symbols(root):
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    return symbols


def parse_symbol_list(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []

    parsed: list[str] = []
    seen: set[str] = set()
    normalized = str(raw).replace("\n", ",").replace(";", ",")
    for token in normalized.split(","):
        symbol = token.strip().upper()
        if not symbol or symbol == "ALL" or symbol in seen:
            continue
        seen.add(symbol)
        parsed.append(symbol)
    return parsed


def _resolve_pretrain_symbols(args: argparse.Namespace) -> list[str]:
    if args.pretrain_symbols and args.pretrain_symbols.strip().lower() != "all":
        return parse_symbol_list(args.pretrain_symbols)
    symbols = discover_crossasset_symbols(
        crypto_root=Path(args.crypto_data_root),
        stock_root=Path(args.stock_data_root),
    )
    if not symbols:
        raise ValueError("No symbols discovered under the provided crypto/stock hourly roots.")
    return symbols


def _build_data_module(symbols: Sequence[str], dataset_cfg: DatasetConfig):
    if len(symbols) > 1:
        return AlpacaMultiSymbolDataModule(symbols, dataset_cfg)
    return AlpacaHourlyDataModule(dataset_cfg)


def _single_symbol_config(base_cfg: DatasetConfig, symbol: str) -> DatasetConfig:
    return DatasetConfig(**{**base_cfg.__dict__, "symbol": symbol})


def filter_usable_symbols(
    symbols: Sequence[str],
    base_cfg: DatasetConfig,
) -> tuple[list[str], dict[str, str]]:
    usable: list[str] = []
    dropped: dict[str, str] = {}
    for symbol in symbols:
        try:
            AlpacaHourlyDataModule(_single_symbol_config(base_cfg, symbol))
        except Exception as exc:
            dropped[symbol] = str(exc)
            continue
        usable.append(symbol)
    return usable, dropped


def _phase_namespace(
    args: argparse.Namespace,
    *,
    run_name: str,
    epochs: int,
    dry_train_steps: Optional[int],
    preload_checkpoint_path: Optional[Path],
) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        feature_noise_std=args.feature_noise_std,
        run_name=run_name,
        dry_train_steps=dry_train_steps,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        maker_fee=args.maker_fee,
        periods_per_year=args.periods_per_year,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        dropout=args.dropout,
        model_arch=args.model_arch,
        attention_backend=args.attention_backend,
        flex_block_size=args.flex_block_size,
        num_memory_tokens=args.num_memory_tokens,
        dilated_strides=args.dilated_strides,
        attention_window=args.attention_window,
        preload_checkpoint_path=str(preload_checkpoint_path) if preload_checkpoint_path else None,
    )


def _parse_csv_ints(raw: Optional[str]) -> tuple[int, ...]:
    if not raw:
        return ()
    return tuple(int(token) for token in raw.split(",") if token.strip())


def _parse_csv_floats(raw: Optional[str]) -> tuple[float, ...]:
    if not raw:
        return ()
    return tuple(float(token) for token in raw.split(",") if token.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run broad cross-asset pretraining followed by per-symbol fine-tuning.")
    parser.add_argument("--symbol", default="ETHUSD", help="Default fine-tune symbol if --finetune-symbols is omitted.")
    parser.add_argument("--pretrain-symbols", default="all", help="'all' or a comma-separated symbol list.")
    parser.add_argument("--finetune-symbols", default=None, help="Comma-separated target symbols for phase-2 fine-tuning.")
    parser.add_argument("--crypto-data-root", default="trainingdatahourly/crypto")
    parser.add_argument("--stock-data-root", default="trainingdatahourly/stocks")
    parser.add_argument("--forecast-cache-root", default="binanceneural/forecast_cache")
    parser.add_argument("--forecast-horizons", default="1,4,12,24")
    parser.add_argument("--epochs-pretrain", type=int, default=6)
    parser.add_argument("--epochs-finetune", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--dry-train-steps-pretrain", type=int, default=None)
    parser.add_argument("--dry-train-steps-finetune", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default="alpaca_crossasset_multiphase")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maker-fee", type=float, default=None)
    parser.add_argument("--periods-per-year", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--loss-type", default="sortino_dd")
    parser.add_argument("--feature-noise-std", type=float, default=0.01)
    parser.add_argument("--model-arch", default="nano", choices=["classic", "nano"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--attention-backend",
        default="flex",
        choices=["auto", "sdpa", "flash", "flex", "external_flash_attn", "flash_attn_cute", "flash4"],
    )
    parser.add_argument("--flex-block-size", type=int, default=128)
    parser.add_argument("--num-memory-tokens", type=int, default=8)
    parser.add_argument("--dilated-strides", default="1,4,24,72")
    parser.add_argument("--attention-window", type=int, default=72)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--context-lengths", default="64,96,192")
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--blend-horizons", default=None)
    parser.add_argument("--blend-weights", default=None)
    parser.add_argument("--eval-days", type=float, default=10.0)
    parser.add_argument("--eval-hours", type=float, default=None)
    parser.add_argument("--allow-mixed-asset", action="store_true")
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--long-only-symbols", default=None)
    parser.add_argument("--short-only-symbols", default=None)
    parser.add_argument("--drop-unusable-symbols", action="store_true")
    parser.add_argument("--use-compile", action="store_true")
    parser.set_defaults(use_compile=True, allow_mixed_asset=True, drop_unusable_symbols=True)
    parser.add_argument("--no-use-compile", action="store_false", dest="use_compile")
    parser.add_argument("--no-allow-mixed-asset", action="store_false", dest="allow_mixed_asset")
    parser.add_argument("--no-drop-unusable-symbols", action="store_false", dest="drop_unusable_symbols")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrain_symbols = _resolve_pretrain_symbols(args)
    finetune_symbols = parse_symbol_list(args.finetune_symbols) or [args.symbol.upper()]
    pretrain_anchor = finetune_symbols[0] if finetune_symbols[0] in pretrain_symbols else pretrain_symbols[0]
    device = _resolve_device(args.device, symbol=pretrain_anchor)

    forecast_horizons = _parse_csv_ints(args.forecast_horizons) or DatasetConfig().forecast_horizons
    context_lengths = _parse_csv_ints(args.context_lengths) or ExperimentConfig().context_lengths
    blend_horizons = _parse_csv_ints(args.blend_horizons) or None
    blend_weights = _parse_csv_floats(args.blend_weights) or None
    experiment_cfg = ExperimentConfig(context_lengths=context_lengths, trim_ratio=args.trim_ratio)
    long_only_symbols = tuple(parse_symbol_list(args.long_only_symbols))
    short_only_symbols = tuple(parse_symbol_list(args.short_only_symbols))

    pretrain_cfg = DatasetConfig(
        symbol=pretrain_anchor,
        data_root=None,
        crypto_data_root=Path(args.crypto_data_root),
        stock_data_root=Path(args.stock_data_root),
        forecast_cache_root=Path(args.forecast_cache_root),
        forecast_horizons=forecast_horizons,
        sequence_length=args.sequence_length,
        allow_mixed_asset_class=args.allow_mixed_asset,
        allow_short=args.allow_short,
        long_only_symbols=long_only_symbols,
        short_only_symbols=short_only_symbols,
    )
    dropped_pretrain_symbols: dict[str, str] = {}
    if args.drop_unusable_symbols:
        pretrain_symbols, dropped_pretrain_symbols = filter_usable_symbols(pretrain_symbols, pretrain_cfg)
        if not pretrain_symbols:
            raise ValueError("No usable pretrain symbols remain after dataset validation.")
        if pretrain_anchor not in pretrain_symbols:
            pretrain_anchor = pretrain_symbols[0]
            pretrain_cfg.symbol = pretrain_anchor
    pretrain_data = _build_data_module(pretrain_symbols, pretrain_cfg)
    pretrain_args = _phase_namespace(
        args,
        run_name=f"{args.run_name}_pretrain",
        epochs=args.epochs_pretrain,
        dry_train_steps=args.dry_train_steps_pretrain,
        preload_checkpoint_path=None,
    )
    pretrain_artifacts = train_model(pretrain_data, pretrain_args, device=device)
    if pretrain_artifacts.best_checkpoint is None:
        raise RuntimeError("Pretraining did not produce a checkpoint.")

    manifest: dict[str, object] = {
        "run_name": args.run_name,
        "device": str(device),
        "pretrain_symbols": pretrain_symbols,
        "dropped_pretrain_symbols": dropped_pretrain_symbols,
        "finetune_symbols": finetune_symbols,
        "pretrain_checkpoint": str(pretrain_artifacts.best_checkpoint),
        "phases": [],
    }

    finetune_runs: list[dict[str, object]] = []
    for symbol in finetune_symbols:
        symbol_output_dir = output_dir / symbol.lower()
        symbol_output_dir.mkdir(parents=True, exist_ok=True)
        finetune_cfg = DatasetConfig(
            symbol=symbol,
            data_root=None,
            crypto_data_root=Path(args.crypto_data_root),
            stock_data_root=Path(args.stock_data_root),
            forecast_cache_root=Path(args.forecast_cache_root),
            forecast_horizons=forecast_horizons,
            sequence_length=args.sequence_length,
            allow_mixed_asset_class=args.allow_mixed_asset,
            allow_short=args.allow_short,
            long_only_symbols=long_only_symbols,
            short_only_symbols=short_only_symbols,
        )
        finetune_data = _build_data_module([symbol], finetune_cfg)
        finetune_args = _phase_namespace(
            args,
            run_name=f"{args.run_name}_finetune_{symbol.lower()}",
            epochs=args.epochs_finetune,
            dry_train_steps=args.dry_train_steps_finetune,
            preload_checkpoint_path=pretrain_artifacts.best_checkpoint,
        )
        finetune_artifacts = train_model(finetune_data, finetune_args, device=device)
        if finetune_artifacts.best_checkpoint is None:
            raise RuntimeError(f"Fine-tuning for {symbol} did not produce a checkpoint.")

        model = _load_model(
            finetune_artifacts.best_checkpoint,
            input_dim=len(finetune_data.feature_columns),
            sequence_length=args.sequence_length,
        )
        eval_result = evaluate_model(
            model=model,
            data=finetune_data,
            horizon=args.horizon,
            aggregate=args.aggregate,
            experiment_cfg=experiment_cfg,
            blend_horizons=blend_horizons,
            blend_weights=blend_weights,
            output_dir=symbol_output_dir,
            eval_days=args.eval_days,
            eval_hours=args.eval_hours,
            device=device,
        )
        if finetune_artifacts.history:
            history_payload = [entry.__dict__ for entry in finetune_artifacts.history]
            (symbol_output_dir / "training_history.json").write_text(json.dumps(history_payload, indent=2))

        finetune_runs.append(
            {
                "symbol": symbol,
                "checkpoint": str(finetune_artifacts.best_checkpoint),
                "total_return": eval_result.total_return,
                "sortino": eval_result.sortino,
                "output_dir": str(symbol_output_dir),
            }
        )

    manifest["phases"] = finetune_runs
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")
    for run in finetune_runs:
        print(
            f"{run['symbol']}: checkpoint={run['checkpoint']} "
            f"total_return={run['total_return']:.4f} sortino={run['sortino']:.4f}"
        )


if __name__ == "__main__":
    main()
