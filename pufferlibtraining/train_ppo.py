#!/usr/bin/env python3
"""
Multi-stage RL training pipeline for stock trading using Amazon Toto forecasts.

The pipeline consists of three stages:
    1. Generic forecaster training on all available stocks.
    2. Specialist fine-tuning per target stock.
    3. Differentiable portfolio allocation RL over stock pairs.

This script orchestrates the stages using the hftraining modules so that the
resulting models can be consumed by the vectorised PufferLib environments.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.leverage_settings import get_leverage_settings

try:  # Defer heavy hftraining imports until the optional extras are installed.
    from hftraining.base_model_trainer import BaseModelTrainer, PortfolioRLConfig
    from hftraining.toto_features import TotoOptions
except Exception as exc:  # pragma: no cover - triggered when extras missing
    BaseModelTrainer = None  # type: ignore[assignment]
    PortfolioRLConfig = None  # type: ignore[assignment]
    TotoOptions = None  # type: ignore[assignment]
    _HFTRAINING_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - exercised when extras present
    _HFTRAINING_IMPORT_ERROR = None


LOGGER = logging.getLogger("pufferlibtraining.pipeline")


def _parse_symbol_list(raw: str, field: str) -> List[str]:
    tokens = [tok.strip().upper() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(f"{field} must contain at least one symbol")
    return tokens


def _parse_pairs(pair_args: Sequence[str], fallback: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for raw in pair_args:
        cleaned = raw.replace("/", ",")
        bits = [b.strip().upper() for b in cleaned.split(",") if b.strip()]
        if len(bits) != 2:
            raise argparse.ArgumentTypeError(
                f"Unable to parse pair '{raw}'. Use SYMBOL_A,SYMBOL_B or SYMBOL_A/SYMBOL_B."
            )
        pairs.append((bits[0], bits[1]))

    if not pairs:
        fallback = [sym.upper() for sym in fallback]
        pairs = [
            (fallback[i], fallback[i + 1])
            for i in range(len(fallback) - 1)
        ]
    return pairs


def _ensure_data(symbols: Iterable[str], data_root: Path) -> None:
    missing: List[str] = []
    for sym in symbols:
        matches = list(data_root.rglob(f"*{sym.upper()}*.csv"))
        if not matches:
            missing.append(sym.upper())
    if missing:
        raise FileNotFoundError(
            f"No CSV data found under '{data_root}' for symbols: {', '.join(sorted(set(missing)))}"
        )


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def sync_vecnormalize_stats(source: Any, target: Any) -> None:
    """Copy running normalisation stats from ``source`` into ``target`` if available.

    Some pipelines reuse a pre-trained VecNormalize wrapper so downstream
    environments and rollout storage share identical observation/return scaling.
    This helper makes that copy while ensuring the target stays in eval mode.
    """
    copied = False
    for attr in ("obs_rms", "ret_rms"):
        if hasattr(source, attr) and hasattr(target, attr):
            setattr(target, attr, getattr(source, attr))
            copied = True

    if not copied:
        return

    if hasattr(target, "set_training_mode"):
        set_training_mode = getattr(target, "set_training_mode")
        if callable(set_training_mode):
            set_training_mode(False)

    if hasattr(target, "training"):
        target.training = False


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full multi-stage Toto-enhanced RL training pipeline."
    )

    leverage_defaults = get_leverage_settings()

    parser.add_argument(
        "--base-stocks",
        type=str,
        default="AAPL,AMZN,MSFT,NVDA,GOOGL",
        help="Comma-separated list of symbols used for the generic forecaster.",
    )
    parser.add_argument(
        "--specialist-stocks",
        type=str,
        default="AAPL,AMZN,MSFT",
        help="Comma-separated list of symbols that receive specialist fine-tuning.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Specify a stock pair (SYMBOL_A,SYMBOL_B). Repeat for multiple pairs. "
             "Defaults to adjacent pairs from --specialist-stocks.",
    )

    parser.add_argument(
        "--trainingdata-dir",
        type=str,
        default="trainingdata",
        help="Directory containing raw CSVs for Toto feature generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pufferlibtraining/models",
        help="Directory where checkpoints will be written.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="pufferlibtraining/logs",
        help="Directory where TensorBoard logs will be stored.",
    )

    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip generic forecaster training (requires --base-checkpoint).",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="",
        help="Path to an existing base checkpoint when skipping stage 1.",
    )
    parser.add_argument(
        "--base-steps",
        type=int,
        default=20000,
        help="Maximum optimisation steps for the base model.",
    )
    parser.add_argument(
        "--base-batch-size",
        type=int,
        default=48,
        help="Batch size for the base model trainer.",
    )
    parser.add_argument(
        "--base-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the generic forecaster.",
    )
    parser.add_argument(
        "--progressive-base-steps",
        type=str,
        default="",
        help="Comma-separated schedule of additional base-training step increments (enables staged training).",
    )

    parser.add_argument(
        "--skip-specialists",
        action="store_true",
        help="Skip per-stock specialist fine-tuning.",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=8,
        help="Number of epochs for each specialist fine-tuning run.",
    )
    parser.add_argument(
        "--finetune-learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate used for specialist fine-tuning.",
    )

    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip differentiable portfolio RL training.",
    )
    parser.add_argument(
        "--rl-epochs",
        type=int,
        default=20,
        help="Epochs for portfolio RL optimisation.",
    )
    parser.add_argument(
        "--rl-hidden-size",
        type=int,
        default=256,
        help="Hidden dimension for the allocation transformer.",
    )
    parser.add_argument(
        "--rl-num-layers",
        type=int,
        default=4,
        help="Number of transformer encoder layers in the allocation model.",
    )
    parser.add_argument(
        "--rl-num-heads",
        type=int,
        default=4,
        help="Number of attention heads for the allocation model.",
    )
    parser.add_argument(
        "--rl-learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for the differentiable portfolio trainer.",
    )
    parser.add_argument(
        "--rl-batch-size",
        type=int,
        default=128,
        help="Batch size for portfolio RL.",
    )
    parser.add_argument(
        "--rl-optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "lion", "adan", "shampoo", "adafactor", "muon", "muon_mix"],
        help="Optimizer used for the allocation transformer.",
    )
    parser.add_argument(
        "--rl-weight-decay",
        type=float,
        default=0.01,
        help="Weight decay applied to RL optimiser parameter groups.",
    )
    parser.add_argument(
        "--rl-warmup-steps",
        type=int,
        default=500,
        help="Warmup steps for the RL cosine schedule.",
    )
    parser.add_argument(
        "--rl-min-lr",
        type=float,
        default=0.0,
        help="Minimum learning rate for the RL cosine schedule.",
    )
    parser.add_argument(
        "--rl-initial-checkpoint-dir",
        type=str,
        default="",
        help="Optional directory containing per-pair RL checkpoints (<SYMA>_<SYMB>_portfolio_best.pt) to resume from.",
    )
    parser.add_argument(
        "--rl-grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm for RL training.",
    )
    parser.add_argument(
        "--rl-no-compile",
        action="store_true",
        help="Disable torch.compile for the allocation transformer.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
        help="Transaction cost used in differentiable profit calculations (basis points).",
    )
    parser.add_argument(
        "--risk-penalty",
        type=float,
        default=0.1,
        help="Sharpe-based risk penalty coefficient.",
    )
    parser.add_argument(
        "--leverage-limit",
        type=float,
        default=leverage_defaults.max_gross_leverage,
        help="Maximum gross exposure for the RL allocation head.",
    )
    parser.add_argument(
        "--borrowing-cost",
        type=float,
        default=leverage_defaults.annual_cost,
        help="Annualised borrowing cost applied to leverage above 1×.",
    )
    parser.add_argument(
        "--trading-days-per-year",
        type=int,
        default=leverage_defaults.trading_days_per_year,
        help="Trading days per year used to annualise borrowing cost.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if TotoOptions().toto_device == "cuda" else "cpu",
        help="Torch device string for Toto forecasts and RL training.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Context window length used for Toto conditioning.",
    )
    parser.add_argument(
        "--toto-horizon",
        type=int,
        default=8,
        help="Prediction horizon (in steps) for Toto forecasts.",
    )
    parser.add_argument(
        "--toto-num-samples",
        type=int,
        default=2048,
        help="Number of forecast samples when using the Toto backend (default: 2048).",
    )
    parser.add_argument(
        "--toto-model-id",
        type=str,
        default="Datadog/Toto-Open-Base-1.0",
        help="Model identifier for Toto feature generation.",
    )
    parser.add_argument(
        "--toto-targets",
        type=str,
        default="close,open,high,low",
        help="Comma-separated list of price lines to feed into Toto forecasts (e.g., close,high,low).",
    )
    parser.add_argument(
        "--toto-predictions-dir",
        type=str,
        default="",
        help="Directory containing historical Toto strategy CSVs to use as additional features.",
    )
    parser.add_argument(
        "--disable-toto",
        action="store_true",
        help="Disable Toto feature generation (falls back to statistical forecasts).",
    )

    parser.add_argument(
        "--summary-path",
        type=str,
        default="pufferlibtraining/models/pipeline_summary.json",
        help="Location to write the training summary JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    return parser


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    if _HFTRAINING_IMPORT_ERROR is not None:
        raise ImportError(
            "hftraining optional dependencies are unavailable. Install with `uv pip sync --extra hf --extra rl --extra mlops` "
            "or install the workspace package via `uv pip install -e pufferlibtraining`."
        ) from _HFTRAINING_IMPORT_ERROR

    data_root = Path(args.trainingdata_dir).expanduser().resolve()
    if not data_root.exists():
        fallback = Path("tototraining") / "trainingdata" / "train"
        if fallback.exists():
            LOGGER.warning(
                "Training data directory '%s' not found. Falling back to '%s'.",
                data_root,
                fallback.resolve(),
            )
            data_root = fallback.resolve()
        else:
            raise FileNotFoundError(f"Training data directory '{data_root}' does not exist")

    base_symbols = _parse_symbol_list(args.base_stocks, "base-stocks")
    specialist_symbols = _parse_symbol_list(args.specialist_stocks, "specialist-stocks")
    all_symbols = sorted(set(base_symbols) | set(specialist_symbols))
    pairs = _parse_pairs(args.pair, specialist_symbols)

    _ensure_data(all_symbols, data_root)

    output_dir = Path(args.output_dir).expanduser().resolve()
    tensorboard_dir = Path(args.tensorboard_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    toto_targets = [t.strip() for t in args.toto_targets.split(",") if t.strip()]
    toto_opts = TotoOptions(
        use_toto=not args.disable_toto,
        horizon=args.toto_horizon,
        context_length=args.sequence_length,
        num_samples=args.toto_num_samples,
        toto_model_id=args.toto_model_id,
        toto_device=args.device,
        target_columns=toto_targets or ("close",),
    )

    predictions_dir: Optional[Path] = None
    if args.toto_predictions_dir:
        predictions_dir = Path(args.toto_predictions_dir).expanduser().resolve()

    LOGGER.info("Initialising BaseModelTrainer with Toto options: %s", toto_opts)
    trainer = BaseModelTrainer(
        base_stocks=base_symbols,
        output_dir=str(output_dir),
        tensorboard_dir=str(tensorboard_dir),
        use_toto_forecasts=not args.disable_toto,
        toto_options=toto_opts,
        data_dir=str(data_root),
        toto_predictions_dir=str(predictions_dir) if predictions_dir else None,
    )

    progressive_schedule: Optional[List[int]] = None

    summary: Dict[str, object] = {
        "base_checkpoint": None,
        "specialists": {},
        "portfolio_pairs": {},
        "config": {
            "base_symbols": base_symbols,
            "specialist_symbols": specialist_symbols,
            "pairs": [list(pair) for pair in pairs],
            "trainingdata_dir": str(data_root),
            "output_dir": str(output_dir),
            "tensorboard_dir": str(tensorboard_dir),
            "toto": {
                "enabled": not args.disable_toto,
                "horizon": args.toto_horizon,
                "context_length": args.sequence_length,
                "num_samples": args.toto_num_samples,
                "model_id": args.toto_model_id,
                "targets": toto_targets or ("close",),
                "predictions_dir": str(predictions_dir) if predictions_dir else None,
            },
            "rl": {
                "transaction_cost_bps": args.transaction_cost_bps,
                "risk_penalty": args.risk_penalty,
                "leverage_limit": args.leverage_limit,
                "borrowing_cost": args.borrowing_cost,
                "trading_days_per_year": args.trading_days_per_year,
                "initial_checkpoint_dir": args.rl_initial_checkpoint_dir,
            },
            "progressive_base_steps": progressive_schedule,
        },
    }

    base_checkpoint_path: Path
    if args.skip_base:
        if not args.base_checkpoint:
            raise ValueError("--skip-base requires --base-checkpoint to be set")
        base_checkpoint_path = Path(args.base_checkpoint).expanduser().resolve()
        if not base_checkpoint_path.exists():
            raise FileNotFoundError(f"Base checkpoint '{base_checkpoint_path}' not found")
        summary["base_checkpoint"] = str(base_checkpoint_path)
        LOGGER.info("Skipping base stage; using checkpoint %s", base_checkpoint_path)
        processor_path = base_checkpoint_path.parent / "data_processor.pkl"
        if processor_path.exists():
            try:
                trainer.processor.load_scalers(str(processor_path))
                LOGGER.info("Loaded data processor scalers from %s", processor_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to load processor scalers from %s: %s", processor_path, exc)
    else:
        LOGGER.info("Stage 1: training generic forecaster on %d symbols", len(base_symbols))
        progressive_schedule = (
            [int(val) for val in args.progressive_base_steps.split(",") if val.strip()]
            if args.progressive_base_steps
            else None
        )
        summary["config"]["progressive_base_steps"] = progressive_schedule

        model, checkpoint = trainer.train_base_model(
            max_steps=args.base_steps,
            batch_size=args.base_batch_size,
            learning_rate=args.base_learning_rate,
            progressive_schedule=progressive_schedule,
        )
        del model  # Model stays on disk; freeing memory
        base_checkpoint_path = Path(checkpoint)
        summary["base_checkpoint"] = checkpoint
        LOGGER.info("Base checkpoint saved to %s", checkpoint)

    if not args.skip_specialists:
        LOGGER.info("Stage 2: specialist fine-tuning for %d symbols", len(specialist_symbols))
        specialist_paths: Dict[str, str] = {}
        for symbol in specialist_symbols:
            LOGGER.info("Fine-tuning specialist for %s", symbol)
            model, path = trainer.finetune_for_stock(
                stock_symbol=symbol,
                base_checkpoint_path=str(base_checkpoint_path),
                num_epochs=args.finetune_epochs,
                learning_rate=args.finetune_learning_rate,
            )
            if model is None or path is None:
                LOGGER.warning("Specialist training for %s skipped or failed", symbol)
                continue
            del model
            specialist_paths[symbol] = path
            LOGGER.info("Specialist checkpoint for %s saved to %s", symbol, path)
        summary["specialists"] = specialist_paths
    else:
        LOGGER.info("Skipping specialist fine-tuning stage")

    if not args.skip_rl:
        LOGGER.info("Stage 3: differentiable portfolio RL on %d pairs", len(pairs))
        rl_config = PortfolioRLConfig(
            hidden_size=args.rl_hidden_size,
            num_layers=args.rl_num_layers,
            num_heads=args.rl_num_heads,
            dropout=0.1,
            learning_rate=args.rl_learning_rate,
            batch_size=args.rl_batch_size,
            epochs=args.rl_epochs,
            transaction_cost_bps=args.transaction_cost_bps,
            risk_penalty=args.risk_penalty,
            device=args.device,
            leverage_limit=args.leverage_limit,
            borrowing_cost=args.borrowing_cost,
            trading_days_per_year=args.trading_days_per_year,
            optimizer=args.rl_optimizer,
            weight_decay=args.rl_weight_decay,
            compile=not args.rl_no_compile,
            grad_clip=args.rl_grad_clip,
            warmup_steps=args.rl_warmup_steps,
            min_learning_rate=args.rl_min_lr,
        )
        pair_metrics: Dict[str, Dict[str, float]] = {}
        initial_dir = Path(args.rl_initial_checkpoint_dir).expanduser().resolve() if args.rl_initial_checkpoint_dir else None
        for sym_a, sym_b in pairs:
            LOGGER.info("Training allocation policy for pair %s/%s", sym_a, sym_b)
            init_ckpt = None
            if initial_dir:
                candidate = initial_dir / f"{sym_a}_{sym_b}_portfolio_best.pt"
                if candidate.exists():
                    init_ckpt = candidate
                    LOGGER.info("Resuming pair %s/%s from %s", sym_a, sym_b, candidate)
            metrics = trainer.train_pair_portfolio((sym_a, sym_b), rl_config=rl_config, initial_checkpoint=init_ckpt)
            pair_metrics[f"{sym_a}_{sym_b}"] = metrics
            LOGGER.info("Pair %s/%s metrics: %s", sym_a, sym_b, metrics)
        summary["portfolio_pairs"] = pair_metrics
    else:
        LOGGER.info("Skipping portfolio RL stage")

    return summary


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    try:
        summary = run_pipeline(args)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    summary_path = Path(args.summary_path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Pipeline summary written to %s", summary_path)
    LOGGER.info("Stages complete: base=%s, specialists=%s, portfolio=%s",
                not args.skip_base, not args.skip_specialists, not args.skip_rl)


if __name__ == "__main__":
    main()
