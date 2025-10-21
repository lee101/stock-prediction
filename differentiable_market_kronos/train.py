from __future__ import annotations

import argparse
from pathlib import Path

from differentiable_market.config import DataConfig, EnvironmentConfig, EvaluationConfig, TrainingConfig

from .config import KronosFeatureConfig
from .trainer import DifferentiableMarketKronosTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Differentiable market trainer with Kronos embeddings")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdata"), help="Root directory of OHLC CSV files")
    parser.add_argument("--data-glob", type=str, default="*.csv", help="Glob pattern for CSV selection")
    parser.add_argument("--max-assets", type=int, default=None, help="Limit number of assets loaded")
    parser.add_argument("--exclude", type=str, nargs="*", default=(), help="Symbols to exclude")
    parser.add_argument("--lookback", type=int, default=192, help="Training lookback window")
    parser.add_argument("--batch-windows", type=int, default=64, help="Number of sampled windows per step")
    parser.add_argument("--rollout-groups", type=int, default=4, help="GRPO rollout group size")
    parser.add_argument("--epochs", type=int, default=2000, help="Training iterations")
    parser.add_argument("--eval-interval", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--save-dir", type=Path, default=Path("differentiable_market_kronos") / "runs", help="Run directory")
    parser.add_argument("--device", type=str, default="auto", help="Device override for policy training")
    parser.add_argument("--dtype", type=str, default="auto", help="dtype override for policy training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--include-cash", action="store_true", help="Append a zero-return cash asset")
    parser.add_argument("--no-muon", action="store_true", help="Disable Muon optimizer")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--microbatch-windows", type=int, default=None, help="Micro-batch window size")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable GRU gradient checkpointing")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Warm-start policy from checkpoint")
    parser.add_argument("--best-k-checkpoints", type=int, default=3, help="Number of top checkpoints to keep")

    # Kronos specific arguments
    parser.add_argument("--kronos-model", type=str, default="NeoQuasar/Kronos-small", help="Kronos model path or HF ID")
    parser.add_argument("--kronos-tokenizer", type=str, default="NeoQuasar/Kronos-Tokenizer-base", help="Kronos tokenizer path or HF ID")
    parser.add_argument("--kronos-context", type=int, default=192, help="Number of timesteps per Kronos embedding window")
    parser.add_argument("--kronos-clip", type=float, default=5.0, help="Clipping value applied during Kronos window normalisation")
    parser.add_argument("--kronos-batch", type=int, default=64, help="Batch size for Kronos embedding inference")
    parser.add_argument("--kronos-device", type=str, default="auto", help="Device for Kronos tokenizer/model")
    parser.add_argument(
        "--kronos-embedding-mode",
        type=str,
        choices=("context", "bits", "both"),
        default="context",
        help="Kronos feature representation to append to the market state",
    )
    parser.add_argument("--kronos-cache-dir", type=Path, default=None, help="Optional cache directory for Kronos embeddings")
    parser.add_argument(
        "--kronos-precision",
        type=str,
        choices=("float32", "bfloat16"),
        default="float32",
        help="Internal precision used for Kronos embeddings",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig(
        root=args.data_root,
        glob=args.data_glob,
        max_assets=args.max_assets,
        exclude_symbols=tuple(args.exclude),
        include_cash=args.include_cash,
    )
    env_cfg = EnvironmentConfig()
    train_cfg = TrainingConfig(
        lookback=args.lookback,
        batch_windows=args.batch_windows,
        rollout_groups=args.rollout_groups,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        use_muon=not args.no_muon,
        use_compile=not args.no_compile,
        microbatch_windows=args.microbatch_windows,
        gradient_checkpointing=args.gradient_checkpointing,
        include_cash=args.include_cash,
        init_checkpoint=args.init_checkpoint,
        best_k_checkpoints=max(1, args.best_k_checkpoints),
    )
    eval_cfg = EvaluationConfig(report_dir=Path("differentiable_market_kronos") / "evals")
    kronos_cfg = KronosFeatureConfig(
        model_path=args.kronos_model,
        tokenizer_path=args.kronos_tokenizer,
        context_length=args.kronos_context,
        clip=args.kronos_clip,
        batch_size=args.kronos_batch,
        device=args.kronos_device,
        embedding_mode=args.kronos_embedding_mode,
        cache_dir=args.kronos_cache_dir,
        precision=args.kronos_precision,
    )

    trainer = DifferentiableMarketKronosTrainer(data_cfg, env_cfg, train_cfg, eval_cfg, kronos_cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
