from __future__ import annotations

import argparse
from pathlib import Path

from differentiable_market.config import DataConfig, EnvironmentConfig, EvaluationConfig, TrainingConfig

from .config import KronosFeatureConfig
from .trainer import DifferentiableMarketKronosTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Differentiable market trainer with Kronos summaries")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    parser.add_argument("--data-glob", type=str, default="*.csv")
    parser.add_argument("--max-assets", type=int, default=None)
    parser.add_argument("--symbols", type=str, nargs="*", default=None)
    parser.add_argument("--exclude", type=str, nargs="*", default=())
    parser.add_argument("--min-timesteps", type=int, default=512)
    parser.add_argument("--lookback", type=int, default=192)
    parser.add_argument("--batch-windows", type=int, default=64)
    parser.add_argument("--rollout-groups", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-dir", type=Path, default=Path("differentiable_market_kronos") / "runs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-cash", action="store_true")
    parser.add_argument("--no-muon", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--microbatch-windows", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--best-k-checkpoints", type=int, default=3)

    parser.add_argument("--kronos-model", type=str, default="NeoQuasar/Kronos-small")
    parser.add_argument("--kronos-tokenizer", type=str, default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--kronos-context", type=int, default=256)
    parser.add_argument("--kronos-horizons", type=int, nargs="*", default=(1, 12, 48))
    parser.add_argument("--kronos-quantiles", type=float, nargs="*", default=(0.1, 0.5, 0.9))
    parser.add_argument("--kronos-sample-count", type=int, default=16)
    parser.add_argument("--kronos-sample-chunk", type=int, default=32)
    parser.add_argument("--kronos-temperature", type=float, default=1.0)
    parser.add_argument("--kronos-top-p", type=float, default=0.9)
    parser.add_argument("--kronos-top-k", type=int, default=0)
    parser.add_argument("--kronos-clip", type=float, default=2.0)
    parser.add_argument("--kronos-device", type=str, default="auto")
    parser.add_argument("--kronos-disable-path-stats", action="store_true")
    parser.add_argument("--kronos-no-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig(
        root=args.data_root,
        glob=args.data_glob,
        max_assets=args.max_assets,
        include_symbols=tuple(args.symbols or ()),
        exclude_symbols=tuple(args.exclude),
        include_cash=args.include_cash,
        min_timesteps=args.min_timesteps,
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
        horizons=tuple(args.kronos_horizons),
        quantiles=tuple(args.kronos_quantiles),
        include_path_stats=not args.kronos_disable_path_stats,
        device=args.kronos_device,
        sample_count=args.kronos_sample_count,
        sample_chunk=args.kronos_sample_chunk,
        temperature=args.kronos_temperature,
        top_p=args.kronos_top_p,
        top_k=args.kronos_top_k,
        clip=args.kronos_clip,
        bf16=not args.kronos_no_bf16,
    )

    trainer = DifferentiableMarketKronosTrainer(data_cfg, env_cfg, train_cfg, eval_cfg, kronos_cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
