from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np

from wandboard import WandBoardLogger

from . import current_symbols
from .data import build_dataset, load_daily_metrics, split_dataset_by_date
from .feature_builder import FeatureSpec
from .trade_windows import load_trade_window_metrics
from .trainer import (
    evaluate_policy,
    train_sortino_policy,
    train_xgboost_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train neural + XGBoost portfolio optimisers on strategytraining metrics."
    )
    parser.add_argument(
        "--metrics-csv",
        default="strategytraining/reports/sizing_strategy_daily_metrics.csv",
        help="Path to the sizing_strategy_daily_metrics.csv artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="strategytrainingneural/reports",
        help="Directory used for model checkpoints and metrics.",
    )
    parser.add_argument(
        "--start-date", default=None, help="Optional ISO date string to filter the dataset."
    )
    parser.add_argument(
        "--end-date", default=None, help="Optional ISO date string to filter the dataset."
    )
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        help="Limit training to the provided strategies (can be specified multiple times).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation (chronological split).",
    )
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs for the neural model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the neural model.")
    parser.add_argument(
        "--return-weight",
        type=float,
        default=0.05,
        help="Weight applied to the annualised return component of the objective.",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Enable symmetric weights via tanh activation for the neural model.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=1.0,
        help="Maximum absolute trade weight output by the neural model.",
    )
    parser.add_argument(
        "--trade-parquet",
        action="append",
        default=[],
        help="Optional parquet glob(s) for trade-level datasets (10-day style windows).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for neural training (e.g., cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=10,
        help="Rolling window (days) for trade-based datasets.",
    )
    parser.add_argument(
        "--asset-class",
        choices=("all", "stock", "crypto"),
        default="all",
        help="Restrict trade datasets to one asset class.",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Limit trade datasets to the provided symbols (can repeat).",
    )
    parser.add_argument(
        "--use-trade-script-symbols",
        action="store_true",
        help="Automatically pull the symbol list from trade_stock_e2e.py for trade datasets.",
    )
    parser.add_argument(
        "--trade-script-path",
        default="trade_stock_e2e.py",
        help="Path to the live trade script for symbol extraction.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=5,
        help="Minimum number of trades per symbol/strategy required when building trade datasets.",
    )
    parser.add_argument(
        "--forecast-cache",
        help="Optional directory of cached chronos2 forecasts to merge into trade datasets.",
    )
    parser.add_argument(
        "--daily-asset-class",
        choices=("all", "stock", "crypto"),
        default="all",
        help="Filter daily metrics to a specific asset class (uses day_class column).",
    )
    parser.add_argument(
        "--strategy-sample-fraction",
        type=float,
        default=1.0,
        help="Randomly keep this fraction of unique strategies for robustness training.",
    )
    parser.add_argument(
        "--strategy-sample-seed",
        type=int,
        default=0,
        help="Seed used when subsampling strategy ids.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional custom run name for WandB/TensorBoard tracking.",
    )
    parser.add_argument(
        "--resume-run-dir",
        help="Existing run directory containing sortino_policy.pt used to warm-start training.",
    )
    return parser.parse_args()


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _resolve_symbols(args: argparse.Namespace) -> Optional[List[str]]:
    base: Optional[List[str]] = None
    if args.use_trade_script_symbols:
        base = current_symbols.load_current_symbols(args.trade_script_path)
    if args.symbols:
        include = {s.upper() for s in args.symbols}
        if base is None:
            base = [s for s in include]
        else:
            base = [s for s in base if s.upper() in include]
    return base


def main() -> None:
    args = parse_args()
    symbol_subset: Optional[List[str]] = None
    if args.trade_parquet:
        symbols = _resolve_symbols(args)
        symbol_subset = symbols
        df = load_trade_window_metrics(
            args.trade_parquet,
            symbols=symbols,
            asset_class=args.asset_class,
            window_days=args.window_days,
            min_trades=args.min_trades,
            forecast_cache_dir=args.forecast_cache,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        dataset_source = "trade_windows"
    else:
        df = load_daily_metrics(
            args.metrics_csv,
            strategy_filter=args.strategies,
            start_date=args.start_date,
            end_date=args.end_date,
            asset_class_filter=args.daily_asset_class,
        )
        dataset_source = "daily_metrics"
        if 0.0 < args.strategy_sample_fraction < 1.0:
            rng = np.random.default_rng(args.strategy_sample_seed)
            strategies = sorted(df["strategy"].unique())
            sample_n = max(1, int(len(strategies) * args.strategy_sample_fraction))
            chosen = set(rng.choice(strategies, size=sample_n, replace=False))
            df = df[df["strategy"].isin(chosen)].reset_index(drop=True)
            print(f"[dataset] sampled {len(chosen)} strategies out of {len(strategies)}")
    resume_state = None
    resume_dir: Optional[Path] = None
    resume_spec: Optional[FeatureSpec] = None
    if args.resume_run_dir:
        resume_dir = Path(args.resume_run_dir)
        state_path = resume_dir / "sortino_policy.pt"
        spec_source = resume_dir / "feature_spec.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing sortino_policy.pt in {resume_dir}")
        if not spec_source.exists():
            raise FileNotFoundError(f"Missing feature_spec.json in {resume_dir}")
        resume_state = torch.load(state_path, map_location="cpu")
        resume_spec = FeatureSpec.from_dict(json.loads(spec_source.read_text()))
        print(f"[neural] warm starting from {resume_dir}")

    dataset = build_dataset(df, feature_spec=resume_spec)
    train_ds, val_ds = split_dataset_by_date(dataset, validation_fraction=args.val_fraction)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_path = output_dir / "feature_spec.json"
    save_json(spec_path, dataset.feature_spec.to_dict())

    logger_config = {
        **{k: v for k, v in vars(args).items() if v is not None},
        "num_rows": len(dataset.frame),
        "num_train_rows": len(train_ds.frame),
        "num_val_rows": len(val_ds.frame),
        "strategies": sorted(dataset.strategy_vocab.keys()),
        "dataset_source": dataset_source,
    }
    if resume_dir is not None:
        logger_config["resume_run_dir"] = str(resume_dir)
    with WandBoardLogger(
        project="strategytraining-neural",
        run_name=args.run_name,
        group="strategytraining",
        tags=("strategytrainingneural",),
        config=logger_config,
        log_metrics=True,
        notes="Neural + XGBoost joint Sortino training on strategytraining metrics.",
    ) as run_logger:
        initial_metrics = {
            "dataset/train_rows": len(train_ds.frame),
            "dataset/val_rows": len(val_ds.frame),
            "dataset/num_strategies": len(dataset.strategy_vocab),
        }
        if symbol_subset:
            initial_metrics["dataset/num_symbols"] = len(symbol_subset)
        run_logger.log(initial_metrics, step=0)

        print(f"[neural] training on {len(train_ds.frame)} rows, validating on {len(val_ds.frame)} rows")
        neural_result = train_sortino_policy(
            train_ds,
            validation_dataset=val_ds,
            epochs=args.epochs,
            learning_rate=args.lr,
            return_weight=args.return_weight,
            allow_short=args.allow_short,
            max_weight=args.max_weight,
            device=args.device,
            logger=run_logger,
            initial_state_dict=resume_state,
        )
        history_payload = [
            entry.__dict__
            for entry in neural_result.history
        ]
        save_json(output_dir / "neural_history.json", history_payload)
        save_json(output_dir / "neural_metrics.json", neural_result.final_metrics)
        torch.save(neural_result.model.state_dict(), output_dir / "sortino_policy.pt")

        print(
            f"[neural] final validation score={neural_result.final_metrics['score']:.3f} "
            f"sortino={neural_result.final_metrics['sortino']:.3f} "
            f"ann_return={neural_result.final_metrics['annual_return']:.3f}"
        )

        print("[xgboost] training boosted baseline...")
        xgb_result = train_xgboost_policy(
            train_ds,
            evaluation_dataset=val_ds,
            return_weight=args.return_weight,
            logger=run_logger,
        )
        xgb_payload = {
            "temperature": xgb_result.temperature,
            "score": xgb_result.score,
            "sortino": xgb_result.sortino,
            "annual_return": xgb_result.annual_return,
        }
        save_json(output_dir / "xgboost_metrics.json", xgb_payload)
        xgb_result.booster.save_model(str(output_dir / "xgboost_policy.json"))
        print(
            f"[xgboost] best temperature={xgb_result.temperature:.3f} "
            f"score={xgb_result.score:.3f} sortino={xgb_result.sortino:.3f} "
            f"ann_return={xgb_result.annual_return:.3f}"
        )
        run_logger.log(
            {
                "comparison/neural_score": neural_result.final_metrics["score"],
                "comparison/xgboost_score": xgb_result.score,
            },
            step=args.epochs + train_ds.features.shape[0],
        )


if __name__ == "__main__":
    main()
