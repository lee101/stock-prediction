from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wandboard import WandBoardLogger

from .data import build_pricing_dataset, load_backtest_frames, split_dataset_by_date
from .trainer import PricingTrainingConfig, train_pricing_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the neural pricing strategy model.")
    parser.add_argument(
        "--backtest-csv",
        dest="backtest_csv",
        action="append",
        default=["backtest_results/*_backtest.csv"],
        help="Glob(s) pointing at *_backtest.csv exports.",
    )
    parser.add_argument(
        "--output-dir",
        default="neuralpricingstrategy/reports",
        help="Directory to store training artifacts.",
    )
    parser.add_argument("--start-date", help="Optional ISO date to filter training dataset.")
    parser.add_argument("--end-date", help="Optional ISO date to filter training dataset.")
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Restrict dataset to these symbols (can repeat).",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pnl-weight", type=float, default=0.2, help="Loss weight for pnl gain head.")
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--clamp-pct", type=float, default=0.08, help="Price delta clamp percentage.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of most recent days reserved for validation.",
    )
    parser.add_argument("--device", default=None, help="Torch device override (cpu, cuda:0, ...).")
    parser.add_argument("--run-name", help="Optional custom experiment name for WandBoard.")
    return parser.parse_args()


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = _parse_args()
    frames = load_backtest_frames(
        args.backtest_csv,
        symbol_filter=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    dataset = build_pricing_dataset(frames, clamp_pct=args.clamp_pct)
    train_ds, val_ds = split_dataset_by_date(dataset, validation_fraction=args.val_fraction)

    config = PricingTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pnl_weight=args.pnl_weight,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    with WandBoardLogger(
        project="neuralpricingstrategy",
        run_name=args.run_name,
        group="neural-pricing",
        tags=("neuralpricingstrategy",),
        log_metrics=True,
        config={
            "dataset/rows": len(dataset.frame),
            "dataset/clamp_pct": args.clamp_pct,
            "dataset/val_fraction": args.val_fraction,
            "symbols": args.symbols or "all",
        },
    ) as logger:
        run_dir = output_root / logger.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        result = train_pricing_model(train_ds, validation_dataset=val_ds, config=config, logger=logger)

        torch.save(result.model.state_dict(), run_dir / "pricing_model.pt")
        _save_json(run_dir / "feature_spec.json", dataset.feature_spec.to_dict())
        _save_json(run_dir / "run_config.json", {"clamp_pct": dataset.clamp_pct})
        _save_json(run_dir / "training_history.json", [entry.__dict__ for entry in result.history])
        _save_json(run_dir / "training_metrics.json", result.final_metrics)

        logger.log(
            {
                "artifacts/run_dir": str(run_dir),
                **result.final_metrics,
            },
            step=config.epochs,
        )
        print(f"[neuralpricing] run saved to {run_dir}")


if __name__ == "__main__":
    main()
