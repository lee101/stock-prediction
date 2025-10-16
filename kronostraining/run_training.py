from __future__ import annotations

import argparse
from pathlib import Path

from .config import KronosTrainingConfig
from .trainer import KronosTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Kronos on the local training dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("trainingdata"), help="Path to training CSV directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kronostraining") / "artifacts",
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument("--lookback", type=int, default=64, help="Historical window length.")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in timesteps.")
    parser.add_argument("--validation-days", type=int, default=30, help="Number of unseen days for validation metrics.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per step.")
    parser.add_argument("--learning-rate", type=float, default=4e-5, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--model-name", type=str, default="NeoQuasar/Kronos-small", help="Base Kronos model identifier.")
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="NeoQuasar/Kronos-Tokenizer-base",
        help="Tokenizer identifier to pair with the model.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--eval-samples", type=int, default=4, help="Autoregressive sample count for evaluation.")
    parser.add_argument("--device", type=str, default=None, help="Explicit torch device, e.g. cuda:0.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> KronosTrainingConfig:
    return KronosTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        lookback_window=args.lookback,
        prediction_length=args.horizon,
        validation_days=args.validation_days,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        eval_sample_count=args.eval_samples,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    trainer = KronosTrainer(config)
    summary = trainer.train()
    metrics = trainer.evaluate_holdout()

    print("\n[kronos] Training summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n[kronos] Validation aggregate metrics:")
    for key, value in metrics["aggregate"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
