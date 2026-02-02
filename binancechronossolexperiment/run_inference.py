from __future__ import annotations

import argparse
from pathlib import Path

from binanceneural.inference import generate_latest_action

from .data import ChronosSolDataModule, SplitConfig
from .inference import load_policy_checkpoint

DEFAULT_MODEL_ID = "chronos2_finetuned/SOLUSDT_lora_20260202_030749/finetuned-ckpt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate latest SOLUSDT action from trained policy")
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--forecast-cache-root", default="binancechronossolexperiment/forecast_cache")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--context-hours", type=int, default=3072)
    parser.add_argument("--chronos-batch-size", type=int, default=32)
    parser.add_argument("--horizons", default="1")
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())
    if not horizons:
        raise ValueError("At least one horizon must be provided")

    data_module = ChronosSolDataModule(
        symbol=args.symbol,
        data_root=Path(args.data_root),
        forecast_cache_root=Path(args.forecast_cache_root),
        forecast_horizons=horizons,
        context_hours=args.context_hours,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=args.chronos_batch_size,
        model_id=args.model_id,
        sequence_length=args.sequence_length,
        split_config=SplitConfig(val_days=1, test_days=1),
        cache_only=args.cache_only,
    )

    model, normalizer, feature_columns, _ = load_policy_checkpoint(args.checkpoint)
    frame = data_module.full_frame.copy()
    action = generate_latest_action(
        model=model,
        frame=frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=args.sequence_length,
        horizon=horizons[0],
    )
    print(action)


if __name__ == "__main__":
    main()
