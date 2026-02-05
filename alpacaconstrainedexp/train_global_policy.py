from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from src.torch_device_utils import require_cuda as require_cuda_device

from newnanoalpacahourlyexp.config import DatasetConfig, ExperimentConfig
from newnanoalpacahourlyexp.data import AlpacaMultiSymbolDataModule
from newnanoalpacahourlyexp.run_experiment import evaluate_model, train_model, _load_model

from .symbols import build_longable_symbols, build_shortable_symbols, normalize_symbols


def _parse_symbols(raw: str | None) -> List[str]:
    if raw is None:
        return []
    symbols = normalize_symbols([token for token in raw.split(",") if token.strip()])
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a global policy across constrained symbols (Alpaca hourly).")
    parser.add_argument("--symbols", default=None)
    parser.add_argument(
        "--long-stocks",
        default=None,
        help="Comma-separated longable stock symbols (default: NVDA,GOOG,MSFT).",
    )
    parser.add_argument(
        "--short-stocks",
        default=None,
        help="Comma-separated shortable stock symbols (default: YELP,EBAY,TRIP,MTCH,KIND,ANGI,Z,EXPE,BKNG,NWSA,NYT).",
    )
    parser.add_argument(
        "--crypto",
        default=None,
        help="Comma-separated crypto symbols to include (default: BTCUSD,ETHUSD,SOLUSD).",
    )
    parser.add_argument("--target-symbol", default=None, help="Symbol to use for validation metrics.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--dry-train-steps", type=int, default=300)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--maker-fee", type=float, default=None)
    parser.add_argument("--periods-per-year", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default="alpaca_cross_global")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--context-lengths", default="64,96,192")
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--forecast-cache-root", default="alpacaconstrainedexp/forecast_cache")
    parser.add_argument("--data-root", default=None)
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

    device = require_cuda_device("global policy training", allow_fallback=False)

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        longable = build_longable_symbols(
            crypto_symbols=_parse_symbols(args.crypto) if args.crypto else None,
            stock_symbols=_parse_symbols(args.long_stocks) if args.long_stocks else None,
        )
        shortable = build_shortable_symbols(
            stock_symbols=_parse_symbols(args.short_stocks) if args.short_stocks else None,
        )
        symbols = normalize_symbols(longable + shortable)
        if not symbols:
            raise ValueError("At least one symbol is required.")
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

    training_artifacts = train_model(data, args, device=device)
    checkpoint_path = training_artifacts.best_checkpoint
    model = _load_model(checkpoint_path, len(data.feature_columns), args.sequence_length)

    eval_data = data.modules.get(target_symbol, data.modules[data.target_symbol]) if hasattr(data, "modules") else data
    result = evaluate_model(
        model=model,
        data=eval_data,
        horizon=args.horizon,
        aggregate=args.aggregate,
        experiment_cfg=experiment_cfg,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        eval_days=args.eval_days,
        eval_hours=args.eval_hours,
        device=device,
    )

    if args.output_dir:
        history_payload = [entry.__dict__ for entry in training_artifacts.history]
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "training_history.json").write_text(json.dumps(history_payload, indent=2))

    print(f"Checkpoint: {checkpoint_path}")
    print(f"total_return: {result.total_return:.4f}")
    print(f"sortino: {result.sortino:.4f}")


if __name__ == "__main__":
    main()
