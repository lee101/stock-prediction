from __future__ import annotations

import argparse
from pathlib import Path

from ..config import DataConfig, EnvironmentConfig, EvaluationConfig
from .backtester import DifferentiableMarketBacktester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run differentiable market backtester")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to policy checkpoint (best.pt/latest.pt)")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdata"), help="Root of OHLC CSV files")
    parser.add_argument("--data-glob", type=str, default="*.csv", help="Glob pattern for OHLC CSV discovery")
    parser.add_argument("--max-assets", type=int, default=None, help="Optionally cap number of assets")
    parser.add_argument("--exclude", type=str, nargs="*", default=(), help="Symbols to exclude")
    parser.add_argument("--window-length", type=int, default=256, help="Evaluation window length")
    parser.add_argument("--stride", type=int, default=64, help="Stride between evaluation windows")
    parser.add_argument("--report-dir", type=Path, default=Path("differentiable_market") / "evals", help="Directory to store evaluation reports")
    parser.add_argument("--no-trades", action="store_true", help="Disable trade log emission")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig(
        root=args.data_root,
        glob=args.data_glob,
        max_assets=args.max_assets,
        exclude_symbols=tuple(args.exclude),
    )
    env_cfg = EnvironmentConfig()
    eval_cfg = EvaluationConfig(
        window_length=args.window_length,
        stride=args.stride,
        report_dir=args.report_dir,
        store_trades=not args.no_trades,
    )
    backtester = DifferentiableMarketBacktester(data_cfg, env_cfg, eval_cfg)
    metrics = backtester.run(args.checkpoint)
    print(metrics)


if __name__ == "__main__":
    main()

