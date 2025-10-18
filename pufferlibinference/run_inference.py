#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from .config import InferenceDataConfig, PufferInferenceConfig
from .engine import PortfolioRLInferenceEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PufferLib portfolio inference over historical data.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained portfolio checkpoint (.pt).")
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated list of symbols matching the checkpoint ordering.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("trainingdata"), help="Directory containing per-symbol CSVs.")
    parser.add_argument("--processor", type=Path, default=None, help="Optional explicit StockDataProcessor scaler path.")
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--resample", type=str, default=None, help="Optional pandas resample rule (e.g., '1D').")
    parser.add_argument("--initial-value", type=float, default=1.0, help="Initial portfolio value for simulation.")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0, help="Extra transaction cost in basis points.")
    parser.add_argument("--leverage-limit", type=float, default=2.0, help="Maximum gross leverage.")
    parser.add_argument("--borrowing-cost", type=float, default=0.0675, help="Annualised borrowing cost above 1× leverage.")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write summary metrics as JSON.")
    parser.add_argument("--decisions-csv", type=Path, help="Optional path to export allocation decisions.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    symbols = [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
    if not symbols:
        raise ValueError("At least one symbol must be supplied via --symbols.")

    data_cfg = InferenceDataConfig(
        symbols=symbols,
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        resample_rule=args.resample,
    )
    inference_cfg = PufferInferenceConfig(
        checkpoint_path=args.checkpoint,
        processor_path=args.processor,
        device=args.device,
        transaction_cost_bps=args.transaction_cost_bps,
        leverage_limit=args.leverage_limit,
        borrowing_cost=args.borrowing_cost,
    )

    engine = PortfolioRLInferenceEngine(inference_cfg, data_cfg)
    result = engine.simulate(initial_value=args.initial_value)

    print("=== PufferLib Inference Summary ===")
    for key, value in sorted(result.summary.items()):
        print(f"{key}: {value:.6f}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result.summary, indent=2))

    if args.decisions_csv:
        args.decisions_csv.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd  # Local import to avoid hard dependency unless needed

        decisions_df = pd.DataFrame(
            {
                "timestamp": [str(dec.timestamp) for dec in result.decisions],
                "portfolio_value": [dec.portfolio_value for dec in result.decisions],
                "turnover": [dec.turnover for dec in result.decisions],
                "trading_cost": [dec.trading_cost for dec in result.decisions],
                "financing_cost": [dec.financing_cost for dec in result.decisions],
                "net_return": [dec.net_return for dec in result.decisions],
                **{f"weight_{sym}": [dec.weights[sym] for dec in result.decisions] for sym in symbols},
            }
        )
        decisions_df.to_csv(args.decisions_csv, index=False)


if __name__ == "__main__":
    main()
