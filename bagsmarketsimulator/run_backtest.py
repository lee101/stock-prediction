#!/usr/bin/env python3
"""Run backtest on Bags.fm neural trading model.

Usage:
    python -m bagsmarketsimulator.run_backtest --mint CODEX_MINT --checkpoint path/to/model.pt

    # Sweep thresholds to find optimal
    python -m bagsmarketsimulator.run_backtest --mint CODEX_MINT --checkpoint path/to/model.pt --sweep
"""

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bagsmarketsimulator.simulator import NeuralSimulator, run_backtest


def main():
    parser = argparse.ArgumentParser(description="Backtest Bags.fm neural model")
    parser.add_argument(
        "--mint",
        type=str,
        default="HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS",
        help="Token mint address (default: CODEX)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("bagsneural/checkpoints/bagsneural_HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS_best.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--ohlc",
        type=Path,
        default=Path("bagstraining/ohlc_data.csv"),
        help="Path to OHLC data CSV",
    )
    parser.add_argument("--buy-threshold", type=float, default=0.50)
    parser.add_argument("--sell-threshold", type=float, default=0.45)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--initial-sol", type=float, default=1.0)
    parser.add_argument("--max-position-sol", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep thresholds to find optimal",
    )
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.sweep:
        # Sweep thresholds
        import torch
        from bagsneural.dataset import FeatureNormalizer, load_ohlc_dataframe
        from bagsneural.model import BagsNeuralModel

        df = load_ohlc_dataframe(args.ohlc, args.mint)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        config = ckpt["config"]
        normalizer = FeatureNormalizer.from_dict(ckpt["normalizer"])
        context = config.get("context", 64)

        input_dim = context * 3
        model = BagsNeuralModel(
            input_dim=input_dim,
            hidden_dims=config.get("hidden", [128, 64]),
            dropout=config.get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["model_state"])

        simulator = NeuralSimulator(
            model=model,
            normalizer=normalizer,
            context_bars=context,
            initial_sol=args.initial_sol,
            max_position_sol=args.max_position_sol,
            device=args.device,
        )

        print(f"\n{'='*60}")
        print("Threshold Sweep Results")
        print(f"{'='*60}")

        results = simulator.sweep_thresholds(
            df=df,
            test_split=args.test_split,
        )

        print(f"\n{'='*60}")
        print("Top 5 Configurations (by Sortino):")
        print(f"{'='*60}")
        for i, r in enumerate(results[:5]):
            print(f"\n{i+1}. buy={r.buy_threshold:.2f} sell={r.sell_threshold:.2f}")
            print(f"   Return: {r.total_return_pct:+.2f}%")
            print(f"   Sharpe: {r.sharpe_ratio:.3f}")
            print(f"   Sortino: {r.sortino_ratio:.3f}")
            print(f"   Max DD: {r.max_drawdown_pct:.2f}%")
            print(f"   Trades: {r.total_trades} (win rate: {r.win_rate*100:.1f}%)")

    else:
        # Single backtest
        result = run_backtest(
            checkpoint_path=args.checkpoint,
            ohlc_path=args.ohlc,
            mint=args.mint,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            test_split=args.test_split,
            initial_sol=args.initial_sol,
            max_position_sol=args.max_position_sol,
            device=args.device,
        )

        print(result.summary())


if __name__ == "__main__":
    main()
