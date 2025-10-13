from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from marketsimulator.environment import activate_simulation  # type: ignore
    from marketsimulator.logging_utils import logger  # type: ignore
else:
    from .environment import activate_simulation
    from .logging_utils import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate trade_stock_e2e with a mocked Alpaca stack.")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA"], help="Symbols to simulate.")
    parser.add_argument("--steps", type=int, default=32, help="Number of simulation steps to run.")
    parser.add_argument("--step-size", type=int, default=1, help="Data rows to advance between iterations.")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Starting cash balance.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of picks to keep each iteration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with activate_simulation(symbols=args.symbols, initial_cash=args.initial_cash) as controller:
        trade_module = importlib.import_module("trade_stock_e2e")

        previous_picks = {}
        for step in range(args.steps):
            timestamp = controller.current_time()
            logger.info(f"[sim] Step {step + 1}/{args.steps} @ {timestamp}")

            analyzed = trade_module.analyze_symbols(args.symbols)
            current = {
                symbol: data
                for symbol, data in list(analyzed.items())[: args.top_k]
                if data["avg_return"] > 0
            }
            if current:
                trade_module.log_trading_plan(current, f"SIM-STEP-{step + 1}")
            trade_module.manage_positions(current, previous_picks, analyzed)

            previous_picks = current
            controller.advance_steps(args.step_size)

        summary = controller.summary()
        logger.info(f"[sim] Final summary: cash={summary['cash']:.2f}, equity={summary['equity']:.2f}")
        if summary["positions"]:
            logger.info(f"[sim] Open positions: {summary['positions']}")


if __name__ == "__main__":
    main()
