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
    parser = argparse.ArgumentParser(description="Simulate show_forecasts output over multiple steps.")
    parser.add_argument("symbol", help="Symbol to display forecasts for.")
    parser.add_argument("--steps", type=int, default=8, help="Number of forecast iterations to run.")
    parser.add_argument("--step-size", type=int, default=1, help="Data rows to advance between iterations.")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Starting cash balance.")
    parser.add_argument(
        "--kronos-only",
        action="store_true",
        help="Force Kronos forecasting pipeline even if another model is selected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [args.symbol]

    with activate_simulation(
        symbols=symbols,
        initial_cash=args.initial_cash,
        force_kronos=args.kronos_only,
    ) as controller:
        alpaca_cli = importlib.import_module("scripts.alpaca_cli")

        for step in range(args.steps):
            timestamp = controller.current_time()
            logger.info(f"[sim] Forecast step {step + 1}/{args.steps} @ {timestamp}")
            alpaca_cli.show_forecasts_for_symbol(args.symbol)
            controller.advance_steps(args.step_size)


if __name__ == "__main__":
    main()
