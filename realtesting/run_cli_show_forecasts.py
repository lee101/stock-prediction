"""Run the `scripts.alpaca_cli show_forecasts` command against the simulated backend.

This helper advances a mocked Alpaca environment forward in time so we can
exercise the CLI the way we would on live markets without touching real keys.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Sequence

from freezegun import freeze_time

from marketsimulator import activate_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate `show_forecasts` over multiple time steps using the mocked Alpaca stack.",
    )
    parser.add_argument("symbol", help="Trading symbol to request forecasts for, e.g. ETHUSD.")
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Number of simulated forecast iterations to run.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="How many data rows to advance between iterations.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Starting cash balance for the simulated account.",
    )
    parser.add_argument(
        "--freeze-ignore",
        nargs="*",
        default=["transformers"],
        help="Module prefixes to skip when freezegun patches datetime.",
    )
    return parser.parse_args()


def run_simulation(
    symbol: str,
    steps: int,
    step_size: int,
    initial_cash: float,
    freeze_ignore: Sequence[str],
) -> None:
    # Import inside the simulation context so we use the patched modules.
    with activate_simulation(symbols=[symbol], initial_cash=initial_cash) as controller:
        alpaca_cli = importlib.import_module("scripts.alpaca_cli")

        for index in range(steps):
            timestamp = controller.current_time()
            print(f"\n[sim] Forecast step {index + 1}/{steps} @ {timestamp.isoformat()}")

            with freeze_time(timestamp, ignore=freeze_ignore):
                alpaca_cli.show_forecasts_for_symbol(symbol)

            if index < steps - 1:
                controller.advance_steps(step_size)


def main() -> None:
    args = parse_args()
    run_simulation(
        symbol=args.symbol,
        steps=args.steps,
        step_size=args.step_size,
        initial_cash=args.initial_cash,
        freeze_ignore=args.freeze_ignore,
    )


if __name__ == "__main__":
    main()
