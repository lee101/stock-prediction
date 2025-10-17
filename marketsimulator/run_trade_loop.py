from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict

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
    parser.add_argument(
        "--kronos-only",
        action="store_true",
        help="Force Kronos forecasting pipeline even if another model is selected.",
    )
    parser.add_argument(
        "--real-analytics",
        dest="real_analytics",
        action="store_true",
        help="Use the full forecasting/backtest stack instead of simulator mocks.",
    )
    parser.add_argument(
        "--mock-analytics",
        dest="real_analytics",
        action="store_false",
        help="Force lightweight simulator analytics (skips heavy forecasting models).",
    )
    parser.set_defaults(real_analytics=True)
    parser.add_argument(
        "--compact-logs",
        action="store_true",
        help="Reduce console log noise by using compact formatting and higher verbosity thresholds.",
    )
    return parser.parse_args()


def _set_logger_level(name: str, level: int) -> None:
    import logging

    log = logging.getLogger(name)
    log.setLevel(level)
    for handler in log.handlers:
        handler.setLevel(level)


def _configure_compact_logging_pre(enabled: bool) -> None:
    if not enabled:
        return

    os.environ.setdefault("COMPACT_TRADING_LOGS", "1")
    from loguru import logger as loguru_logger  # type: ignore

    loguru_logger.remove()
    loguru_logger.add(
        sys.stdout,
        level=os.getenv("SIM_LOGURU_LEVEL", "WARNING"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def _configure_compact_logging_post(enabled: bool) -> None:
    if not enabled:
        return

    import logging

    levels: Dict[str, int] = {
        "backtest_test3_inline": logging.WARNING,
        "data_curate_daily": logging.WARNING,
        "sizing_utils": logging.WARNING,
    }
    for name, level in levels.items():
        _set_logger_level(name, level)


def main() -> None:
    args = parse_args()
    _configure_compact_logging_pre(args.compact_logs)
    mode = "real" if args.real_analytics else "mock"
    logger.info(f"[sim] Analytics mode set to {mode.upper()} forecasting stack.")

    with activate_simulation(
        symbols=args.symbols,
        initial_cash=args.initial_cash,
        use_mock_analytics=not args.real_analytics,
        force_kronos=args.kronos_only,
    ) as controller:
        trade_module = importlib.import_module("trade_stock_e2e")
        _configure_compact_logging_post(args.compact_logs)

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
