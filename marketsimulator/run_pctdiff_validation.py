from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marketsimulator.environment import _install_env_stub  # type: ignore
from marketsimulator.logging_utils import logger  # type: ignore
from marketsimulator import backtest_test3_inline as bt  # type: ignore


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Run pctdiff validation against the backtest analytics stack")
    parser.add_argument("symbol", help="Symbol to validate (e.g. BTCUSD)")
    parser.add_argument(
        "--max-return",
        type=float,
        default=None,
        help="Maximum absolute daily return to allow before flagging (defaults to PCTDIFF_MAX_DAILY_RETURN)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=None,
        help="Override the number of simulations per symbol",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _install_env_stub()
    limit = args.max_return
    logger.info("Validating pctdiff for %s (limit=%s, simulations=%s)", args.symbol, limit, args.num_simulations)
    max_abs = bt.validate_pctdiff(args.symbol, num_simulations=args.num_simulations, max_return=limit)
    logger.info("Pctdiff validation completed: max |return|=%.4f", max_abs)


if __name__ == "__main__":
    main()
