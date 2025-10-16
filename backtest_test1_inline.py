#!/usr/bin/env python3
"""Compatibility wrapper to run the inline backtest with REAL_TESTING on by default."""

import os
import sys

if "REAL_TESTING" not in os.environ:
    os.environ["REAL_TESTING"] = "1"

from backtest_test3_inline import backtest_forecasts  # noqa: E402


def main() -> None:
    symbol = "ETHUSD"
    if len(sys.argv) >= 2:
        symbol = sys.argv[1]
    backtest_forecasts(symbol)


if __name__ == "__main__":
    main()
