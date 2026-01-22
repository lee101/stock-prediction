#!/usr/bin/env python3
"""Convenience wrapper to backtest Bags.fm neural model."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from bagsneural.backtest import main


if __name__ == "__main__":
    main()
