#!/usr/bin/env python3
"""Convenience wrapper to backtest Bags.fm neural model."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bagsneural.backtest import main


if __name__ == "__main__":
    main()
