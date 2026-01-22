#!/usr/bin/env python3
"""Convenience wrapper to train multi-token Bags.fm neural model."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from bagsneural.train_multi import main


if __name__ == "__main__":
    main()
