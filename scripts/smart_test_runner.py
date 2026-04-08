#!/usr/bin/env python3
"""CLI entry point for the smart test runner."""

import importlib
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    impl = importlib.import_module("scripts._smart_test_runner")
    impl.main()


if __name__ == "__main__":
    main()
