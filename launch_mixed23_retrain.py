#!/usr/bin/env python3
"""Compatibility shim for the mixed23 retrain launcher."""
from __future__ import annotations

from scripts.launch_mixed23_retrain import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
