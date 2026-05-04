#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`scripts.eval_all_checkpoints`.

The maintained implementation lives under ``scripts/``. Keep this root
entrypoint as a thin shim so ``python eval_all_checkpoints.py`` and legacy
imports do not drift from ``python scripts/eval_all_checkpoints.py``.
"""

from __future__ import annotations

from scripts import eval_all_checkpoints as _maintained_eval_all_checkpoints
from scripts.eval_all_checkpoints import *  # noqa: F403


main = _maintained_eval_all_checkpoints.main


def __getattr__(name: str):
    return getattr(_maintained_eval_all_checkpoints, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_maintained_eval_all_checkpoints)))


if __name__ == "__main__":
    main()
