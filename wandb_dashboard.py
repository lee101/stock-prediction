#!/usr/bin/env python3
"""Compatibility shim that executes the maintained dashboard script in-place."""

from __future__ import annotations

from pathlib import Path


_IMPL_PATH = Path(__file__).resolve().parent / "scripts" / "wandb_dashboard.py"
globals()["__file__"] = str(_IMPL_PATH)
exec(compile(_IMPL_PATH.read_text(), str(_IMPL_PATH), "exec"), globals(), globals())
