"""Shared CLI validation for XGB production-realism cost knobs."""

from __future__ import annotations

import argparse
import math


def validate_nonnegative_realism_args(
    args: argparse.Namespace,
    fields: tuple[tuple[str, str], ...] = (
        ("fee_rate", "fee_rate"),
        ("fill_buffer_bps", "fill_buffer_bps"),
        ("commission_bps", "commission_bps"),
    ),
) -> list[str]:
    """Return validation failures for finite, non-negative numeric CLI fields."""
    failures: list[str] = []
    for attr, label in fields:
        try:
            value = float(getattr(args, attr))
        except (TypeError, ValueError):
            failures.append(f"{label} must be finite and non-negative")
            continue
        if not math.isfinite(value) or value < 0.0:
            failures.append(f"{label} must be finite and non-negative")
    return failures
