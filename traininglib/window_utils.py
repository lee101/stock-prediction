"""Shared helpers for window-based dataset configuration."""

from __future__ import annotations

from typing import Iterable, Tuple


def sanitize_bucket_choices(requested: int, provided: Iterable[int], flag_name: str, *, logger=None) -> Tuple[int, ...]:
    buckets = {int(requested)}
    dropped: list[int] = []
    for value in provided:
        bucket_value = int(value)
        if bucket_value <= requested:
            buckets.add(bucket_value)
        else:
            dropped.append(bucket_value)

    if dropped and logger is not None:
        dropped_str = ", ".join(str(item) for item in sorted(dropped))
        logger(f"Ignoring {flag_name} values greater than requested {requested}: {dropped_str}")

    return tuple(sorted(buckets))
