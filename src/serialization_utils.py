from __future__ import annotations

from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import Any


def serialize_for_checkpoint(value: Any) -> Any:
    """Convert an object into a pickle-stable, JSON-friendly payload.

    Motivation: some Python versions (notably 3.13+) pickle `pathlib.Path` objects
    from the internal module `pathlib._local`. Loading those pickles in older
    Python versions (<=3.12) can fail with:

        ModuleNotFoundError: No module named 'pathlib._local'

    By converting Path objects (and nested dataclasses) to plain built-ins we
    keep checkpoints portable across Python versions and environments.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Convert Paths eagerly to avoid cross-version pickle issues.
    if isinstance(value, Path):
        return str(value)

    # Numpy scalars sometimes sneak in via config merges; keep them portable.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = None
        else:
            if scalar is None or isinstance(scalar, (str, int, float, bool)):
                return scalar

    if is_dataclass(value):
        payload: dict[str, Any] = {}
        for f in fields(value):
            try:
                payload[f.name] = serialize_for_checkpoint(getattr(value, f.name))
            except Exception:
                # Best-effort metadata; never fail checkpointing due to config serialization.
                payload[f.name] = repr(getattr(value, f.name, None))
        return payload

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = serialize_for_checkpoint(v)
        return out

    if isinstance(value, list):
        return [serialize_for_checkpoint(v) for v in value]

    if isinstance(value, tuple):
        return tuple(serialize_for_checkpoint(v) for v in value)

    if isinstance(value, set):
        # Sort for determinism when elements are comparable; fall back to repr.
        try:
            return sorted(serialize_for_checkpoint(v) for v in value)  # type: ignore[return-value]
        except TypeError:
            return sorted((repr(v) for v in value))

    return repr(value)

