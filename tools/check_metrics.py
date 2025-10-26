#!/usr/bin/env python3
"""Validate metrics summary JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_FIELDS: dict[str, type] = {
    "return": (float, int),
    "sharpe": (float, int),
    "pnl": (float, int),
    "balance": (float, int),
}

OPTIONAL_NUMERIC_FIELDS: dict[str, type] = {
    "steps": (int,),
}

OPTIONAL_LIST_FIELDS: dict[str, type] = {
    "symbols": list,
}


def discover(glob: str) -> Iterable[Path]:
    return sorted(Path(".").glob(glob))


def validate_numeric(name: str, value: object) -> str | None:
    allowed = REQUIRED_FIELDS | OPTIONAL_NUMERIC_FIELDS
    expected = allowed[name]
    if not isinstance(value, expected):
        return f"{name}: expected {expected}, got {type(value).__name__}"
    if isinstance(value, (float, int)) and isinstance(value, float):
        if not math.isfinite(value):
            return f"{name}: value {value} is not finite"
    return None


def validate_file(path: Path) -> Sequence[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON ({exc})"]

    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"{path}: missing required field '{field}'")

    for field in REQUIRED_FIELDS:
        if field in data:
            err = validate_numeric(field, data[field])
            if err:
                errors.append(f"{path}: {err}")

    for field in OPTIONAL_NUMERIC_FIELDS:
        if field in data:
            err = validate_numeric(field, data[field])
            if err:
                errors.append(f"{path}: {err}")

    for field in OPTIONAL_LIST_FIELDS:
        if field in data and not isinstance(data[field], list):
            errors.append(f"{path}: {field} should be a list, got {type(data[field]).__name__}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        default="run*_summary.json",
        help="Glob pattern for summary files (default: %(default)s).",
    )
    args = parser.parse_args()

    files = list(discover(args.glob))
    if not files:
        raise SystemExit(f"No files matched pattern {args.glob!r}")

    all_errors: list[str] = []
    for file in files:
        all_errors.extend(validate_file(file))

    if all_errors:
        for err in all_errors:
            print(err)
        raise SystemExit(1)

    print(f"Validated {len(files)} file(s): OK")


if __name__ == "__main__":
    main()
