#!/usr/bin/env python3
"""Extract summary metrics from a marketsimulator run log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional


PATTERNS = {
    "return": re.compile(r"return[^-+0-9]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    "sharpe": re.compile(r"sharpe[^-+0-9]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    "pnl": re.compile(r"pnl[^-+0-9]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    "balance": re.compile(r"balance[^-+0-9]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
}


def extract_metrics(text: str) -> Dict[str, Optional[float]]:
    """Scan log text and pull the last numeric mention for each metric."""
    result: Dict[str, Optional[float]] = {key: None for key in PATTERNS}
    lines = text.splitlines()
    for line in lines:
        for key, pattern in PATTERNS.items():
            match = pattern.search(line)
            if not match:
                continue
            value = match.group(1)
            try:
                result[key] = float(value)
            except ValueError:
                continue
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        required=True,
        type=Path,
        help="Path to the marketsimulator log file to parse.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the JSON summary.",
    )
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="ignore")
    metrics = extract_metrics(text)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
