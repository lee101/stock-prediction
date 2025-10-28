from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .trainer import run_with_config


def parse_overrides(raw_items: Optional[list[str]]) -> Dict[str, Any]:
    if not raw_items:
        return {}
    overrides: Dict[str, Any] = {}
    for item in raw_items:
        key, _, value = item.partition("=")
        if not key or not _:
            raise ValueError(f"Invalid override '{item}'. Use dotted.key=value syntax.")
        segments = key.split(".")
        cursor = overrides
        for segment in segments[:-1]:
            cursor = cursor.setdefault(segment, {})
        try:
            parsed: Any = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        cursor[segments[-1]] = parsed
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the PufferLib trading agent with the pufferlibtraining2 pipeline.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument(
        "--override",
        action="append",
        help="Inline override in dotted.path=value JSON format (e.g. train.total_timesteps=1000000).",
    )
    parser.add_argument("--print-summary", action="store_true", help="Print the summary JSON to stdout.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = parse_overrides(args.override)
    summary = run_with_config(args.config, overrides=overrides if overrides else None)
    if args.print_summary:
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
