#!/usr/bin/env python3
"""Convert JSON metrics summaries into a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence


def discover(path_glob: str) -> Iterable[Path]:
    return sorted(Path(".").glob(path_glob))


def load_summary(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["summary_path"] = str(path)
    if "log_path" not in data and path.name.endswith("_summary.json"):
        data["log_path"] = str(path.with_name(path.name.replace("_summary.json", ".log")))
    return data


def write_csv(rows: Sequence[dict[str, object]], output: Path) -> None:
    if not rows:
        raise SystemExit("No summary files matched the pattern.")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default="run*_summary.json",
        help="Glob pattern for summary JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV file.",
    )
    args = parser.parse_args()

    rows = [load_summary(path) for path in discover(args.input_glob)]
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
