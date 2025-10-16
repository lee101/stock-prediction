#!/usr/bin/env python3
"""
Parse coverage.xml and list files under a coverage threshold.

Optionally generate basic auto-tests for those files.

Usage:
  python tools/report_coverage_gaps.py --xml coverage.xml --threshold 80
  python tools/report_coverage_gaps.py --xml coverage.xml --threshold 80 --generate-tests
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileCoverage:
    filename: str
    percent: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="coverage.xml")
    p.add_argument("--threshold", type=float, default=80.0)
    p.add_argument("--generate-tests", action="store_true")
    return p.parse_args()


def parse_coverage_xml(xml_path: str) -> list[FileCoverage]:
    if not os.path.exists(xml_path):
        raise SystemExit(f"Coverage XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    results: list[FileCoverage] = []

    # Cobertura XML produced by pytest-cov: try to read <class line-rate>
    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename")
        line_rate = cls.attrib.get("line-rate")
        if not filename:
            continue
        if line_rate is not None:
            try:
                percent = float(line_rate) * 100.0
            except ValueError:
                continue
            results.append(FileCoverage(filename=filename, percent=percent))

    # Fallback: compute from <lines><line hits=...>
    if not results:
        for cls in root.findall(".//class"):
            filename = cls.attrib.get("filename")
            if not filename:
                continue
            lines = cls.find("lines")
            if lines is None:
                continue
            total = 0
            covered = 0
            for line in lines.findall("line"):
                total += 1
                hits = int(line.attrib.get("hits", "0"))
                if hits > 0:
                    covered += 1
            percent = 100.0 * covered / total if total else 0.0
            results.append(FileCoverage(filename=filename, percent=percent))

    # Normalize filenames
    for r in results:
        r.filename = str(Path(r.filename))

    # Deduplicate by best coverage entry per file
    best: dict[str, FileCoverage] = {}
    for r in results:
        if r.filename not in best or r.percent > best[r.filename].percent:
            best[r.filename] = r
    return list(best.values())


def main() -> None:
    args = parse_args()
    entries = parse_coverage_xml(args.xml)
    under = sorted([e for e in entries if e.percent < args.threshold], key=lambda e: e.percent)

    if not entries:
        print("No coverage entries found. Did you generate coverage.xml?")
        sys.exit(2)

    print(f"Found {len(entries)} files with coverage. Threshold = {args.threshold:.1f}%\n")
    print("Lowest coverage files:")
    for e in under[:50]:
        print(f"  {e.percent:6.2f}%  {e.filename}")

    if args.generate_tests and under:
        print("\nGenerating basic auto-tests for low-coverage files...")
        # Lazy import to avoid dependency when not needed
        from gen_basic_tests import generate_for_files  # type: ignore

        project_root = Path(__file__).resolve().parents[1]
        files = [str((project_root / e.filename).resolve()) for e in under]
        out_dir = project_root / "tests" / "auto"
        out_dir.mkdir(parents=True, exist_ok=True)
        generated = generate_for_files(files, out_dir)
        print(f"Generated {generated} test files in {out_dir}")


if __name__ == "__main__":
    main()

