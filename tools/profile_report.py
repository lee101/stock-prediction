#!/usr/bin/env python3
"""Generate markdown profiling report from existing profile files.

Reads files produced by pufferlib_market/profile_training.py and
generates a human-readable report at profiles/report.md.

Usage:
    python tools/profile_report.py [--profiles-dir pufferlib_market/profiles/]
    python tools/profile_report.py --profiles-dir /path/to/profiles/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import from pufferlib_market
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pufferlib_market.profile_training import generate_markdown_report, PROFILES_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate markdown profiling report from existing profile files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=PROFILES_DIR,
        help="Directory containing profile files (default: profiles/)",
    )
    args = parser.parse_args()

    profiles_dir = args.profiles_dir

    # Check for any profile files
    if profiles_dir.exists():
        candidate_files = (
            list(profiles_dir.glob("*.json"))
            + list(profiles_dir.glob("*.svg"))
            + list(profiles_dir.glob("*.txt"))
        )
    else:
        candidate_files = []

    if not candidate_files:
        print(f"No profile files found in {profiles_dir}")
        print(
            "Run profiling first:\n"
            "  python pufferlib_market/profile_training.py --quick"
        )
        sys.exit(0)

    report_path = generate_markdown_report(profiles_dir)
    print(f"Report written to: {report_path}")

    # Print a brief preview of the throughput line
    try:
        lines = report_path.read_text().splitlines()
        for line in lines:
            if line.startswith("Steps/sec:") or line.startswith("GPU:"):
                print(f"  {line}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
