#!/usr/bin/env python3

"""
Utility to relax PettingZoo's Requires-Python metadata so we can run on Python 3.13.

PettingZoo 1.15.0 hasn't updated its metadata yet, so installers mark it as
incompatible even though it works in practice. Running this script inside the
virtualenv updates the METADATA file to accept anything below Python 4.
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path


TARGET_SNIPPET = "Requires-Python: >=3.7, <3.11"
REPLACEMENT_SNIPPET = "Requires-Python: >=3.7, <4"


def patch_metadata(metadata_path: Path) -> bool:
    """Replace the stale Requires-Python guard if present."""
    try:
        text = metadata_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False

    if TARGET_SNIPPET not in text:
        return False

    metadata_path.write_text(text.replace(TARGET_SNIPPET, REPLACEMENT_SNIPPET), encoding="utf-8")
    return True


def main() -> int:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    candidates = sorted(site_packages.glob("pettingzoo-*.dist-info/METADATA"))

    if not candidates:
        print("pettingzoo METADATA file not found in this environment.", file=sys.stderr)
        return 1

    patched_any = False
    for metadata_path in candidates:
        if patch_metadata(metadata_path):
            print(f"Patched {metadata_path}")
            patched_any = True

    if not patched_any:
        print("Nothing to patch; metadata already permits Python 3.13+.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
