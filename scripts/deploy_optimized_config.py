#!/usr/bin/env python3
"""Deploy optimized trading parameters to supervisor."""
from __future__ import annotations
import argparse
import re
import subprocess
from pathlib import Path

SUPERVISOR_CONF = Path("/etc/supervisor/conf.d/binanceexp1-selector.conf")
BEST_PARAMS = {
    "intensity_scale": 6.0,  # from 5.0
    "default_offset": 0.0,
    "offset_map": "ETHUSD=0.0003,SOLUSD=0.0005",
    "max_hold_hours": 6,
    "min_edge": 0.0,
}


def update_config(dry_run: bool = True):
    if not SUPERVISOR_CONF.exists():
        print(f"Config not found: {SUPERVISOR_CONF}")
        return False

    content = SUPERVISOR_CONF.read_text()
    orig = content

    # Update intensity-scale
    content = re.sub(r'--intensity-scale\s+[\d.]+', f'--intensity-scale {BEST_PARAMS["intensity_scale"]}', content)

    # Update max-hold-hours
    content = re.sub(r'--max-hold-hours\s+\d+', f'--max-hold-hours {BEST_PARAMS["max_hold_hours"]}', content)

    if content == orig:
        print("No changes needed")
        return False

    print("Changes:")
    print(f"  intensity-scale: 5.0 -> {BEST_PARAMS['intensity_scale']}")

    if dry_run:
        print("\nDry run - no changes applied")
        print("Run with --apply to apply changes")
        return True

    # Write to temp file
    tmp = Path("/tmp/binanceexp1-selector.conf.new")
    tmp.write_text(content)

    # Copy with sudo and restart
    cmd = f"sudo cp {tmp} {SUPERVISOR_CONF} && sudo supervisorctl reread && sudo supervisorctl update && sudo supervisorctl restart binanceexp1-selector"
    print(f"\nApplying: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print("Applied successfully")
    print(result.stdout)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply changes (requires sudo)")
    args = parser.parse_args()

    update_config(dry_run=not args.apply)


if __name__ == "__main__":
    main()
