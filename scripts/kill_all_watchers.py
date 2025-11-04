#!/usr/bin/env python
"""
Kill all MaxDiff watcher processes (NOT the positions themselves).

This utility terminates all running watcher processes:
- Entry watchers: Monitor for limit price entry opportunities
- Exit watchers: Monitor for take-profit exit opportunities (misleadingly called "close-position")

NOTE: This ONLY kills the watcher processes. It does NOT:
- Close any open positions
- Cancel any orders (unless --cancel-orders flag is used)

Use this when you have duplicate/conflicting watchers that need to be cleaned up.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import argparse

def find_watcher_processes():
    """Find all maxdiff_cli.py watcher processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )

        watchers = []
        for line in result.stdout.split('\n'):
            if 'maxdiff_cli.py' in line and ('close-position' in line or 'entry' in line):
                if 'grep' not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        watchers.append((pid, line))

        return watchers
    except subprocess.CalledProcessError as e:
        print(f"Error finding processes: {e}")
        return []


def kill_watchers(pids, signal_type=signal.SIGTERM):
    """Kill watcher processes by PID."""
    killed = []
    failed = []

    for pid in pids:
        try:
            os.kill(int(pid), signal_type)
            killed.append(pid)
        except ProcessLookupError:
            print(f"  Process {pid} already terminated")
        except PermissionError:
            print(f"  No permission to kill {pid}")
            failed.append(pid)
        except Exception as e:
            print(f"  Error killing {pid}: {e}")
            failed.append(pid)

    return killed, failed


def cancel_watcher_orders(cancel_orders: bool = False):
    """Optionally cancel orders associated with watchers."""
    if not cancel_orders:
        return

    print("\n[Optional] Canceling orders from watchers...")
    try:
        import subprocess
        result = subprocess.run(
            ["python", "cancel_multi_orders.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("✓ Canceled watcher orders")
        else:
            print(f"⚠ Order cancellation had issues: {result.stderr[:200]}")
    except Exception as e:
        print(f"⚠ Could not cancel orders: {e}")


def main():
    """Kill all watcher processes."""
    parser = argparse.ArgumentParser(
        description="Kill MaxDiff watcher processes (NOT positions)"
    )
    parser.add_argument(
        "--cancel-orders",
        action="store_true",
        help="Also cancel any orders associated with watchers"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts"
    )
    args = parser.parse_args()

    print("Finding all MaxDiff watchers...")
    print("(Note: 'close-position' watchers are EXIT watchers, not actual position closures)")
    watchers = find_watcher_processes()

    if not watchers:
        print("No active watchers found.")
        return 0

    print(f"\nFound {len(watchers)} active watcher processes:")
    for pid, cmdline in watchers:
        # Extract symbol and watcher type from command line
        symbol = "UNKNOWN"
        watcher_type = "unknown"
        if "close-position" in cmdline:
            watcher_type = "EXIT"
            parts = cmdline.split()
            for i, part in enumerate(parts):
                if part == "close-position" and i + 1 < len(parts):
                    symbol = parts[i + 1]
                    break
        elif "entry" in cmdline or "open-position" in cmdline:
            watcher_type = "ENTRY"
            # Try to extract symbol
            parts = cmdline.split()
            for part in parts:
                if part.isupper() and len(part) <= 8:
                    symbol = part
                    break
        print(f"  PID {pid}: {symbol:8s} [{watcher_type} watcher]")

    pids = [pid for pid, _ in watchers]

    if not args.yes:
        response = input(f"\nKill {len(pids)} watcher processes? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0

    # Try graceful termination first
    print(f"\nSending SIGTERM to {len(pids)} watcher processes...")
    killed, failed = kill_watchers(pids, signal.SIGTERM)

    if killed:
        print(f"Sent SIGTERM to {len(killed)} processes")
        time.sleep(2)

    # Check if any processes are still alive
    remaining = find_watcher_processes()

    if remaining:
        print(f"\n{len(remaining)} processes still running, sending SIGKILL...")
        remaining_pids = [pid for pid, _ in remaining]
        killed, failed = kill_watchers(remaining_pids, signal.SIGKILL)
        time.sleep(1)

        # Final check
        still_alive = find_watcher_processes()
        if still_alive:
            print(f"\nWARNING: {len(still_alive)} processes could not be killed:")
            for pid, cmdline in still_alive:
                print(f"  PID {pid}")
            return 1

    print(f"\n✓ Successfully killed all {len(watchers)} watcher processes")

    # Optionally cancel associated orders
    if args.cancel_orders:
        cancel_watcher_orders(cancel_orders=True)

    # Also clear out stale watcher config files with "launched" state
    config_dir = Path("strategy_state/maxdiff_watchers")
    if config_dir.exists():
        stale_configs = []
        for config_file in config_dir.glob("*.json"):
            try:
                import json
                with open(config_file) as f:
                    config = json.load(f)
                    if config.get("state") in ("launched", "exit_submitted", "awaiting_position"):
                        stale_configs.append(config_file)
            except Exception:
                pass

        if stale_configs:
            print(f"\nFound {len(stale_configs)} stale watcher configs:")
            for config_file in stale_configs[:5]:
                print(f"  {config_file.name}")
            if len(stale_configs) > 5:
                print(f"  ... and {len(stale_configs) - 5} more")

            if args.yes:
                response = 'y'
            else:
                response = input("\nUpdate these configs to 'inactive' state? [y/N]: ")

            if response.lower() == 'y':
                for config_file in stale_configs:
                    try:
                        import json
                        with open(config_file) as f:
                            config = json.load(f)
                        config["state"] = "inactive"
                        config["active"] = False
                        with open(config_file, 'w') as f:
                            json.dump(config, f, indent=2)
                    except Exception as e:
                        print(f"  Error updating {config_file.name}: {e}")
                print(f"✓ Updated {len(stale_configs)} config files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
