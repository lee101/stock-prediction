#!/usr/bin/env python3
"""
Update hyperparams/best based on pct_return_mae instead of price_mae.

This script:
1. Loads configs from hyperparams_extended/{kronos,toto}
2. Compares based on validation pct_return_mae (what matters for trading!)
3. Updates hyperparams/best with the best performer
4. Generates a report of changes
"""
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

# Directories
EXTENDED_DIR = Path("hyperparams_extended")
BEST_DIR = Path("hyperparams/best")
BACKUP_DIR = Path("hyperparams/best_backup")
KRONOS_DIR = Path("hyperparams/kronos")
TOTO_DIR = Path("hyperparams/toto")

# Ensure directories exist
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
KRONOS_DIR.mkdir(parents=True, exist_ok=True)
TOTO_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Optional[dict]:
    """Load config from JSON file."""
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def save_config(path: Path, data: dict):
    """Save config to JSON file."""
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def get_pct_return_mae(config: Optional[dict]) -> float:
    """Extract validation pct_return_mae from config."""
    if not config:
        return float("inf")
    return config.get("validation", {}).get("pct_return_mae", float("inf"))


def compare_configs(symbol: str) -> Tuple[Optional[str], Optional[dict], Dict[str, float]]:
    """
    Compare kronos vs toto configs for a symbol based on pct_return_mae.

    Returns:
        (best_model, best_config, metrics_dict)
    """
    # Load extended configs
    kronos_config = load_config(EXTENDED_DIR / "kronos" / f"{symbol}.json")
    toto_config = load_config(EXTENDED_DIR / "toto" / f"{symbol}.json")

    # Get current best
    current_config = load_config(BEST_DIR / f"{symbol}.json")

    metrics = {
        "current": get_pct_return_mae(current_config),
        "kronos": get_pct_return_mae(kronos_config),
        "toto": get_pct_return_mae(toto_config),
    }

    # Find best based on pct_return_mae
    candidates = []
    if kronos_config and metrics["kronos"] != float("inf"):
        candidates.append(("kronos", kronos_config, metrics["kronos"]))
    if toto_config and metrics["toto"] != float("inf"):
        candidates.append(("toto", toto_config, metrics["toto"]))

    if not candidates:
        return None, None, metrics

    best_model, best_config, _ = min(candidates, key=lambda x: x[2])
    return best_model, best_config, metrics


def update_best_configs(dry_run: bool = True):
    """Update best configs based on pct_return_mae."""

    print("=" * 70)
    print("Updating Best Configurations Based on pct_return_mae")
    print("=" * 70)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE UPDATE'}")
    print()

    # Get all symbols from extended directories
    symbols = set()
    for model_dir in [EXTENDED_DIR / "kronos", EXTENDED_DIR / "toto"]:
        if model_dir.exists():
            symbols.update(p.stem for p in model_dir.glob("*.json"))

    symbols = sorted(symbols)

    changes = []
    improvements = []
    regressions = []
    no_change = []

    for symbol in symbols:
        best_model, best_config, metrics = compare_configs(symbol)

        if not best_model:
            print(f"⚠️  {symbol:10s} - No valid configs found")
            continue

        current_mae = metrics["current"]
        new_mae = metrics[best_model]

        # Calculate improvement
        if current_mae != float("inf"):
            improvement_pct = ((current_mae - new_mae) / current_mae * 100)
        else:
            improvement_pct = 100.0

        # Determine current model
        current_config = load_config(BEST_DIR / f"{symbol}.json")
        current_model = current_config.get("model", "unknown") if current_config else "none"

        # Check if we should update
        should_update = (new_mae < current_mae) or (current_mae == float("inf"))

        status = ""
        if should_update and current_model != best_model:
            status = f"UPDATE: {current_model} → {best_model}"
            changes.append(symbol)
            if improvement_pct > 0:
                improvements.append((symbol, improvement_pct))
            else:
                regressions.append((symbol, improvement_pct))
        elif should_update:
            status = f"IMPROVE: {best_model} (better config)"
            changes.append(symbol)
            if improvement_pct > 0:
                improvements.append((symbol, improvement_pct))
        else:
            status = f"NO CHANGE: {current_model} already best"
            no_change.append(symbol)

        # Print status
        icon = "✅" if improvement_pct > 5 else "📊" if improvement_pct > 0 else "⚠️"
        print(f"{icon} {symbol:10s} {status:30s} "
              f"mae: {new_mae:.6f} (current: {current_mae:.6f}, {improvement_pct:+.1f}%)")

        # Update if not dry run
        if not dry_run and should_update:
            # Backup current config
            if current_config:
                backup_path = BACKUP_DIR / f"{symbol}.json"
                save_config(backup_path, current_config)

            # Save to best
            best_path = BEST_DIR / f"{symbol}.json"
            save_config(best_path, best_config)

            # Also save to model-specific directory
            model_path = (KRONOS_DIR if best_model == "kronos" else TOTO_DIR) / f"{symbol}.json"
            save_config(model_path, best_config)

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total symbols evaluated: {len(symbols)}")
    print(f"Configs to update: {len(changes)}")
    print(f"  - Improvements: {len(improvements)}")
    print(f"  - Regressions: {len(regressions)}")
    print(f"No change needed: {len(no_change)}")

    if improvements:
        print(f"\nTop 5 Improvements:")
        for symbol, improvement in sorted(improvements, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {symbol:10s} {improvement:+.1f}%")

    if regressions:
        print(f"\nWarning: {len(regressions)} regressions detected:")
        for symbol, change in regressions[:5]:
            print(f"  {symbol:10s} {change:+.1f}%")

    if not dry_run:
        print(f"\n✅ Configs updated in {BEST_DIR}")
        print(f"📦 Backups saved to {BACKUP_DIR}")
    else:
        print(f"\n⚠️  DRY RUN - No files were modified")
        print(f"   Run with --apply to update configs")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Update best configs based on pct_return_mae"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update the files (default is dry-run)",
    )
    args = parser.parse_args()

    update_best_configs(dry_run=not args.apply)


if __name__ == "__main__":
    main()
