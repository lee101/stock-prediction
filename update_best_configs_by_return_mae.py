#!/usr/bin/env python
"""
Re-select best models based on pct_return_mae instead of price_mae.

This matters for trading because we care about return prediction accuracy,
not absolute price prediction accuracy.
"""
import json
from pathlib import Path
from typing import Dict, Any


def update_model_selection(symbol: str) -> Dict[str, Any]:
    """Select best model based on pct_return_mae."""

    kronos_path = Path("hyperparams/kronos") / f"{symbol}.json"
    toto_path = Path("hyperparams/toto") / f"{symbol}.json"

    if not kronos_path.exists() or not toto_path.exists():
        print(f"⚠️  {symbol}: Missing configs")
        return None

    with open(kronos_path) as f:
        kronos = json.load(f)
    with open(toto_path) as f:
        toto = json.load(f)

    kronos_return_mae = kronos["test"]["pct_return_mae"]
    toto_return_mae = toto["test"]["pct_return_mae"]

    # Select model with lower pct_return_mae
    if kronos_return_mae < toto_return_mae:
        best_model = "kronos"
        best_config = kronos
        best_mae = kronos_return_mae
        improvement = ((toto_return_mae - kronos_return_mae) / toto_return_mae * 100)
    else:
        best_model = "toto"
        best_config = toto
        best_mae = toto_return_mae
        improvement = ((kronos_return_mae - toto_return_mae) / kronos_return_mae * 100)

    # Create selection record
    selection = {
        "symbol": symbol,
        "model": best_model,
        "config": best_config["config"],
        "validation": best_config["validation"],
        "test": best_config["test"],
        "windows": best_config["windows"],
        "config_path": f"hyperparams/{best_model}/{symbol}.json",
        "metadata": {
            "source": "update_best_configs_by_return_mae",
            "selection_metric": "test_pct_return_mae",
            "selection_value": best_mae,
            "kronos_pct_return_mae": kronos_return_mae,
            "toto_pct_return_mae": toto_return_mae,
            "improvement_pct": improvement,
        }
    }

    return selection


def main():
    """Update all model selections to use pct_return_mae."""

    best_dir = Path("hyperparams/best")
    best_dir.mkdir(parents=True, exist_ok=True)

    # Get all symbols that have both kronos and toto configs
    kronos_symbols = {p.stem for p in Path("hyperparams/kronos").glob("*.json")}
    toto_symbols = {p.stem for p in Path("hyperparams/toto").glob("*.json")}
    symbols = sorted(kronos_symbols & toto_symbols)

    print(f"Updating model selections for {len(symbols)} symbols...")
    print(f"Selection metric: test_pct_return_mae ⭐")
    print()

    changes = []
    no_changes = []

    for symbol in symbols:
        # Load old selection
        old_selection_path = best_dir / f"{symbol}.json"
        old_model = None
        if old_selection_path.exists():
            with open(old_selection_path) as f:
                old_selection = json.load(f)
                old_model = old_selection.get("model")

        # Create new selection
        new_selection = update_model_selection(symbol)
        if not new_selection:
            continue

        new_model = new_selection["model"]
        return_mae = new_selection["metadata"]["selection_value"]
        improvement = new_selection["metadata"]["improvement_pct"]

        # Save new selection
        new_path = best_dir / f"{symbol}.json"
        with open(new_path, "w") as f:
            json.dump(new_selection, f, indent=2)

        # Track changes
        if old_model and old_model != new_model:
            changes.append(f"{symbol:12s}: {old_model:8s} → {new_model:8s} (return_mae={return_mae:.4f}, +{improvement:.1f}%)")
        else:
            status = "NEW" if not old_model else "SAME"
            no_changes.append(f"{symbol:12s}: {new_model:8s} ({status}, return_mae={return_mae:.4f})")

    print("\n=== CHANGES ===")
    if changes:
        for line in changes:
            print(f"✓ {line}")
    else:
        print("No changes")

    print(f"\n=== NO CHANGES ({len(no_changes)}) ===")
    for line in no_changes[:10]:
        print(f"  {line}")
    if len(no_changes) > 10:
        print(f"  ... and {len(no_changes) - 10} more")

    print(f"\n✓ Updated {len(symbols)} model selections in hyperparams/best/")
    print(f"  Changes: {len(changes)}")
    print(f"  No change: {len(no_changes)}")


if __name__ == "__main__":
    main()
