#!/usr/bin/env python
"""
Re-select best models based on pct_return_mae, including Chronos2 candidates.

This matters for trading because we care about return prediction accuracy,
not absolute price prediction accuracy.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

HYPERPARAM_ROOT = Path(os.getenv("HYPERPARAM_ROOT", "hyperparams"))
MODEL_DIRS = {
    "kronos": HYPERPARAM_ROOT / "kronos",
    "toto": HYPERPARAM_ROOT / "toto",
    "chronos2": HYPERPARAM_ROOT / "chronos2",
}


def _load_model_payload(model: str, symbol: str) -> Optional[Dict[str, Any]]:
    path = MODEL_DIRS[model] / f"{symbol}.json"
    if not path.exists():
        return None
    with path.open() as fp:
        payload = json.load(fp)
    return payload


def update_model_selection(symbol: str) -> Optional[Dict[str, Any]]:
    """Select the best available model based on pct_return_mae."""

    candidates: Dict[str, Dict[str, Any]] = {}
    for model_name in MODEL_DIRS:
        payload = _load_model_payload(model_name, symbol)
        if payload is None:
            continue
        mae = payload["validation"]["pct_return_mae"]
        candidates[model_name] = {
            "payload": payload,
            "mae": mae,
        }

    if not candidates:
        print(f"⚠️  {symbol}: No hyperparameter records found across {list(MODEL_DIRS)}")
        return None

    best_model = min(candidates.items(), key=lambda item: item[1]["mae"])[0]
    best_entry = candidates[best_model]
    best_payload = best_entry["payload"]
    best_mae = best_entry["mae"]

    sorted_mae = sorted(entry["mae"] for entry in candidates.values())
    if len(sorted_mae) > 1 and sorted_mae[1] > 0:
        improvement = ((sorted_mae[1] - best_mae) / sorted_mae[1]) * 100
    else:
        improvement = 0.0

    candidate_mae_map = {model: entry["mae"] for model, entry in candidates.items()}

    selection = {
        "symbol": symbol,
        "model": best_model,
        "config": best_payload["config"],
        "validation": best_payload["validation"],
        "test": best_payload["test"],
        "windows": best_payload["windows"],
        "config_path": f"hyperparams/{best_model}/{symbol}.json",
        "metadata": {
            "source": "update_best_configs_by_return_mae",
            "selection_metric": "validation_pct_return_mae",
            "selection_value": best_mae,
            "kronos_pct_return_mae": candidate_mae_map.get("kronos"),
            "toto_pct_return_mae": candidate_mae_map.get("toto"),
            "chronos2_pct_return_mae": candidate_mae_map.get("chronos2"),
            "candidate_pct_return_mae": candidate_mae_map,
            "improvement_pct": improvement,
        },
    }

    return selection


def main():
    """Update all model selections to use pct_return_mae."""

    best_dir = HYPERPARAM_ROOT / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    symbol_sets = []
    for model_dir in MODEL_DIRS.values():
        if model_dir.exists():
            symbol_sets.append({p.stem for p in model_dir.glob("*.json")})
    symbols = sorted(set().union(*symbol_sets)) if symbol_sets else []

    print(f"Updating model selections for {len(symbols)} symbols...")
    print(f"Selection metric: validation_pct_return_mae ⭐")
    print()

    changes = []
    no_changes = []

    for symbol in symbols:
        # Load old selection
        old_selection_path = best_dir / f"{symbol}.json"
        old_model = None
        old_val_mae = None
        if old_selection_path.exists():
            with open(old_selection_path) as f:
                old_selection = json.load(f)
            old_model = old_selection.get("model")
            old_val_mae = old_selection.get("validation", {}).get("pct_return_mae")

        # Create new selection
        new_selection = update_model_selection(symbol)
        if not new_selection:
            continue

        new_model = new_selection["model"]
        return_mae = new_selection["metadata"]["selection_value"]
        improvement = new_selection["metadata"]["improvement_pct"]
        new_val_mae = new_selection["validation"]["pct_return_mae"]

        if old_val_mae is not None and new_val_mae >= old_val_mae:
            no_changes.append(
                f"{symbol:12s}: {new_model:8s} (SKIP, val_mae={new_val_mae:.4f} ≥ {old_val_mae:.4f})"
            )
            continue

        # Save new selection
        new_path = best_dir / f"{symbol}.json"
        with open(new_path, "w") as f:
            json.dump(new_selection, f, indent=2)

        # Track changes
        if old_model and old_model != new_model:
            changes.append(f"{symbol:12s}: {old_model:8s} → {new_model:8s} (return_mae={return_mae:.4f}, +{improvement:.1f}%)")
        else:
            status = "NEW" if not old_model else "SAME"
            no_changes.append(f"{symbol:12s}: {new_model:8s} ({status}, val_mae={return_mae:.4f})")

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
