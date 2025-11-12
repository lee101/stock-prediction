#!/usr/bin/env python3
"""
Test script to validate augmentation strategies.

Tests that:
1. Transformations work without errors
2. Inverse transformations are accurate
3. Data shapes are preserved
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preaug_sweeps.augmentations import AUGMENTATION_REGISTRY, get_augmentation
from kronostraining.data_utils import load_symbol_dataframe


def test_augmentation_roundtrip(strategy_name: str, df: pd.DataFrame) -> dict:
    """
    Test that augmentation -> inverse is accurate.

    Returns:
        Dict with test results
    """
    print(f"\nTesting: {strategy_name}")
    print("-" * 60)

    # Get augmentation
    augmentation = get_augmentation(strategy_name)

    # Extract price features
    price_cols = ["open", "high", "low", "close", "volume", "amount"]
    df_features = df[price_cols].copy()

    original_values = df_features.values.copy()

    # Transform
    try:
        df_aug = augmentation.transform_dataframe(df_features)
        print(f"✓ Transform successful")
        print(f"  Original range: [{original_values.min():.4f}, {original_values.max():.4f}]")
        print(f"  Augmented range: [{df_aug.values.min():.4f}, {df_aug.values.max():.4f}]")
    except Exception as e:
        print(f"✗ Transform failed: {e}")
        return {"status": "failed", "stage": "transform", "error": str(e)}

    # Simulate predictions (use last 30 rows as "predictions")
    pred_aug = df_aug.tail(30).values

    # Inverse transform
    try:
        context_df = df_features.head(len(df_features) - 30)
        pred_original = augmentation.inverse_transform_predictions(
            pred_aug,
            context_df,
            columns=df_features.columns,
        )
        print(f"✓ Inverse transform successful")
    except Exception as e:
        print(f"✗ Inverse transform failed: {e}")
        return {"status": "failed", "stage": "inverse", "error": str(e)}

    # Check accuracy
    actual_original = original_values[-30:]
    mae = np.mean(np.abs(pred_original - actual_original))
    max_error = np.max(np.abs(pred_original - actual_original))
    relative_error = mae / (np.mean(np.abs(actual_original)) + 1e-8)

    print(f"  MAE: {mae:.6f}")
    print(f"  Max Error: {max_error:.6f}")
    print(f"  Relative Error: {relative_error:.6f}")

    # Determine if acceptable
    if strategy_name == "baseline":
        # Baseline should be perfect
        threshold = 1e-6
    else:
        # Others should be reasonably accurate
        threshold = 0.1  # 10% relative error

    status = "pass" if relative_error < threshold else "warning"

    if status == "pass":
        print(f"✓ Roundtrip accuracy: PASS")
    else:
        print(f"⚠ Roundtrip accuracy: WARNING (relative error {relative_error:.4f} > {threshold})")

    return {
        "status": status,
        "mae": mae,
        "max_error": max_error,
        "relative_error": relative_error,
    }


def main():
    """Run tests on all augmentation strategies."""
    print("=" * 80)
    print("AUGMENTATION VALIDATION TESTS")
    print("=" * 80)

    # Load test data
    data_file = Path("trainingdata/ETHUSD.csv")
    if not data_file.exists():
        print(f"Error: Test data not found at {data_file}")
        return 1

    print(f"\nLoading test data: {data_file}")
    df = load_symbol_dataframe(data_file)
    print(f"Loaded {len(df)} rows")

    # Test each strategy
    results = {}
    for strategy_name in AUGMENTATION_REGISTRY.keys():
        try:
            result = test_augmentation_roundtrip(strategy_name, df)
            results[strategy_name] = result
        except Exception as e:
            print(f"\n✗ {strategy_name} FAILED: {e}")
            results[strategy_name] = {"status": "error", "error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = [k for k, v in results.items() if v["status"] == "pass"]
    warnings = [k for k, v in results.items() if v["status"] == "warning"]
    failed = [k for k, v in results.items() if v["status"] in ("failed", "error")]

    print(f"\n✓ Passed: {len(passed)}/{len(results)}")
    for name in passed:
        rel_err = results[name].get("relative_error", 0)
        print(f"  - {name:25s} (relative error: {rel_err:.6f})")

    if warnings:
        print(f"\n⚠ Warnings: {len(warnings)}")
        for name in warnings:
            rel_err = results[name].get("relative_error", 0)
            print(f"  - {name:25s} (relative error: {rel_err:.6f})")

    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for name in failed:
            error = results[name].get("error", "Unknown")
            print(f"  - {name:25s} ({error})")

    print("\n" + "=" * 80)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
