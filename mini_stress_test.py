#!/usr/bin/env python3
"""
Mini stress test - runs a subset of scenarios to find issues faster.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _random_walk(n_points, volatility=0.02, scale=100.0):
    """Generate random walk data."""
    np.random.seed(42)
    returns = np.random.randn(n_points) * volatility
    prices = scale * np.exp(np.cumsum(returns))
    return prices


def _with_jumps(n_points):
    """Generate data with sudden jumps."""
    np.random.seed(42)
    returns = np.random.randn(n_points) * 0.02
    jump_indices = [n_points // 4, n_points // 2, 3 * n_points // 4]
    for idx in jump_indices:
        if 0 <= idx < n_points:
            returns[idx] += np.random.choice([-0.3, 0.3])
    prices = 100.0 * np.exp(np.cumsum(returns))
    return prices


def _near_zero(n_points):
    """Generate values very close to zero."""
    np.random.seed(42)
    prices = np.abs(np.random.randn(n_points) * 1e-4 + 1e-3)
    return prices


def _with_outliers(n_points):
    """Generate data with extreme outliers."""
    np.random.seed(42)
    returns = np.random.randn(n_points) * 0.02
    outlier_indices = [n_points // 3, 2 * n_points // 3]
    for idx in outlier_indices:
        if 0 <= idx < n_points:
            returns[idx] = np.random.choice([-0.8, 0.8])
    prices = 100.0 * np.exp(np.cumsum(returns))
    return prices


def _build_scenarios():
    return {
        "normal": lambda n: _random_walk(n, volatility=0.02),
        "high_vol": lambda n: _random_walk(n, volatility=0.15),
        "very_small": lambda n: _random_walk(n, volatility=0.02, scale=1e-4),
        "jumps": lambda n: _with_jumps(n),
        "near_zero": lambda n: _near_zero(n),
        "outliers": lambda n: _with_outliers(n),
    }


def create_df(prices, n_points):
    """Convert prices to OHLC dataframe."""
    opens = prices * (1 + np.random.randn(n_points) * 0.002)
    closes = prices * (1 + np.random.randn(n_points) * 0.002)
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_points)) * 0.005)
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_points)) * 0.005)

    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n_points, freq="D"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "symbol": "TEST",
    })


_CHRONOS2_WRAPPER = None


def _get_chronos2_wrapper():
    global _CHRONOS2_WRAPPER
    if _CHRONOS2_WRAPPER is None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.models.chronos2_wrapper import Chronos2OHLCWrapper as _Wrapper

        _CHRONOS2_WRAPPER = _Wrapper
    return _CHRONOS2_WRAPPER


def test_scenario(name, data_fn, iterations=3):
    """Test a scenario multiple times."""
    print(f"\nTesting: {name}")
    print("-" * 60)

    failures = []
    mae_diffs = []
    Chronos2OHLCWrapper = _get_chronos2_wrapper()

    for i in range(iterations):
        try:
            # Create data
            n_points = 160
            prices = data_fn(n_points)
            df = create_df(prices, n_points)
            context = df.iloc[:-16]

            # Eager
            eager_wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id="amazon/chronos-2",
                device_map=device,
                default_context_length=128,
                torch_compile=False,
            )

            eager_result = eager_wrapper.predict_ohlc(
                context_df=context,
                symbol="TEST",
                prediction_length=16,
                context_length=128,
            )
            eager_preds = eager_result.median["close"].values
            eager_wrapper.unload()

            # Compiled
            compiled_wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id="amazon/chronos-2",
                device_map=device,
                default_context_length=128,
                torch_compile=True,
                compile_mode="reduce-overhead",
            )

            compiled_result = compiled_wrapper.predict_ohlc(
                context_df=context,
                symbol="TEST",
                prediction_length=16,
                context_length=128,
            )
            compiled_preds = compiled_result.median["close"].values
            compiled_wrapper.unload()

            # Compare
            if np.isnan(eager_preds).any() or np.isnan(compiled_preds).any():
                failures.append(f"Iter {i+1}: NaN in predictions")
                continue

            if np.isinf(eager_preds).any() or np.isinf(compiled_preds).any():
                failures.append(f"Iter {i+1}: Inf in predictions")
                continue

            mae_diff = float(np.mean(np.abs(eager_preds - compiled_preds)))
            mae_diffs.append(mae_diff)

            if mae_diff > 0.01:
                failures.append(f"Iter {i+1}: Large MAE diff {mae_diff:.6f}")
            else:
                print(f"  Iter {i+1}: ✓ MAE={mae_diff:.6f}")

            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            failures.append(f"Iter {i+1}: {str(e)[:80]}")
            print(f"  Iter {i+1}: ✗ {str(e)[:80]}")

    # Summary
    if failures:
        print(f"  ❌ {len(failures)}/{iterations} failed")
        for fail in failures:
            print(f"     - {fail}")
    else:
        avg_mae = np.mean(mae_diffs) if mae_diffs else 0
        print(f"  ✅ All passed (avg MAE: {avg_mae:.6f})")

    return len(failures) == 0


def main() -> int:
    print("=" * 60)
    print("Mini Stress Test for Chronos2 Compilation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    scenarios = _build_scenarios()

    # Run tests
    results = {}
    for name, data_fn in scenarios.items():
        results[name] = test_scenario(name, data_fn, iterations=3)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Passed: {passed}/{total}")
    print()

    for name, scenario_passed in results.items():
        status = "✅" if scenario_passed else "❌"
        print(f"  {status} {name}")

    if passed == total:
        print("\n✅ All scenarios passed!")
        return 0
    print(f"\n❌ {total - passed} scenario(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
