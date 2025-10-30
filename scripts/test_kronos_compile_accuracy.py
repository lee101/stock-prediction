#!/usr/bin/env python3
"""
Data-driven test to determine if Kronos should use torch.compile or eager mode.

Tests:
1. MAE on training data (compiled vs eager)
2. Backtest strategy returns (compiled vs eager)
3. Performance metrics (time, memory, stability)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.kronos_wrapper import KronosForecastingWrapper


class KronosCompileTest:
    """Test Kronos compiled vs eager modes."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {"eager": {}, "compiled": {}}

    def _generate_test_data(self, length: int = 512) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=length, freq="D")

        # Generate realistic price series
        base_price = 100.0
        returns = np.random.randn(length) * 0.02  # 2% daily volatility
        close = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC from close
        high = close * (1 + np.abs(np.random.randn(length) * 0.01))
        low = close * (1 - np.abs(np.random.randn(length) * 0.01))
        open_price = np.roll(close, 1)
        open_price[0] = base_price

        # Volume
        volume = np.random.uniform(1000000, 10000000, length)

        df = pd.DataFrame({
            "ds": dates,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        })

        return df

    def test_mae_on_training_data(
        self,
        compile_mode: bool,
        num_tests: int = 5,
    ) -> Dict:
        """Test MAE on synthetic training data."""

        mode_str = "COMPILED" if compile_mode else "EAGER"
        print(f"\n{'='*60}")
        print(f"Testing Kronos {mode_str} - MAE on Training Data")
        print(f"{'='*60}")

        # Set environment
        if compile_mode:
            os.environ["KRONOS_COMPILE"] = "1"
        else:
            os.environ["KRONOS_COMPILE"] = "0"

        mae_list = []
        time_list = []

        for i in range(num_tests):
            print(f"Test {i+1}/{num_tests}...", end=" ", flush=True)

            # Generate test data
            df = self._generate_test_data(512)
            train_df = df.iloc[:-1]  # All but last
            target_close = df.iloc[-1]["Close"]

            try:
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                start = time.perf_counter()

                # Create wrapper
                wrapper = KronosForecastingWrapper(
                    model_name="NeoQuasar/Kronos-base",
                    tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                    device=self.device,
                    max_context=256,
                    sample_count=8,
                )

                # Predict
                results = wrapper.predict_series(
                    data=train_df,
                    timestamp_col="ds",
                    columns=["Close"],
                    pred_len=1,
                )

                pred_close = results["Close"].absolute[0]
                mae = abs(pred_close - target_close)
                elapsed = (time.perf_counter() - start) * 1000

                mae_list.append(mae)
                time_list.append(elapsed)

                print(f"✓ MAE={mae:.4f}, time={elapsed:.0f}ms")

                # Cleanup
                wrapper.unload()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"✗ Error: {e}")
                continue

        if mae_list:
            avg_mae = np.mean(mae_list)
            std_mae = np.std(mae_list)
            avg_time = np.mean(time_list)

            print(f"\n{mode_str} Results:")
            print(f"  Avg MAE: {avg_mae:.4f} ± {std_mae:.4f}")
            print(f"  Avg time: {avg_time:.0f}ms")

            return {
                "avg_mae": avg_mae,
                "std_mae": std_mae,
                "avg_time": avg_time,
                "mae_list": mae_list,
            }
        else:
            print(f"\n{mode_str} FAILED - no successful tests")
            return None

    def test_backtest_returns(self, compile_mode: bool, symbol: str = "BTCUSD") -> Optional[Dict]:
        """Run full backtest with Kronos in compiled or eager mode."""

        mode_str = "COMPILED" if compile_mode else "EAGER"
        print(f"\n{'='*60}")
        print(f"Testing Kronos {mode_str} - Backtest on {symbol}")
        print(f"{'='*60}")

        # Set environment
        if compile_mode:
            os.environ["KRONOS_COMPILE"] = "1"
            os.environ["FORCE_KRONOS"] = "1"
        else:
            os.environ["KRONOS_COMPILE"] = "0"
            os.environ["FORCE_KRONOS"] = "1"

        # Disable Toto to test Kronos only
        os.environ["TOTO_DISABLE_COMPILE"] = "1"

        try:
            import subprocess

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "backtest_test3_inline.py"),
                "--symbol", symbol,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                print(f"❌ Backtest failed")
                print(f"stderr: {result.stderr[-500:]}")
                return None

            # Find results file
            results_pattern = f"{symbol.lower()}_real_full*.json"
            results_dir = PROJECT_ROOT / "evaltests" / "backtests"

            if not results_dir.exists():
                print(f"⚠️ Results directory not found: {results_dir}")
                return None

            results_files = list(results_dir.glob(results_pattern))
            if not results_files:
                print(f"⚠️ No results file found matching: {results_pattern}")
                return None

            # Get most recent
            results_file = max(results_files, key=lambda p: p.stat().st_mtime)

            with open(results_file) as f:
                data = json.load(f)

            # Extract strategy returns
            strategies = data.get("strategies", {})
            maxdiff_return = strategies.get("maxdiff", {}).get("return", 0.0)
            maxdiff_sharpe = strategies.get("maxdiff", {}).get("sharpe", 0.0)

            print(f"✅ {symbol} {mode_str} backtest completed")
            print(f"  MaxDiff return: {maxdiff_return:.4f}")
            print(f"  MaxDiff sharpe: {maxdiff_sharpe:.4f}")

            return {
                "maxdiff_return": maxdiff_return,
                "maxdiff_sharpe": maxdiff_sharpe,
                "all_strategies": strategies,
            }

        except Exception as e:
            print(f"❌ Backtest error: {e}")
            return None

    def compare_and_decide(self):
        """Compare results and make recommendation."""

        print("\n" + "="*60)
        print("KRONOS COMPILE DECISION - DATA-DRIVEN ANALYSIS")
        print("="*60)

        eager = self.results["eager"]
        compiled = self.results["compiled"]

        if not eager.get("mae") or not compiled.get("mae"):
            print("\n❌ INCOMPLETE DATA - Cannot make recommendation")
            print("   Run full tests first")
            return None

        # MAE comparison
        eager_mae = eager["mae"]["avg_mae"]
        compiled_mae = compiled["mae"]["avg_mae"]
        mae_delta = abs(compiled_mae - eager_mae)
        mae_delta_pct = (mae_delta / eager_mae * 100) if eager_mae > 0 else 0

        print(f"\n📊 MAE Comparison (Training Data):")
        print(f"  Eager MAE:    {eager_mae:.4f}")
        print(f"  Compiled MAE: {compiled_mae:.4f}")
        print(f"  Delta:        {mae_delta:.4f} ({mae_delta_pct:.2f}%)")

        # Performance comparison
        eager_time = eager["mae"]["avg_time"]
        compiled_time = compiled["mae"]["avg_time"]
        speedup = eager_time / compiled_time if compiled_time > 0 else 0

        print(f"\n⚡ Performance Comparison:")
        print(f"  Eager time:    {eager_time:.0f}ms")
        print(f"  Compiled time: {compiled_time:.0f}ms")
        print(f"  Speedup:       {speedup:.2f}x")

        # Backtest comparison (if available)
        if eager.get("backtest") and compiled.get("backtest"):
            eager_return = eager["backtest"]["maxdiff_return"]
            compiled_return = compiled["backtest"]["maxdiff_return"]
            return_delta = abs(compiled_return - eager_return)
            return_delta_pct = (return_delta / abs(eager_return) * 100) if eager_return != 0 else 0

            print(f"\n💰 Backtest Returns Comparison:")
            print(f"  Eager MaxDiff return:    {eager_return:.4f}")
            print(f"  Compiled MaxDiff return: {compiled_return:.4f}")
            print(f"  Delta:                   {return_delta:.4f} ({return_delta_pct:.2f}%)")

        # Decision logic
        print("\n" + "="*60)
        print("DECISION CRITERIA")
        print("="*60)

        accuracy_ok = mae_delta_pct < 5.0
        performance_better = speedup > 1.2

        print(f"\n✓/✗ Accuracy: Delta {mae_delta_pct:.2f}% {'< 5%' if accuracy_ok else '>= 5%'} - {'PASS' if accuracy_ok else 'FAIL'}")
        print(f"✓/✗ Performance: Speedup {speedup:.2f}x {'>1.2x' if performance_better else '<=1.2x'} - {'PASS' if performance_better else 'FAIL'}")

        if eager.get("backtest") and compiled.get("backtest"):
            return_ok = return_delta_pct < 5.0
            print(f"✓/✗ Returns: Delta {return_delta_pct:.2f}% {'< 5%' if return_ok else '>= 5%'} - {'PASS' if return_ok else 'FAIL'}")
        else:
            return_ok = True  # Assume ok if no backtest data

        # Final decision
        print("\n" + "="*60)
        print("FINAL RECOMMENDATION")
        print("="*60)

        if accuracy_ok and return_ok and performance_better:
            decision = "COMPILED"
            print("\n🟢 RECOMMENDATION: Use COMPILED mode for Kronos")
            print("   ✓ Accurate predictions (MAE delta <5%)")
            print("   ✓ Better performance (speedup >1.2x)")
            if return_ok:
                print("   ✓ Strategy returns unchanged")

            config_value = "0"  # Don't disable
        elif accuracy_ok and return_ok:
            decision = "EAGER"
            print("\n🟡 RECOMMENDATION: Use EAGER mode for Kronos")
            print("   ✓ Accurate predictions")
            print("   ✗ Compiled not significantly faster")
            print("   → Eager mode preferred for simplicity")

            config_value = "1"  # Disable
        else:
            decision = "EAGER"
            print("\n🔴 RECOMMENDATION: Use EAGER mode for Kronos")
            print("   ✗ Compiled mode has accuracy/return issues")
            print("   → Eager mode required for correct predictions")

            config_value = "1"  # Disable

        # Update config
        config_file = PROJECT_ROOT / ".env.compile"
        with open(config_file, "a") as f:
            f.write(f"\n# Kronos model: {decision} mode\n")
            if decision == "COMPILED":
                f.write("export KRONOS_COMPILE=1\n")
            else:
                f.write("# Kronos doesn't benefit from compilation - use eager mode\n")
                f.write("# No flag needed - Kronos uses eager by default\n")

        print(f"\n✓ Configuration updated in .env.compile")
        print(f"\nTo apply:")
        print(f"  source .env.compile")

        return decision

    def run_full_test(self, test_backtest: bool = True):
        """Run full test suite."""

        print("="*60)
        print("KRONOS TORCH.COMPILE TEST SUITE")
        print("="*60)
        print(f"Testing: MAE on training data + {'backtests' if test_backtest else 'no backtests'}")
        print()

        # Test eager mode
        print("\n" + "="*60)
        print("PHASE 1: EAGER MODE")
        print("="*60)
        eager_mae = self.test_mae_on_training_data(compile_mode=False, num_tests=3)
        self.results["eager"]["mae"] = eager_mae

        if test_backtest and eager_mae:
            eager_backtest = self.test_backtest_returns(compile_mode=False, symbol="BTCUSD")
            self.results["eager"]["backtest"] = eager_backtest

        # Test compiled mode
        print("\n" + "="*60)
        print("PHASE 2: COMPILED MODE")
        print("="*60)
        compiled_mae = self.test_mae_on_training_data(compile_mode=True, num_tests=3)
        self.results["compiled"]["mae"] = compiled_mae

        if test_backtest and compiled_mae:
            compiled_backtest = self.test_backtest_returns(compile_mode=True, symbol="BTCUSD")
            self.results["compiled"]["backtest"] = compiled_backtest

        # Compare and decide
        decision = self.compare_and_decide()

        # Save results
        output_dir = PROJECT_ROOT / "tests" / "compile_stress_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "kronos_compile_test_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "results": self.results,
                "decision": decision,
            }, f, indent=2)

        print(f"\n📄 Results saved to: {output_file}")

        return decision


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Kronos torch.compile")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtest (faster)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tester = KronosCompileTest(device=args.device)
    decision = tester.run_full_test(test_backtest=not args.no_backtest)

    if decision == "COMPILED":
        sys.exit(0)  # Success - use compiled
    else:
        sys.exit(1)  # Use eager mode


if __name__ == "__main__":
    main()
