#!/usr/bin/env python3
"""
Convenience script to run compile stress tests with various configurations.

Usage:
    python scripts/run_compile_stress_test.py --mode full --iterations 10
    python scripts/run_compile_stress_test.py --mode quick --model toto
    python scripts/run_compile_stress_test.py --mode production-check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_compile_integration_stress import CompileStressTestRunner
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compile integration stress tests")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "production-check"],
        default="quick",
        help="Test mode: quick (3 iter), full (10 iter), production-check (20 iter)",
    )
    parser.add_argument(
        "--model",
        choices=["toto", "kronos", "both"],
        default="both",
        help="Which model to test",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override number of iterations",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length for test series",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of samples for predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine iterations based on mode
    if args.iterations:
        num_iterations = args.iterations
    else:
        mode_iterations = {
            "quick": 3,
            "full": 10,
            "production-check": 20,
        }
        num_iterations = mode_iterations[args.mode]

    print(f"Running compile stress test in {args.mode.upper()} mode")
    print(f"Device: {args.device}")
    print(f"Iterations: {num_iterations}")
    print(f"Model(s): {args.model}")
    print()

    runner = CompileStressTestRunner(
        device=args.device,
        num_iterations=num_iterations,
        context_length=args.context_length,
        pred_length=1,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )

    all_results = []

    # Test Toto
    if args.model in ["toto", "both"]:
        print("\n" + "=" * 80)
        print("TOTO MODEL STRESS TEST")
        print("=" * 80)

        series = runner._generate_synthetic_series(args.context_length)
        targets = [series[-1] * 1.01]  # Predict 1% increase

        try:
            compiled_results, eager_results = runner.test_toto_compiled_vs_eager(
                series, targets
            )
            all_results.extend(compiled_results + eager_results)
            print(f"\n✅ Toto test completed: {len(compiled_results)} compiled, {len(eager_results)} eager")
        except Exception as e:
            print(f"\n❌ Toto test failed: {e}")
            if args.mode == "production-check":
                raise

    # Test Kronos
    if args.model in ["kronos", "both"]:
        print("\n" + "=" * 80)
        print("KRONOS MODEL STRESS TEST")
        print("=" * 80)

        series = runner._generate_synthetic_series(args.context_length)
        df = pd.DataFrame({
            "ds": pd.date_range("2020-01-01", periods=len(series), freq="D"),
            "Close": series,
        })
        targets = [series[-1] * 1.01]

        try:
            _, kronos_results = runner.test_kronos_compiled_vs_eager(df, targets)
            all_results.extend(kronos_results)
            print(f"\n✅ Kronos test completed: {len(kronos_results)} iterations")
        except Exception as e:
            print(f"\n❌ Kronos test failed: {e}")
            # Kronos failures are acceptable as it may not be available
            if args.mode != "production-check":
                print("Continuing without Kronos results...")

    # Save results
    if all_results:
        output_suffix = f"_{args.mode}" if args.mode != "quick" else ""
        runner.save_results(
            all_results,
            f"compile_stress_results{output_suffix}.json"
        )
        runner.generate_report(
            all_results,
            f"compile_stress_report{output_suffix}.md"
        )

        print("\n" + "=" * 80)
        print("STRESS TEST COMPLETE")
        print("=" * 80)
        print(f"Total test runs: {len(all_results)}")
        print(f"Results saved to {runner.output_dir}")

        # Production check validation
        if args.mode == "production-check":
            print("\n" + "=" * 80)
            print("PRODUCTION READINESS CHECK")
            print("=" * 80)

            # Check for critical issues
            import numpy as np

            issues = []

            # Group by model and mode
            toto_compiled = [r for r in all_results if r.model_name == "Toto" and r.compile_mode != "eager"]
            toto_eager = [r for r in all_results if r.model_name == "Toto" and r.compile_mode == "eager"]

            if toto_compiled and toto_eager:
                # Check MAE divergence
                compiled_mae = np.mean([r.accuracy.mae for r in toto_compiled])
                eager_mae = np.mean([r.accuracy.mae for r in toto_eager])
                mae_delta_pct = abs(compiled_mae - eager_mae) / eager_mae * 100 if eager_mae != 0 else 0

                print(f"\nToto MAE Delta: {mae_delta_pct:.2f}%")
                if mae_delta_pct > 5.0:
                    issues.append(f"❌ Toto MAE divergence too high: {mae_delta_pct:.2f}%")
                else:
                    print(f"✅ Toto MAE within acceptable range")

                # Check recompilations
                total_recompiles = sum([r.performance.recompilations for r in toto_compiled])
                print(f"Toto total recompilations: {total_recompiles}")
                if total_recompiles > num_iterations * 10:
                    issues.append(f"⚠️  Toto excessive recompilations: {total_recompiles}")

                # Check performance
                compiled_time = np.mean([r.performance.inference_time_ms for r in toto_compiled])
                eager_time = np.mean([r.performance.inference_time_ms for r in toto_eager])
                speedup = eager_time / compiled_time if compiled_time > 0 else 0

                print(f"Toto compiled avg time: {compiled_time:.2f}ms")
                print(f"Toto eager avg time: {eager_time:.2f}ms")
                print(f"Speedup: {speedup:.2f}x")

                if speedup < 0.8:  # Compiled is slower
                    issues.append(f"⚠️  Toto compiled slower than eager: {speedup:.2f}x")

            if issues:
                print("\n" + "=" * 80)
                print("PRODUCTION READINESS: FAILED")
                print("=" * 80)
                for issue in issues:
                    print(issue)
                print("\nRecommendation: Consider running in eager mode for production")
                sys.exit(1)
            else:
                print("\n" + "=" * 80)
                print("PRODUCTION READINESS: PASSED ✅")
                print("=" * 80)
                print("Models are ready for production with torch.compile")
                sys.exit(0)


if __name__ == "__main__":
    main()
