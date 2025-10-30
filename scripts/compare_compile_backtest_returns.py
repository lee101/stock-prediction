#!/usr/bin/env python3
"""
Compare actual trading strategy returns between compiled and non-compiled modes.

This tests whether torch.compile affects real PnL by running full backtests
with both configurations and comparing strategy returns.

Usage:
    python scripts/compare_compile_backtest_returns.py --symbols BTCUSD ETHUSD
    python scripts/compare_compile_backtest_returns.py --quick
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_backtest(
    symbol: str,
    compiled: bool,
    output_suffix: str,
) -> Optional[Dict]:
    """
    Run a backtest with torch.compile enabled or disabled.

    Returns:
        Dictionary with backtest results or None if failed
    """
    env = os.environ.copy()

    # Configure compilation
    if compiled:
        env["TOTO_DISABLE_COMPILE"] = "0"
        env["TOTO_COMPILE_MODE"] = "max-autotune"
        mode_str = "COMPILED"
    else:
        env["TOTO_DISABLE_COMPILE"] = "1"
        mode_str = "EAGER"

    print(f"\n{'='*80}")
    print(f"Running {symbol} backtest in {mode_str} mode")
    print(f"{'='*80}")

    # Run backtest
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "backtest_test3_inline.py"),
        "--symbol", symbol,
    ]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"❌ Backtest failed for {symbol} {mode_str}")
            print(f"stderr: {result.stderr[-500:]}")  # Last 500 chars
            return None

        # Parse output to find results
        # Look for the backtest JSON file
        results_pattern = f"{symbol.lower()}_{output_suffix}.json"
        results_dir = PROJECT_ROOT / "evaltests" / "backtests"
        results_file = results_dir / results_pattern

        if not results_file.exists():
            print(f"⚠️ Results file not found: {results_file}")
            return None

        with open(results_file) as f:
            data = json.load(f)

        print(f"✅ {symbol} {mode_str} backtest completed")
        return data

    except subprocess.TimeoutExpired:
        print(f"❌ Backtest timeout for {symbol} {mode_str}")
        return None
    except Exception as e:
        print(f"❌ Backtest error for {symbol} {mode_str}: {e}")
        return None


def extract_strategy_metrics(data: Dict) -> Dict[str, Dict[str, float]]:
    """Extract strategy metrics from backtest results."""
    metrics = {}

    if "strategies" not in data:
        return metrics

    for strategy_name, strategy_data in data["strategies"].items():
        metrics[strategy_name] = {
            "return": strategy_data.get("return", 0.0),
            "sharpe": strategy_data.get("sharpe", 0.0),
            "final_day": strategy_data.get("final_day", 0.0),
        }

    return metrics


def compare_results(
    symbol: str,
    compiled_data: Optional[Dict],
    eager_data: Optional[Dict],
) -> Dict:
    """Compare compiled vs eager backtest results."""

    comparison = {
        "symbol": symbol,
        "compiled_available": compiled_data is not None,
        "eager_available": eager_data is not None,
        "strategies": {},
    }

    if not compiled_data or not eager_data:
        return comparison

    compiled_metrics = extract_strategy_metrics(compiled_data)
    eager_metrics = extract_strategy_metrics(eager_data)

    # Compare each strategy
    all_strategies = set(compiled_metrics.keys()) | set(eager_metrics.keys())

    for strategy in all_strategies:
        c_metrics = compiled_metrics.get(strategy, {})
        e_metrics = eager_metrics.get(strategy, {})

        comparison["strategies"][strategy] = {
            "compiled_return": c_metrics.get("return", 0.0),
            "eager_return": e_metrics.get("return", 0.0),
            "compiled_sharpe": c_metrics.get("sharpe", 0.0),
            "eager_sharpe": e_metrics.get("sharpe", 0.0),
            "return_delta": c_metrics.get("return", 0.0) - e_metrics.get("return", 0.0),
            "sharpe_delta": c_metrics.get("sharpe", 0.0) - e_metrics.get("sharpe", 0.0),
        }

    return comparison


def generate_report(comparisons: List[Dict], output_path: Path) -> None:
    """Generate markdown report comparing compiled vs eager backtest results."""

    lines = ["# Torch Compile Backtest Comparison Report", ""]
    lines.append("This report compares actual trading strategy returns between compiled and eager modes.")
    lines.append("")

    for comparison in comparisons:
        symbol = comparison["symbol"]
        lines.append(f"## {symbol}")
        lines.append("")

        if not comparison["compiled_available"] or not comparison["eager_available"]:
            lines.append("⚠️ **Incomplete data** - could not run both modes")
            lines.append("")
            continue

        # Strategy comparison table
        lines.append("### Strategy Performance Comparison")
        lines.append("")
        lines.append("| Strategy | Compiled Return | Eager Return | Δ Return | Compiled Sharpe | Eager Sharpe | Δ Sharpe |")
        lines.append("|----------|-----------------|--------------|----------|-----------------|--------------|----------|")

        strategies = comparison["strategies"]
        for strategy_name, metrics in sorted(strategies.items()):
            c_return = metrics["compiled_return"]
            e_return = metrics["eager_return"]
            return_delta = metrics["return_delta"]
            c_sharpe = metrics["compiled_sharpe"]
            e_sharpe = metrics["eager_sharpe"]
            sharpe_delta = metrics["sharpe_delta"]

            # Format with color indicators
            return_indicator = "🔴" if abs(return_delta) > 0.05 else "🟢" if abs(return_delta) > 0.01 else ""
            sharpe_indicator = "🔴" if abs(sharpe_delta) > 5.0 else "🟢" if abs(sharpe_delta) > 1.0 else ""

            lines.append(
                f"| {strategy_name} | {c_return:.4f} | {e_return:.4f} | {return_delta:+.4f} {return_indicator} | "
                f"{c_sharpe:.2f} | {e_sharpe:.2f} | {sharpe_delta:+.2f} {sharpe_indicator} |"
            )

        lines.append("")

        # Analysis
        lines.append("### Analysis")
        lines.append("")

        # Find max deltas
        max_return_delta = max(
            (abs(m["return_delta"]), name, m["return_delta"])
            for name, m in strategies.items()
        )
        max_sharpe_delta = max(
            (abs(m["sharpe_delta"]), name, m["sharpe_delta"])
            for name, m in strategies.items()
        )

        lines.append(f"**Max Return Delta**: {max_return_delta[1]} ({max_return_delta[2]:+.4f})")
        lines.append(f"**Max Sharpe Delta**: {max_sharpe_delta[1]} ({max_sharpe_delta[2]:+.2f})")
        lines.append("")

        # Quality assessment
        if max_return_delta[0] > 0.05:
            lines.append("🔴 **FAIL**: Significant return divergence (>5%) detected")
            lines.append("   - torch.compile is affecting strategy PnL")
            lines.append("   - **Recommendation**: Disable torch.compile for production")
        elif max_return_delta[0] > 0.01:
            lines.append("🟡 **WARNING**: Moderate return divergence (1-5%) detected")
            lines.append("   - Minor PnL impact from torch.compile")
            lines.append("   - **Recommendation**: Monitor closely or disable if risk-averse")
        else:
            lines.append("🟢 **PASS**: Minimal return divergence (<1%)")
            lines.append("   - torch.compile has minimal impact on strategy PnL")
            lines.append("   - **Recommendation**: Safe to use if performance benefits outweigh costs")

        lines.append("")

    # Overall recommendation
    lines.append("## Overall Recommendation")
    lines.append("")

    # Check if any symbol had significant issues
    has_fail = any(
        max(abs(m["return_delta"]) for m in c["strategies"].values()) > 0.05
        for c in comparisons
        if c["strategies"]
    )

    has_warning = any(
        max(abs(m["return_delta"]) for m in c["strategies"].values()) > 0.01
        for c in comparisons
        if c["strategies"]
    )

    if has_fail:
        lines.append("🔴 **DO NOT USE** torch.compile in production")
        lines.append("")
        lines.append("**Reason**: Significant strategy PnL divergence detected (>5%)")
        lines.append("")
        lines.append("**Action**:")
        lines.append("```bash")
        lines.append("export TOTO_DISABLE_COMPILE=1")
        lines.append("python trade_stock_e2e.py")
        lines.append("```")
    elif has_warning:
        lines.append("🟡 **USE WITH CAUTION** - torch.compile may affect PnL")
        lines.append("")
        lines.append("**Reason**: Moderate strategy PnL divergence detected (1-5%)")
        lines.append("")
        lines.append("**Options**:")
        lines.append("1. Disable for safety: `export TOTO_DISABLE_COMPILE=1`")
        lines.append("2. Monitor closely if using compiled mode")
        lines.append("3. Run stress tests regularly to detect drift")
    else:
        lines.append("🟢 **SAFE TO USE** torch.compile (if performance benefits)")
        lines.append("")
        lines.append("**Reason**: Minimal strategy PnL divergence detected (<1%)")
        lines.append("")
        lines.append("**Note**: Performance benefits must outweigh:")
        lines.append("- Recompilation overhead")
        lines.append("- Compilation time on first run")
        lines.append("- Increased memory usage")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by {Path(__file__).name}*")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare backtest returns between compiled and eager modes"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSD", "ETHUSD"],
        help="Symbols to test (default: BTCUSD ETHUSD)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with BTCUSD only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaltests" / "compile_backtest_comparison.md",
        help="Output report path",
    )

    args = parser.parse_args()

    if args.quick:
        symbols = ["BTCUSD"]
    else:
        symbols = args.symbols

    print("=" * 80)
    print("TORCH COMPILE BACKTEST COMPARISON")
    print("=" * 80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Output: {args.output}")
    print()

    comparisons = []

    for symbol in symbols:
        # Run eager mode
        eager_data = run_backtest(
            symbol=symbol,
            compiled=False,
            output_suffix="real_full_eager",
        )

        # Run compiled mode
        compiled_data = run_backtest(
            symbol=symbol,
            compiled=True,
            output_suffix="real_full_compiled",
        )

        # Compare
        comparison = compare_results(symbol, compiled_data, eager_data)
        comparisons.append(comparison)

        # Print quick summary
        if comparison["strategies"]:
            print(f"\n📊 {symbol} Quick Summary:")
            for strategy, metrics in comparison["strategies"].items():
                return_delta = metrics["return_delta"]
                indicator = "🔴" if abs(return_delta) > 0.05 else "🟢" if abs(return_delta) < 0.01 else "🟡"
                print(f"  {indicator} {strategy}: Δ return = {return_delta:+.4f}")

    # Generate report
    generate_report(comparisons, args.output)

    # Print final recommendation
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    has_fail = any(
        max(abs(m["return_delta"]) for m in c["strategies"].values()) > 0.05
        for c in comparisons
        if c["strategies"]
    )

    if has_fail:
        print("🔴 DO NOT USE torch.compile in production")
        print("   Significant PnL divergence detected (>5%)")
        sys.exit(1)
    else:
        print("🟢 torch.compile appears safe (PnL divergence <5%)")
        print("   Review report for details")
        sys.exit(0)


if __name__ == "__main__":
    main()
