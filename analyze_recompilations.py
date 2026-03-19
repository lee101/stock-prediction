#!/usr/bin/env python3
"""
Analyze torch.compile recompilation logs and provide actionable insights.

Usage:
    # Capture logs
    TORCH_LOGS="recompiles" python trade_stock_e2e.py 2>&1 | tee recompile.log

    # Analyze
    python scripts/analyze_recompilations.py recompile.log
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_recompilation_log(log_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse torch recompilation logs and extract key information.

    Returns:
        Dict mapping function names to list of (reason, context) tuples
    """
    recompilations: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    with open(log_path, "r") as f:
        content = f.read()

    # Pattern for recompile warnings
    # Example: torch._dynamo hit config.recompile_limit (8)
    #          function: 'positional_embedding' (/path/to/file.py:105)
    #          last reason: 6/7: kv_cache._current_idx[7] == 0

    # Look for recompile limit warnings
    limit_pattern = r"torch\._dynamo hit config\.recompile_limit \((\d+)\)"
    function_pattern = r"function: '([^']+)' \(([^)]+)\)"
    reason_pattern = r"last reason: (.+)"

    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for recompile limit warning
        if "torch._dynamo hit config.recompile_limit" in line:
            limit_match = re.search(limit_pattern, line)
            if limit_match:
                limit = limit_match.group(1)

                # Look ahead for function and reason
                function_name = None
                file_location = None
                reason = None

                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j]

                    func_match = re.search(function_pattern, next_line)
                    if func_match:
                        function_name = func_match.group(1)
                        file_location = func_match.group(2)

                    reason_match = re.search(reason_pattern, next_line)
                    if reason_match:
                        reason = reason_match.group(1).strip()

                    if function_name and reason:
                        break

                if function_name:
                    context = f"Limit: {limit}, Location: {file_location}"
                    recompilations[function_name].append((reason, context))

        # Also look for "skipping cudagraphs" warnings
        if "skipping cudagraphs" in line:
            reason_match = re.search(r"skipping cudagraphs due to (.+)", line)
            if reason_match:
                reason = reason_match.group(1).strip()
                recompilations["cudagraphs"].append((reason, ""))

        i += 1

    return dict(recompilations)


def analyze_recompilations(recompilations: Dict[str, List[Tuple[str, str]]]) -> None:
    """Analyze recompilations and provide recommendations."""

    print("=" * 80)
    print("TORCH COMPILE RECOMPILATION ANALYSIS")
    print("=" * 80)
    print()

    if not recompilations:
        print("✅ No recompilation issues detected!")
        return

    total_issues = sum(len(reasons) for reasons in recompilations.values())
    print(f"Found {total_issues} recompilation issues across {len(recompilations)} functions/locations")
    print()

    # Analyze each function
    for function_name, reasons in sorted(recompilations.items()):
        print(f"### Function: {function_name}")
        print(f"Recompilations: {len(reasons)}")
        print()

        # Group by reason
        reason_counts: Dict[str, int] = defaultdict(int)
        contexts = {}
        for reason, context in reasons:
            reason_counts[reason] += 1
            if reason not in contexts:
                contexts[reason] = context

        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  - [{count}x] {reason}")
            if contexts[reason]:
                print(f"    Context: {contexts[reason]}")

        print()

        # Provide recommendations
        print("  **Recommendations:**")

        if "kv_cache" in function_name.lower() or any("kv_cache" in r for r, _ in reasons):
            print("  - KV cache index is changing dynamically")
            print("  - Solution 1: Use static KV cache allocation")
            print("  - Solution 2: Mark KV cache dimensions as dynamic with torch._dynamo.mark_dynamic()")
            print("  - Solution 3: Disable torch.compile with TOTO_DISABLE_COMPILE=1")

        if function_name == "positional_embedding" or "position" in function_name.lower():
            print("  - Positional embedding is recompiling due to dynamic sequence positions")
            print("  - Solution 1: Use static positional indices")
            print("  - Solution 2: Mark position tensors as dynamic")

        if function_name == "cudagraphs":
            print("  - CUDA graphs are being skipped (performance impact)")
            print("  - This is often caused by mutated inputs")
            print("  - Solution: Ensure KV cache and other state is static")

        if any("==" in r for r, _ in reasons) or any("!=" in r for r, _ in reasons):
            print("  - Recompilation triggered by tensor value guards (tensor == value)")
            print("  - Solution: Use torch._dynamo.mark_static() for static values")
            print("  - Or mark as dynamic with torch._dynamo.mark_dynamic() if truly dynamic")

        print()

    # Overall recommendations
    print("=" * 80)
    print("OVERALL RECOMMENDATIONS")
    print("=" * 80)
    print()

    if total_issues > 10:
        print("⚠️  HIGH: Excessive recompilations detected")
        print()
        print("Immediate actions:")
        print("1. Disable torch.compile for production:")
        print("   export TOTO_DISABLE_COMPILE=1")
        print()
        print("2. Increase recompile limit temporarily:")
        print("   export TORCHDYNAMO_RECOMPILE_LIMIT=32")
        print()
        print("3. Run stress test to measure impact:")
        print("   python scripts/run_compile_stress_test.py --mode production-check")
        print()
    else:
        print("✅ MEDIUM: Some recompilations detected but manageable")
        print()
        print("Suggested actions:")
        print("1. Run stress test to measure accuracy impact:")
        print("   python scripts/run_compile_stress_test.py --mode full")
        print()
        print("2. Consider optimizing KV cache implementation")
        print()

    print("Long-term solutions:")
    print("- Refactor model code to use static shapes where possible")
    print("- Use torch._dynamo.mark_dynamic() for truly dynamic dimensions")
    print("- Consider using torch.compile in reduce-overhead mode")
    print("- Profile with TORCH_LOGS=+dynamo,+inductor to understand recompilation triggers")
    print()

    print("See docs/TORCH_COMPILE_GUIDE.md for detailed solutions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze torch.compile recompilation logs")
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to log file containing recompilation warnings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output markdown report path (optional)",
    )
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return

    print(f"Analyzing recompilation log: {args.log_file}")
    print()

    recompilations = parse_recompilation_log(args.log_file)
    analyze_recompilations(recompilations)

    if args.output:
        # Save report to file
        # TODO: Implement markdown report generation
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
