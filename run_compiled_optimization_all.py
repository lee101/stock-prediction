#!/usr/bin/env python3
"""
Parallel optimization runner for all symbols using COMPILED models.

This script runs extensive hyperparameter searches across all available symbols,
using compiled Toto and Kronos models for maximum performance.
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()

HYPERPARAM_DIR = Path("hyperparams/best")
OUTPUT_DIR = Path("hyperparams/optimized_compiled")


def get_all_symbols() -> List[str]:
    """Get all symbols that have existing hyperparameter configs."""
    if not HYPERPARAM_DIR.exists():
        raise FileNotFoundError(f"Hyperparameter directory not found: {HYPERPARAM_DIR}")

    symbols = sorted([p.stem for p in HYPERPARAM_DIR.glob("*.json")])
    return symbols


def optimize_symbol(symbol: str, trials: int, model: str, eval_runs: int = 3) -> Dict:
    """
    Run optimization for a single symbol.

    Args:
        symbol: Stock symbol
        trials: Number of trials per model
        model: Which model to optimize (toto/kronos/both)
        eval_runs: Number of evaluation runs per config

    Returns:
        Result dictionary
    """
    start_time = time.time()

    try:
        # Run optimization as subprocess
        cmd = [
            "python",
            "optimize_compiled_models.py",
            "--symbol", symbol,
            "--trials", str(trials),
            "--model", model,
            "--eval-runs", str(eval_runs),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per symbol
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Parse results from output
            output_path = OUTPUT_DIR / f"{symbol}.json"
            if output_path.exists():
                with output_path.open("r") as f:
                    data = json.load(f)
                    return {
                        "symbol": symbol,
                        "status": "success",
                        "model": data.get("model", "unknown"),
                        "val_mae": data["validation"]["pct_return_mae"],
                        "test_mae": data["test"]["pct_return_mae"],
                        "improved": data["metadata"].get("previous_mae") is not None,
                        "elapsed_s": elapsed,
                    }
            else:
                # No improvement, config not updated
                return {
                    "symbol": symbol,
                    "status": "no_improvement",
                    "elapsed_s": elapsed,
                }
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return {
                "symbol": symbol,
                "status": "failed",
                "error": error_msg,
                "elapsed_s": elapsed,
            }

    except subprocess.TimeoutExpired:
        return {
            "symbol": symbol,
            "status": "timeout",
            "elapsed_s": time.time() - start_time,
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "status": "error",
            "error": str(e),
            "elapsed_s": time.time() - start_time,
        }


def run_parallel_optimization(
    symbols: List[str],
    trials: int = 100,
    max_workers: int = 2,  # Conservative - compiled models use more memory
    model: str = "toto",
    eval_runs: int = 3,
) -> List[Dict]:
    """
    Run optimization in parallel across multiple symbols.

    Args:
        symbols: List of symbols to optimize
        trials: Trials per model per symbol
        max_workers: Number of parallel workers (default: 2 for GPU memory)
        model: Which model to optimize (toto/kronos/both)
        eval_runs: Number of evaluation runs per config to reduce variance

    Returns:
        List of result dictionaries
    """
    results = []

    console.print(f"\n[bold cyan]Starting Parallel Compiled Model Optimization[/bold cyan]")
    console.print(f"Symbols: {len(symbols)}")
    console.print(f"Trials per model: {trials}")
    console.print(f"Eval runs per config: {eval_runs} (reduces variance)")
    console.print(f"Max workers: {max_workers}")
    console.print(f"Model: {model}")
    console.print(f"Output: {OUTPUT_DIR}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        task = progress.add_task(
            f"[cyan]Optimizing {len(symbols)} symbols...",
            total=len(symbols)
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(optimize_symbol, symbol, trials, model, eval_runs): symbol
                for symbol in symbols
            }

            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Update progress
                    progress.update(task, advance=1)

                    # Print status
                    if result["status"] == "success":
                        improved_icon = "🎯" if result.get("improved") else "✅"
                        console.print(
                            f"{improved_icon} [green]{symbol:10s}[/green] "
                            f"val_mae={result['val_mae']:.6f} "
                            f"test_mae={result['test_mae']:.6f} "
                            f"({result['elapsed_s']:.1f}s)"
                        )
                    elif result["status"] == "no_improvement":
                        console.print(
                            f"⏭️  [yellow]{symbol:10s}[/yellow] No improvement - skipped "
                            f"({result['elapsed_s']:.1f}s)"
                        )
                    elif result["status"] == "timeout":
                        console.print(
                            f"⏰ [yellow]{symbol:10s}[/yellow] Timeout after {result['elapsed_s']:.1f}s"
                        )
                    else:
                        console.print(
                            f"❌ [red]{symbol:10s}[/red] {result['status']}"
                        )

                except Exception as e:
                    console.print(f"❌ [red]{symbol:10s}[/red] Exception: {e}")
                    results.append({
                        "symbol": symbol,
                        "status": "exception",
                        "error": str(e),
                    })
                    progress.update(task, advance=1)

    return results


def print_summary(results: List[Dict]):
    """Print summary table of results."""
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]COMPILED MODEL OPTIMIZATION SUMMARY[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # Create summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("Status", width=15)
    table.add_column("Val MAE", justify="right", width=12)
    table.add_column("Test MAE", justify="right", width=12)
    table.add_column("Time (s)", justify="right", width=10)

    successful = []
    improved = []
    no_improvement = []
    failed = []

    for result in sorted(results, key=lambda x: x.get("val_mae", float("inf"))):
        symbol = result["symbol"]
        status = result["status"]

        if status == "success":
            successful.append(result)
            if result.get("improved"):
                improved.append(result)
            status_str = "[green]✓ Improved[/green]" if result.get("improved") else "[blue]✓ Success[/blue]"
            table.add_row(
                symbol,
                status_str,
                f"{result.get('val_mae', 0):.6f}",
                f"{result.get('test_mae', 0):.6f}",
                f"{result.get('elapsed_s', 0):.1f}",
            )
        elif status == "no_improvement":
            no_improvement.append(result)
            table.add_row(
                symbol,
                "[yellow]⏭ No improve[/yellow]",
                "—",
                "—",
                f"{result.get('elapsed_s', 0):.1f}",
            )
        else:
            failed.append(result)
            table.add_row(
                symbol,
                f"[red]✗ {status}[/red]",
                "—",
                "—",
                f"{result.get('elapsed_s', 0):.1f}",
            )

    console.print(table)

    # Print statistics
    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"  Total symbols: {len(results)}")
    console.print(f"  [green]Successful: {len(successful)}[/green]")
    console.print(f"  [bold green]Improved: {len(improved)}[/bold green]")
    console.print(f"  [yellow]No improvement: {len(no_improvement)}[/yellow]")
    console.print(f"  [red]Failed: {len(failed)}[/red]")

    if successful:
        total_time = sum(r.get("elapsed_s", 0) for r in successful)
        avg_time = total_time / len(successful)
        console.print(f"  Avg time per symbol: {avg_time:.1f}s")

        # Top 5 best MAE
        console.print(f"\n[bold]Top 5 Best Validation MAE:[/bold]")
        top5 = sorted(successful, key=lambda x: x.get("val_mae", float("inf")))[:5]
        for i, r in enumerate(top5, 1):
            improved_icon = "🎯" if r.get("improved") else "  "
            console.print(
                f"  {i}. {improved_icon} {r['symbol']:10s} val_mae={r['val_mae']:.6f} test_mae={r['test_mae']:.6f}"
            )

    if improved:
        console.print(f"\n[bold green]Improved Symbols ({len(improved)}):[/bold green]")
        for r in improved:
            console.print(f"  🎯 {r['symbol']:10s} val_mae={r['val_mae']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Run parallel optimization for compiled models")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Specific symbols to optimize (default: all)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Trials per model (default: 100 for extensive search)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2 for GPU memory)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["toto", "kronos", "both"],
        default="toto",
        help="Which model to optimize (default: toto)",
    )
    parser.add_argument(
        "--eval-runs",
        type=int,
        default=3,
        help="Number of evaluation runs per config to reduce variance (default: 3)",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        help="Save summary to JSON file",
    )
    args = parser.parse_args()

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = get_all_symbols()

    if not symbols:
        console.print("[red]No symbols found![/red]")
        return

    console.print(f"[bold]Found {len(symbols)} symbols:[/bold]")
    console.print(f"  {', '.join(symbols)}\n")

    # Run optimization
    start_time = time.time()
    results = run_parallel_optimization(
        symbols=symbols,
        trials=args.trials,
        max_workers=args.workers,
        model=args.model,
        eval_runs=args.eval_runs,
    )
    total_time = time.time() - start_time

    # Print summary
    print_summary(results)

    console.print(f"\n[bold]Total execution time: {total_time/60:.1f} minutes[/bold]")

    # Save summary if requested
    if args.save_summary:
        summary_path = Path(args.save_summary)
        with summary_path.open("w") as f:
            json.dump({
                "results": results,
                "total_time_s": total_time,
                "config": {
                    "symbols": symbols,
                    "trials": args.trials,
                    "workers": args.workers,
                    "model": args.model,
                    "eval_runs": args.eval_runs,
                },
            }, f, indent=2)
        console.print(f"\n📝 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
