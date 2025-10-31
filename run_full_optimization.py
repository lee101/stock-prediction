#!/usr/bin/env python3
"""
Parallel optimization runner for all stocks.

Runs comprehensive optimization (Toto, Kronos standard, Kronos ensemble)
across all stock pairs in parallel for maximum efficiency.
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()

DATA_DIR = Path("trainingdata")
OUTPUT_DIR = Path("hyperparams_optimized_all")


def get_all_symbols() -> List[str]:
    """Get all available stock symbols."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    symbols = sorted([p.stem for p in DATA_DIR.glob("*.csv")])
    return symbols


def optimize_symbol(symbol: str, trials: int, output_dir: str) -> Dict:
    """
    Run optimization for a single symbol.

    Args:
        symbol: Stock symbol
        trials: Number of trials per model
        output_dir: Output directory

    Returns:
        Result dictionary
    """
    start_time = time.time()

    try:
        # Run optimization as subprocess
        result = subprocess.run(
            [
                "python",
                "optimize_all_models.py",
                "--symbol", symbol,
                "--trials", str(trials),
                "--output-dir", output_dir,
            ],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per symbol
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Parse results
            output_path = Path(output_dir)
            best_mae = float("inf")
            best_model = None

            for model_type in ["toto", "kronos_standard", "kronos_ensemble"]:
                result_path = output_path / model_type / f"{symbol}.json"
                if result_path.exists():
                    with result_path.open("r") as f:
                        data = json.load(f)
                        mae = data["validation"]["pct_return_mae"]
                        if mae < best_mae:
                            best_mae = mae
                            best_model = model_type

            return {
                "symbol": symbol,
                "status": "success",
                "best_model": best_model,
                "best_mae": best_mae,
                "elapsed_s": elapsed,
            }
        else:
            return {
                "symbol": symbol,
                "status": "failed",
                "error": result.stderr[-500:] if result.stderr else "Unknown error",
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
    trials: int = 30,
    max_workers: int = 3,
    output_dir: str = str(OUTPUT_DIR),
) -> List[Dict]:
    """
    Run optimization in parallel across multiple symbols.

    Args:
        symbols: List of symbols to optimize
        trials: Trials per model per symbol
        max_workers: Number of parallel workers
        output_dir: Output directory

    Returns:
        List of result dictionaries
    """
    results = []

    console.print(f"\n[bold cyan]Starting Parallel Optimization[/bold cyan]")
    console.print(f"Symbols: {len(symbols)}")
    console.print(f"Trials per model: {trials}")
    console.print(f"Max workers: {max_workers}")
    console.print(f"Output: {output_dir}\n")

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
                executor.submit(optimize_symbol, symbol, trials, output_dir): symbol
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
                        console.print(
                            f"✅ [green]{symbol:10s}[/green] "
                            f"{result['best_model']:20s} "
                            f"mae={result['best_mae']:.6f} "
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
    console.print(f"[bold cyan]OPTIMIZATION SUMMARY[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # Create summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("Status", width=12)
    table.add_column("Best Model", width=20)
    table.add_column("MAE", justify="right", width=12)
    table.add_column("Time (s)", justify="right", width=10)

    successful = []
    failed = []

    for result in sorted(results, key=lambda x: x.get("best_mae", float("inf"))):
        symbol = result["symbol"]
        status = result["status"]

        if status == "success":
            successful.append(result)
            table.add_row(
                symbol,
                "[green]✓ Success[/green]",
                result.get("best_model", "N/A"),
                f"{result.get('best_mae', 0):.6f}",
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
    console.print(f"  [red]Failed: {len(failed)}[/red]")

    if successful:
        total_time = sum(r.get("elapsed_s", 0) for r in successful)
        avg_time = total_time / len(successful)
        console.print(f"  Avg time per symbol: {avg_time:.1f}s")

        # Model preference
        model_counts = {}
        for r in successful:
            model = r.get("best_model", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1

        console.print(f"\n[bold]Model Selection:[/bold]")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {model:20s}: {count}")

        # Top 5 best MAE
        console.print(f"\n[bold]Top 5 Best MAE:[/bold]")
        top5 = sorted(successful, key=lambda x: x.get("best_mae", float("inf")))[:5]
        for r in top5:
            console.print(
                f"  {r['symbol']:10s} {r['best_model']:20s} mae={r['best_mae']:.6f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Run parallel optimization across all stocks")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Specific symbols to optimize (default: all)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Trials per model type (default: 30)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory",
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

    # Run optimization
    start_time = time.time()
    results = run_parallel_optimization(
        symbols=symbols,
        trials=args.trials,
        max_workers=args.workers,
        output_dir=args.output_dir,
    )
    total_time = time.time() - start_time

    # Print summary
    print_summary(results)

    console.print(f"\n[bold]Total execution time: {total_time:.1f}s[/bold]")

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
                },
            }, f, indent=2)
        console.print(f"\n📝 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
