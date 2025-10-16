"""
Command line entry point for running the regression benchmark across optimizers.

Usage:
    python -m traininglib.benchmark_cli --optimizers adamw shampoo muon --runs 3
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable, Sequence

from .benchmarking import RegressionBenchmark
from .optimizers import optimizer_registry


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare optimizers on a synthetic regression task.")
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["adamw", "adam", "shampoo", "muon", "lion", "adafactor"],
        help="Names registered in traininglib.optimizers (default: %(default)s).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of seeds to evaluate per optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs per run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for the synthetic regression benchmark.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=16,
        help="Input dimensionality of the synthetic dataset.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden layer size of the MLP.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=1,
        help="Output dimensionality.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="Number of synthetic samples per run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a text table.",
    )
    return parser.parse_args(argv)


def _format_table(results: dict[str, dict]) -> str:
    lines = []
    header = f"{'optimizer':<12} {'mean_loss':>12} {'std_dev':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, payload in results.items():
        mean_loss = payload["final_loss_mean"]
        std_loss = payload["final_loss_std"]
        lines.append(f"{name:<12} {mean_loss:12.6f} {std_loss:10.6f}")
    return "\n".join(lines)


def run_cli(argv: Sequence[str] | None = None) -> str:
    args = _parse_args(argv)
    missing = [name for name in args.optimizers if name.lower() not in optimizer_registry]
    if missing:
        available = ", ".join(sorted(optimizer_registry.names()))
        raise ValueError(f"Unknown optimizer(s): {missing}. Available: {available}")

    bench = RegressionBenchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_samples=args.num_samples,
    )
    results = bench.compare(args.optimizers, runs=args.runs)
    if args.json:
        output = json.dumps(results, indent=2)
    else:
        output = _format_table(results)
    print(output)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    run_cli(argv)


if __name__ == "__main__":  # pragma: no cover
    main()
