#!/usr/bin/env python3
"""
CLI entrypoint for benchmarking neural trading strategies side-by-side.

Example:
    python -m experiments.run_neural_strategies \
        --config experiments/neural_strategies/configs/toto_distill_small.json \
        --config experiments/neural_strategies/configs/dual_attention_small.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from experiments.neural_strategies import get_experiment_class, list_registered_strategies

# Ensure strategies register themselves with the registry on import.
import experiments.neural_strategies.toto_distillation  # noqa: F401
import experiments.neural_strategies.dual_attention  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural trading strategy experiments.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to an experiment config JSON file. Can be repeated.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory containing experiment configs (all *.json files will be used).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON path to write the aggregated metrics table.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered strategies and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print("Registered strategies:")
        for key, value in list_registered_strategies().items():
            print(f"  - {key}: {value}")
        return

    config_paths = _gather_config_paths(args.config, args.config_dir)
    if not config_paths:
        raise SystemExit("No experiment configs provided. Use --config or --config-dir.")

    aggregated = []
    for path in config_paths:
        config = json.loads(Path(path).read_text())
        strategy = config.get("strategy")
        if strategy is None:
            raise ValueError(f"Missing 'strategy' field in config {path}")
        experiment_cls = get_experiment_class(strategy)
        experiment = experiment_cls(config=config, config_path=Path(path))
        result = experiment.run()
        aggregated.append(result)
        print(result.to_json())

    _print_summary_table(aggregated)
    if args.output:
        output_path = Path(args.output)
        payload = [json.loads(res.to_json()) for res in aggregated]
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote aggregated metrics to {output_path}")


def _gather_config_paths(configs: Iterable[str], config_dir: str | None) -> List[Path]:
    paths = [Path(c).expanduser() for c in configs]
    if config_dir:
        dir_path = Path(config_dir).expanduser()
        if not dir_path.exists():
            raise FileNotFoundError(f"Config directory '{dir_path}' not found")
        paths.extend(sorted(dir_path.glob("*.json")))
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _print_summary_table(results: List) -> None:
    if not results:
        return
    print("\n=== Experiment Summary ===")
    header = ["Name"] + sorted({k for res in results for k in res.metrics})
    print(" | ".join(f"{col:>20}" for col in header))
    for res in results:
        row = [res.name]
        for metric in header[1:]:
            value = res.metrics.get(metric)
            if value is None:
                row.append("n/a")
            else:
                row.append(f"{value:>.6f}")
        print(" | ".join(f"{col:>20}" for col in row))


if __name__ == "__main__":
    main()
