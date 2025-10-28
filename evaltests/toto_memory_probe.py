"""Utility to profile Toto inference GPU memory usage across parameter sweeps."""
import argparse
from itertools import product
from typing import Iterable, List, Tuple

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest_test3_inline import load_toto_pipeline, profile_toto_memory, resolve_toto_params


def _parse_range(value: str) -> List[int]:
    parts = value.split(":")
    if len(parts) == 1:
        return [int(parts[0])]
    if len(parts) == 2:
        start, stop = map(int, parts)
        step = 1
    elif len(parts) == 3:
        start, stop, step = map(int, parts)
        if step == 0:
            raise ValueError("range step cannot be zero")
    else:
        raise ValueError(f"invalid range specification: {value}")
    if start <= stop:
        return list(range(start, stop + step, step))
    return list(range(start, stop - step, -step))


def _expand(values: Iterable[str] | None) -> List[int]:
    if not values:
        return []
    expanded: List[int] = []
    for raw in values:
        expanded.extend(_parse_range(raw))
    return sorted(set(expanded))


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile Toto inference GPU memory usage.")
    parser.add_argument("symbol", nargs="?", default="AAPL", help="Symbol to resolve default parameters for.")
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        action="append",
        help="Num samples to test (value or start:stop:step). Repeatable.",
    )
    parser.add_argument(
        "--samples-per-batch",
        dest="samples_per_batch",
        action="append",
        help="Samples per batch to test (value or start:stop:step). Repeatable.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of repeat runs per combination.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Synthetic context length supplied to the pipeline during profiling.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=7,
        help="Prediction horizon passed to Toto during profiling.",
    )
    args = parser.parse_args(argv)

    load_toto_pipeline()  # Ensure the model is resident before profiling

    combos: List[Tuple[int, int]] = []
    ns_values = _expand(args.num_samples)
    spb_values = _expand(args.samples_per_batch)

    if ns_values and spb_values:
        combos = list(product(ns_values, spb_values))
    else:
        defaults = resolve_toto_params(args.symbol)
        default_combo = (int(defaults["num_samples"]), int(defaults["samples_per_batch"]))
        combos = [default_combo]
        if ns_values:
            combos = [(ns, default_combo[1]) for ns in ns_values]
        if spb_values:
            combos = [(default_combo[0], spb) for spb in spb_values]

    print("Profiling Toto GPU memory usage for", args.symbol)
    print("runs", args.runs, "context_length", args.context_length, "prediction_length", args.prediction_length)
    header = f"{'num_samples':>12} {'samples/batch':>14} {'peak MB':>10} {'delta MB':>10} {'runs':>6}"
    print(header)
    print("-" * len(header))

    for num_samples, samples_per_batch in combos:
        summary = profile_toto_memory(
            symbol=args.symbol,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            runs=args.runs,
            reset_between_runs=True,
        )
        print(
            f"{summary['num_samples']:12d} {summary['samples_per_batch']:14d}"
            f" {summary['peak_mb']:10.2f} {summary['delta_mb']:10.2f} {summary['runs']:6d}"
        )


if __name__ == "__main__":
    main()
