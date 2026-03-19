from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkRow:
    metric: str
    count: int
    python_mean: float
    fast_mean: float
    python_std: float
    fast_std: float
    mean_delta: float
    abs_diff_mean: float
    rel_diff_mean: float
    max_abs_diff: float


class DriftError(RuntimeError):
    """Raised when fast-env drift exceeds configured tolerances."""


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _to_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def load_rows(csv_path: Path) -> list[BenchmarkRow]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            BenchmarkRow(
                metric=row["metric"],
                count=_to_int(row.get("count")),
                python_mean=_to_float(row.get("python_mean")),
                fast_mean=_to_float(row.get("fast_mean")),
                python_std=_to_float(row.get("python_std")),
                fast_std=_to_float(row.get("fast_std")),
                mean_delta=_to_float(row.get("mean_delta")),
                abs_diff_mean=_to_float(row.get("abs_diff_mean")),
                rel_diff_mean=_to_float(row.get("rel_diff_mean")),
                max_abs_diff=_to_float(row.get("max_abs_diff")),
            )
            for row in reader
        ]
    if not rows:
        raise DriftError(f"No benchmark rows found in {csv_path}")
    return rows


def validate_rows(
    rows: Iterable[BenchmarkRow],
    *,
    max_abs_diff: float,
    max_rel_diff: float,
    runtime_slack: float,
) -> None:
    violations: list[str] = []
    for row in rows:
        if row.metric == "runtime_seconds":
            python_time = max(row.python_mean, 1e-9)
            ratio = row.fast_mean / python_time
            if ratio > 1.0 + runtime_slack:
                violations.append(
                    f"runtime drift for fast env ({ratio:.3f}x slower than python, allowed {(1.0 + runtime_slack):.3f}x)"
                )
            continue

        if row.metric == "observation_max_abs_diff":
            if row.max_abs_diff > max_abs_diff:
                violations.append(f"observation diff {row.max_abs_diff:.3e} exceeds max_abs_diff {max_abs_diff:.3e}")
            continue

        abs_ok = row.max_abs_diff <= max_abs_diff
        rel_ok = row.rel_diff_mean <= max_rel_diff
        if not (abs_ok or rel_ok):
            violations.append(
                f"{row.metric} drift abs={row.max_abs_diff:.3e} rel={row.rel_diff_mean:.3e} exceeds "
                f"thresholds (abs<={max_abs_diff:.3e} or rel<={max_rel_diff:.3e})"
            )

    if violations:
        message = "\n".join(["Fast env drift detected:", *(" - " + v for v in violations)])
        raise DriftError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fast-env benchmark output")
    parser.add_argument("--csv", type=Path, required=True, help="Path to bench_fast_vs_python CSV file")
    parser.add_argument(
        "--max-abs-diff",
        type=float,
        default=5e-4,
        help="Maximum tolerated absolute difference between env metrics",
    )
    parser.add_argument(
        "--max-rel-diff",
        type=float,
        default=2e-2,
        help="Maximum tolerated relative difference between env metrics",
    )
    parser.add_argument(
        "--runtime-slack",
        type=float,
        default=0.35,
        help="Allowed multiplicative slowdown for fast env vs python",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    validate_rows(
        rows,
        max_abs_diff=args.max_abs_diff,
        max_rel_diff=args.max_rel_diff,
        runtime_slack=args.runtime_slack,
    )
    print(
        "Fast env benchmark validated for "
        f"{len(rows)} rows (max_abs_diff<={args.max_abs_diff:.3e}, max_rel_diff<={args.max_rel_diff:.3e})"
    )


if __name__ == "__main__":
    main()
