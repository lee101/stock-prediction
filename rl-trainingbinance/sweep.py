from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from train import build_arg_parser, train

TUNABLE_FIELDS = (
    "max_gross_leverage",
    "max_position_weight",
    "downside_penalty",
    "drawdown_penalty",
    "turnover_penalty",
    "concentration_penalty",
    "leverage_penalty",
    "smoothness_penalty",
    "volatility_penalty",
    "lr",
    "ent_coef",
)


def parse_csv_floats(value: str) -> list[float]:
    parts = [item.strip() for item in str(value).split(",")]
    return [float(item) for item in parts if item]


def build_sweep_configs(grid: dict[str, Iterable[float]]) -> list[dict[str, float]]:
    normalized = {name: list(values) for name, values in grid.items()}
    names = [name for name, values in normalized.items() if values]
    if not names:
        return [{}]
    value_lists = [normalized[name] for name in names]
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*value_lists)]


def _apply_overrides(base_args: argparse.Namespace, overrides: dict[str, float], *, output_dir: Path, seed: int) -> argparse.Namespace:
    payload = deepcopy(vars(base_args))
    payload.update(overrides)
    payload["output_dir"] = str(output_dir)
    payload["seed"] = int(seed)
    return argparse.Namespace(**payload)


def _flatten_row(
    run_id: str,
    overrides: dict[str, float],
    manifest: dict[str, Any],
    base_args: argparse.Namespace,
) -> dict[str, Any]:
    summary = dict(manifest.get("best_summary") or {})
    training_cfg = dict(manifest.get("training_config") or {})
    row: dict[str, Any] = {"run_id": run_id, **summary}
    window_summaries = dict(manifest.get("best_window_summaries") or {})
    for label, window_summary in window_summaries.items():
        safe_label = str(label).replace("-", "_")
        for key, value in dict(window_summary).items():
            row[f"{key}_{safe_label}"] = value
    for field in TUNABLE_FIELDS:
        row[field] = overrides.get(field, training_cfg.get(field, getattr(base_args, field, None)))
    return row


def _write_ranking(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred_columns = [
        "run_id",
        "score",
        "p10_total_return",
        "median_total_return",
        "median_sortino",
        "p90_max_drawdown",
        "median_volatility",
        "mean_turnover",
        *TUNABLE_FIELDS,
    ]
    extra_columns = sorted({key for row in rows for key in row.keys()} - set(preferred_columns))
    columns = preferred_columns + extra_columns
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def build_arg_parser_for_sweep() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid sweep for rl-trainingbinance risk and sizing hyperparameters.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--max-gross-leverages", default="3.0,5.0")
    parser.add_argument("--max-position-weights", default="0.75,1.0,1.25")
    parser.add_argument("--downside-penalties", default="2.0,4.0")
    parser.add_argument("--drawdown-penalties", default="0.5,1.0")
    parser.add_argument("--turnover-penalties", default="0.01")
    parser.add_argument("--concentration-penalties", default="0.02")
    parser.add_argument("--leverage-penalties", default="0.01")
    parser.add_argument("--smoothness-penalties", default="2.0,4.0")
    parser.add_argument("--volatility-penalties", default="0.1,0.2")
    parser.add_argument("--learning-rates", default="0.0003")
    parser.add_argument("--entropy-coefs", default="0.01")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sort-by", default="score")
    return parser


def main() -> None:
    parser = build_arg_parser_for_sweep()
    args, forwarded_args = parser.parse_known_args()

    train_parser = build_arg_parser(require_output_dir=False)
    forwarded_args = list(forwarded_args)
    if forwarded_args[:1] == ["--"]:
        forwarded_args = forwarded_args[1:]
    train_args = train_parser.parse_args(forwarded_args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    grid = {
        "max_gross_leverage": parse_csv_floats(args.max_gross_leverages),
        "max_position_weight": parse_csv_floats(args.max_position_weights),
        "downside_penalty": parse_csv_floats(args.downside_penalties),
        "drawdown_penalty": parse_csv_floats(args.drawdown_penalties),
        "turnover_penalty": parse_csv_floats(args.turnover_penalties),
        "concentration_penalty": parse_csv_floats(args.concentration_penalties),
        "leverage_penalty": parse_csv_floats(args.leverage_penalties),
        "smoothness_penalty": parse_csv_floats(args.smoothness_penalties),
        "volatility_penalty": parse_csv_floats(args.volatility_penalties),
        "lr": parse_csv_floats(args.learning_rates),
        "ent_coef": parse_csv_floats(args.entropy_coefs),
    }
    configs = build_sweep_configs(grid)
    if args.limit is not None:
        configs = configs[: max(int(args.limit), 0)]

    manifests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for idx, overrides in enumerate(configs, start=1):
        run_id = f"run_{idx:03d}"
        run_dir = output_root / run_id
        run_seed = int(args.seed) + (idx - 1) * int(args.seed_step)
        run_args = _apply_overrides(train_args, overrides, output_dir=run_dir, seed=run_seed)
        print(json.dumps({"run_id": run_id, "seed": run_seed, "overrides": overrides}, sort_keys=True))
        manifest = train(run_args)
        manifests.append({"run_id": run_id, "seed": run_seed, "overrides": overrides, "manifest": manifest})
        rows.append(_flatten_row(run_id, overrides, manifest, train_args))

    rows.sort(key=lambda row: float(row.get(args.sort_by, -float("inf"))), reverse=True)
    _write_ranking(output_root / "ranking.csv", rows)
    (output_root / "ranking.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    (output_root / "manifests.json").write_text(json.dumps(manifests, indent=2, sort_keys=True))
    if rows:
        print(json.dumps(rows[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
