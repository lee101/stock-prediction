from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from .config import DataConfig, EnvironmentConfig, EvaluationConfig, TrainingConfig
from .trainer import DifferentiableMarketTrainer
from .utils import ensure_dir


DEFAULT_GRID: Dict[str, List[object]] = {
    "train.lookback": [96, 128],
    "train.batch_windows": [32, 48],
    "train.rollout_groups": [2, 4],
    "train.epochs": [300, 500],
    "env.risk_aversion": [0.05, 0.1],
    "env.drawdown_lambda": [0.0, 0.05],
    "train.include_cash": [False, True],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated hyper-parameter experiment runner for the differentiable market trainer.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("trainingdata"), help="Path to OHLC CSV directory.")
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path("differentiable_market") / "experiment_runs",
        help="Directory where experiment outputs are written.",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        help="Optional JSON file describing the search grid. Keys follow the pattern 'train.lookback', 'env.risk_aversion', etc.",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        help="Optional JSON file with baseline config blocks: {'data': {...}, 'env': {...}, 'train': {...}, 'eval': {...}}.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the trial order (helpful when you expect to interrupt the job).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional limit on the number of experiments to run after shuffling/cardinality.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Override evaluation interval for every experiment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for shuffling and as the default training seed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved experiment plan without executing any training.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional annotation string stored with each experiment summary.",
    )
    return parser.parse_args()


def load_grid(path: Path | None) -> Dict[str, List[object]]:
    if path is None:
        return DEFAULT_GRID
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Grid JSON must be an object.")
    grid: Dict[str, List[object]] = {}
    for key, value in payload.items():
        if not isinstance(value, list) or not value:
            raise ValueError(f"Grid entry '{key}' must be a non-empty list.")
        grid[key] = value
    return grid


def load_baselines(path: Path | None) -> Tuple[DataConfig, EnvironmentConfig, TrainingConfig, EvaluationConfig]:
    data_cfg = DataConfig()
    env_cfg = EnvironmentConfig()
    train_cfg = TrainingConfig()
    eval_cfg = EvaluationConfig()
    if path is None:
        return data_cfg, env_cfg, train_cfg, eval_cfg
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Baseline config must be a JSON object.")
    for block_name, cfg in (
        ("data", data_cfg),
        ("env", env_cfg),
        ("train", train_cfg),
        ("eval", eval_cfg),
    ):
        block = payload.get(block_name)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"Baseline block '{block_name}' must be an object.")
        for key, value in block.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"{block_name} config has no attribute '{key}'")
            setattr(cfg, key, value)
    return data_cfg, env_cfg, train_cfg, eval_cfg


def iter_trials(grid: Dict[str, List[object]], seed: int, shuffle: bool) -> Iterator[Dict[str, object]]:
    keys = sorted(grid.keys())
    combos = [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]
    if shuffle:
        random.Random(seed).shuffle(combos)
    for combo in combos:
        yield combo


def apply_overrides(
    data_cfg: DataConfig,
    env_cfg: EnvironmentConfig,
    train_cfg: TrainingConfig,
    eval_cfg: EvaluationConfig,
    overrides: Dict[str, object],
) -> None:
    for key, value in overrides.items():
        if "." not in key:
            raise ValueError(f"Override key '{key}' must begin with 'data.', 'env.', 'train.', or 'eval.'")
        prefix, attr = key.split(".", 1)
        if prefix == "data":
            target = data_cfg
        elif prefix == "env":
            target = env_cfg
        elif prefix == "train":
            target = train_cfg
        elif prefix == "eval":
            target = eval_cfg
        else:
            raise ValueError(f"Unknown override prefix '{prefix}'")
        if not hasattr(target, attr):
            raise AttributeError(f"{prefix} config has no attribute '{attr}'")
        current_value = getattr(target, attr, None)
        if (
            attr in {"init_checkpoint", "save_dir", "cache_dir"}
            or attr.endswith("_dir")
            or attr.endswith("_path")
            or attr.endswith("_root")
        ):
            if value is None or value == "":
                coerced = None
            else:
                coerced = Path(value)
        elif attr == "wandb_tags":
            if value is None:
                coerced = ()
            elif isinstance(value, (list, tuple, set)):
                coerced = tuple(value)
            else:
                coerced = tuple(str(v).strip() for v in str(value).split(",") if v)
        elif isinstance(current_value, Path):
            coerced = Path(value)
        else:
            coerced = value
        setattr(target, attr, coerced)


def slugify(index: int, overrides: Dict[str, object]) -> str:
    parts = [f"exp{index:03d}"]
    for key in sorted(overrides):
        value = str(overrides[key]).replace(".", "p").replace("/", "-").replace(" ", "")
        parts.append(f"{key.replace('.', '-')}-{value}")
    name = "_".join(parts)
    return name[:180]


def read_eval_summary(metrics_path: Path) -> Dict[str, object]:
    if not metrics_path.exists():
        return {}
    best_eval = None
    last_eval = None
    last_train = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = record.get("phase")
            if phase == "eval":
                last_eval = record
                if best_eval is None or record.get("eval_objective", -math.inf) > best_eval.get("eval_objective", -math.inf):
                    best_eval = record
            elif phase == "train":
                last_train = record
    summary: Dict[str, object] = {}
    if last_train:
        summary["last_train"] = last_train
    if last_eval:
        summary["last_eval"] = last_eval
    if best_eval:
        summary["best_eval"] = best_eval
    return summary


def run_experiments(args: argparse.Namespace) -> None:
    grid = load_grid(args.grid)
    base_data, base_env, base_train, base_eval = load_baselines(args.baseline_config)
    base_data.root = args.data_root
    ensure_dir(args.save_root)
    trials = list(iter_trials(grid, seed=args.seed, shuffle=args.shuffle))
    if args.max_trials is not None:
        trials = trials[: args.max_trials]
    if not trials:
        print("No experiments resolved from the provided grid.")
        return
    if args.dry_run:
        print(f"Prepared {len(trials)} experiments (dry run):")
        for idx, overrides in enumerate(trials, start=1):
            print(f"{idx:03d}: {slugify(idx, overrides)}")
        return
    log_path = args.save_root / "experiment_log.jsonl"
    for idx, overrides in enumerate(trials, start=1):
        run_seed = overrides.get("train.seed", args.seed)
        start = time.time()
        data_cfg = replace(base_data)
        env_cfg = replace(base_env)
        train_cfg = replace(base_train)
        eval_cfg = replace(base_eval)
        train_cfg.seed = run_seed
        train_cfg.eval_interval = args.eval_interval
        apply_overrides(data_cfg, env_cfg, train_cfg, eval_cfg, overrides)
        slug = slugify(idx, overrides)
        experiment_dir = ensure_dir(args.save_root / slug)
        if any(experiment_dir.iterdir()):
            print(f"[{idx}/{len(trials)}] Skipping {slug} (existing outputs)")
            continue
        train_cfg.save_dir = experiment_dir
        print(f"[{idx}/{len(trials)}] Running {slug}")
        trainer = DifferentiableMarketTrainer(data_cfg, env_cfg, train_cfg, eval_cfg)
        trainer.fit()
        duration = time.time() - start
        summary = read_eval_summary(trainer.metrics_path)
        payload = {
            "index": idx,
            "name": slug,
            "overrides": overrides,
            "run_dir": str(trainer.run_dir),
            "metrics_path": str(trainer.metrics_path),
            "duration_sec": duration,
            "seed": run_seed,
            "notes": args.notes,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        payload.update(summary)
        with log_path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
        print(f"[{idx}/{len(trials)}] Completed {slug} in {duration/60:.2f} minutes")


def main() -> None:
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
