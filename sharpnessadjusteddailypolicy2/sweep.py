from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("TORCH_NO_COMPILE", "1")

from neuraldailytraining import DailyDataModule, NeuralDailyTrainer

from .config import DEFAULT_EXPERIMENT_ORDER, RESULTS_ROOT, build_experiment, experiment_names


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]



def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")



def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _print_leaderboard(results: list[dict[str, Any]]) -> None:
    ok = [row for row in results if row.get("status") == "ok"]
    ok.sort(key=lambda row: row.get("best_val_score", float("-inf")), reverse=True)
    if not ok:
        print("No successful runs yet.")
        return
    print("\nLeaderboard:")
    print(f"{'Experiment':28} {'Seed':>5} {'Score':>10} {'Sortino':>10} {'Return':>10} {'Secs':>8}")
    for row in ok:
        print(
            f"{row['experiment']:28} {row['seed']:5d} "
            f"{row['best_val_score']:10.4f} {row['best_val_sortino']:10.4f} "
            f"{row['best_val_return']:10.4f} {row['duration_seconds']:8.1f}"
        )



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short-run daily training sweep for stability comparisons.")
    parser.add_argument(
        "--experiments",
        default=",".join(DEFAULT_EXPERIMENT_ORDER),
        help=f"Comma-separated experiment names. Available: {', '.join(experiment_names())}",
    )
    parser.add_argument("--seeds", default="1337", help="Comma-separated integer seeds.")
    parser.add_argument("--output", help="Path to JSON output file.")
    parser.add_argument("--device", help="Torch device override.")
    parser.add_argument("--data-root", help="Daily CSV root, defaults to trainingdata/train.")
    parser.add_argument("--forecast-cache", help="Forecast cache root.")
    parser.add_argument("--symbols", nargs="*", help="Optional explicit symbol subset.")
    parser.add_argument("--batch-size", type=int, help="Override batch size for all runs.")
    parser.add_argument("--sequence-length", type=int, help="Override sequence length for all runs.")
    parser.add_argument("--epochs", type=int, help="Override epochs for all runs.")
    parser.add_argument("--max-train-batches", type=int, help="Cap train dataloader batches per epoch.")
    parser.add_argument("--max-val-batches", type=int, help="Cap val dataloader batches per epoch.")
    parser.add_argument("--num-workers", type=int, help="Override dataloader worker count.")
    parser.add_argument("--use-compile", action=argparse.BooleanOptionalAction, default=None)
    return parser



def main() -> None:
    args = _build_parser().parse_args()
    experiments = _parse_csv(args.experiments)
    if not experiments:
        raise ValueError("No experiments selected.")
    seeds = [int(item) for item in _parse_csv(args.seeds)]
    if not seeds:
        raise ValueError("No seeds selected.")

    dataset_overrides: dict[str, Any] = {}
    if args.data_root:
        dataset_overrides["data_root"] = Path(args.data_root)
    if args.forecast_cache:
        dataset_overrides["forecast_cache_dir"] = Path(args.forecast_cache)
    if args.symbols:
        dataset_overrides["symbols"] = tuple(sym.upper() for sym in args.symbols)

    training_overrides: dict[str, Any] = {}
    for key in ("batch_size", "sequence_length", "epochs", "max_train_batches", "max_val_batches", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            training_overrides[key] = value
    if args.use_compile is not None:
        training_overrides["use_compile"] = args.use_compile

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else (RESULTS_ROOT / f"daily_sweep_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    print(f"Running {len(experiments)} experiments across {len(seeds)} seed(s). Output: {output_path}")
    for experiment_name in experiments:
        for seed in seeds:
            spec, dataset_config, training_config = build_experiment(
                experiment_name,
                seed=seed,
                device=args.device,
                dataset_overrides=dataset_overrides,
                training_overrides=training_overrides,
            )
            run_stamp = time.strftime("%Y%m%d_%H%M%S")
            run_name = f"sadp2_{experiment_name}_s{seed}_{run_stamp}"
            training_config = replace(training_config, run_name=run_name)
            print(f"\n=== {experiment_name} seed={seed} run={run_name} ===")
            _set_all_seeds(training_config.seed)
            try:
                module = DailyDataModule(dataset_config)
                trainer = NeuralDailyTrainer(training_config, module)
                artifacts = trainer.train()
                summary = artifacts.summary
                if summary is None:
                    raise RuntimeError("Trainer did not return a summary.")
                result = {
                    "status": "ok",
                    "experiment": experiment_name,
                    "description": spec.description,
                    "seed": seed,
                    "run_name": run_name,
                    "duration_seconds": summary.duration_seconds,
                    "best_val_score": summary.best_val_score,
                    "best_val_sortino": summary.best_val_sortino,
                    "best_val_return": summary.best_val_return,
                    "best_binary_sortino": summary.best_binary_sortino,
                    "best_binary_return": summary.best_binary_return,
                    "final_train_score": summary.final_train_score,
                    "final_train_return": summary.final_train_return,
                    "num_symbols": summary.num_symbols,
                    "train_samples": summary.train_samples,
                    "val_samples": summary.val_samples,
                    "best_checkpoint": summary.best_checkpoint,
                    "checkpoint_dir": summary.checkpoint_dir,
                    "history": summary.history,
                    "training_config": asdict(training_config),
                    "dataset_config": asdict(dataset_config),
                }
            except Exception as exc:  # noqa: BLE001
                result = {
                    "status": "error",
                    "experiment": experiment_name,
                    "description": spec.description,
                    "seed": seed,
                    "run_name": run_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "training_config": asdict(training_config),
                    "dataset_config": asdict(dataset_config),
                }
                print(f"Run failed: {exc}")
            results.append(result)
            output_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
            _print_leaderboard(results)

    print(f"\nSaved {len(results)} result records to {output_path}")


if __name__ == "__main__":
    main()
