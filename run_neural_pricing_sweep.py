#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from neuralpricingstrategy.data import build_pricing_dataset, load_backtest_frames, split_dataset_by_date
from neuralpricingstrategy.trainer import PricingTrainingConfig, train_pricing_model
from src.fixtures import all_crypto_symbols


@dataclass
class SweepRunConfig:
    clamp_pct: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    pnl_weight: float
    hidden_dim: int
    depth: int
    dropout: float
    seed: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiple neural pricing models and evaluate them in the market simulator."
    )
    parser.add_argument(
        "--backtest-csv",
        action="append",
        default=["backtest_results/*_backtest.csv"],
        help="Glob(s) pointing at *_backtest.csv exports.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to include for training and simulation (default: all crypto pairs).",
    )
    parser.add_argument(
        "--output-dir",
        default="neuralpricingstrategy/reports",
        help="Directory to store training artifacts and sweep summaries.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of model variants to train.")
    parser.add_argument("--epochs", type=int, default=220, help="Training epochs per run.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction for splits.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--eval-steps", type=int, default=30, help="Simulation steps (days).")
    parser.add_argument("--eval-step-size", type=int, default=1, help="Simulation step size.")
    parser.add_argument("--top-k", type=int, default=4, help="Top K positions to hold each step.")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Initial cash for simulation.")
    parser.add_argument("--max-hold-days", type=float, default=3.0, help="Max holding period in days.")
    parser.add_argument("--fast-sim", action="store_true", help="Enable fast simulator mode.")
    parser.add_argument("--kronos-only", action="store_true", help="Force Kronos-only forecasting.")
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Skip the market-simulator evaluation step (train models only).",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_sweep_configs(runs: int, base_seed: int, epochs: int) -> List[SweepRunConfig]:
    hidden_dims = [128, 192, 256]
    depths = [2, 3, 4]
    dropouts = [0.0, 0.1, 0.2]
    learning_rates = [1e-3, 5e-4, 3e-4, 1e-4]
    weight_decays = [1e-4, 3e-4, 1e-3]
    pnl_weights = [0.1, 0.2, 0.3]
    batch_sizes = [256, 512, 768]
    clamp_pcts = [0.05, 0.08, 0.1]

    grid = list(
        itertools.product(
            hidden_dims,
            depths,
            dropouts,
            learning_rates,
            weight_decays,
            pnl_weights,
            batch_sizes,
            clamp_pcts,
        )
    )
    rng = random.Random(base_seed)
    rng.shuffle(grid)
    if runs > len(grid):
        raise ValueError(f"Requested {runs} runs but only {len(grid)} unique configs available.")

    configs: List[SweepRunConfig] = []
    for idx, entry in enumerate(grid[:runs], start=1):
        hidden_dim, depth, dropout, lr, wd, pnl_weight, batch_size, clamp_pct = entry
        configs.append(
            SweepRunConfig(
                clamp_pct=float(clamp_pct),
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(lr),
                weight_decay=float(wd),
                pnl_weight=float(pnl_weight),
                hidden_dim=int(hidden_dim),
                depth=int(depth),
                dropout=float(dropout),
                seed=base_seed + idx,
            )
        )
    return configs


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_max_hold_map(symbols: Sequence[str], max_hold_days: float) -> str:
    seconds = max(1.0, max_hold_days) * 24.0 * 3600.0
    return ",".join(f"{symbol}:{seconds:.0f}" for symbol in symbols)


def _run_simulation(
    *,
    run_dir: Path,
    symbols: Sequence[str],
    steps: int,
    step_size: int,
    top_k: int,
    initial_cash: float,
    output_dir: Path,
    max_hold_days: float,
    fast_sim: bool,
    kronos_only: bool,
) -> Optional[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{run_dir.name}_metrics.json"
    trades_path = output_dir / f"{run_dir.name}_trades.csv"
    summary_path = output_dir / f"{run_dir.name}_trades_summary.json"
    log_path = output_dir / f"{run_dir.name}_sim.log"

    env = os.environ.copy()
    env["MARKETSIM_ENABLE_NEURAL_PRICING"] = "1"
    env["MARKETSIM_NEURAL_PRICING_RUN_DIR"] = str(run_dir)
    env["MARKETSIM_USE_MOCK_ANALYTICS"] = "0"
    env["MARKETSIM_SYMBOL_MAX_HOLD_SECONDS_MAP"] = _build_max_hold_map(symbols, max_hold_days)

    cmd = [
        sys.executable,
        "-m",
        "marketsimulator.run_trade_loop",
        "--symbols",
        *symbols,
        "--steps",
        str(max(1, steps)),
        "--step-size",
        str(max(1, step_size)),
        "--top-k",
        str(max(1, top_k)),
        "--initial-cash",
        str(initial_cash),
        "--flatten-end",
        "--metrics-json",
        str(metrics_path),
        "--trades-csv",
        str(trades_path),
        "--trades-summary-json",
        str(summary_path),
        "--real-analytics",
    ]
    if fast_sim:
        cmd.append("--fast-sim")
    if kronos_only:
        cmd.append("--kronos-only")

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n")
        handle.flush()
        result = subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)

    if result.returncode != 0 or not metrics_path.exists():
        return None

    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    args = _parse_args()
    symbols = [s.upper() for s in (args.symbols or all_crypto_symbols)]

    sweep_root = Path(args.output_dir) / f"neural_pricing_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    frames = load_backtest_frames(args.backtest_csv, symbol_filter=symbols)
    configs = _generate_sweep_configs(args.runs, args.seed, args.epochs)

    _save_json(sweep_root / "sweep_config.json", {"runs": [asdict(cfg) for cfg in configs]})

    summary_rows: List[Dict[str, object]] = []

    configs_by_clamp: Dict[float, List[SweepRunConfig]] = {}
    for cfg in configs:
        configs_by_clamp.setdefault(cfg.clamp_pct, []).append(cfg)

    for clamp_pct, clamp_configs in configs_by_clamp.items():
        dataset = build_pricing_dataset(frames, clamp_pct=clamp_pct)
        train_ds, val_ds = split_dataset_by_date(dataset, validation_fraction=args.val_fraction)

        for idx, cfg in enumerate(clamp_configs, start=1):
            run_name = f"run_{clamp_pct:.2f}_{idx:02d}_seed{cfg.seed}"
            run_dir = sweep_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            _set_seed(cfg.seed)
            train_config = PricingTrainingConfig(
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                pnl_weight=cfg.pnl_weight,
                hidden_dim=cfg.hidden_dim,
                depth=cfg.depth,
                dropout=cfg.dropout,
            )

            result = train_pricing_model(train_ds, validation_dataset=val_ds, config=train_config)

            torch.save(result.model.state_dict(), run_dir / "pricing_model.pt")
            _save_json(run_dir / "feature_spec.json", dataset.feature_spec.to_dict())
            _save_json(
                run_dir / "run_config.json",
                {
                    "clamp_pct": clamp_pct,
                    "hidden_dim": cfg.hidden_dim,
                    "depth": cfg.depth,
                    "dropout": cfg.dropout,
                    "epochs": cfg.epochs,
                    "batch_size": cfg.batch_size,
                    "learning_rate": cfg.learning_rate,
                    "weight_decay": cfg.weight_decay,
                    "pnl_weight": cfg.pnl_weight,
                    "seed": cfg.seed,
                },
            )
            _save_json(run_dir / "training_history.json", [entry.__dict__ for entry in result.history])
            _save_json(run_dir / "training_metrics.json", result.final_metrics)

            sim_metrics = None
            if not args.skip_sim:
                sim_metrics = _run_simulation(
                    run_dir=run_dir,
                    symbols=symbols,
                    steps=args.eval_steps,
                    step_size=args.eval_step_size,
                    top_k=args.top_k,
                    initial_cash=args.initial_cash,
                    output_dir=sweep_root / "sim_outputs",
                    max_hold_days=args.max_hold_days,
                    fast_sim=args.fast_sim,
                    kronos_only=args.kronos_only,
                )

            row: Dict[str, object] = {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "clamp_pct": clamp_pct,
                "hidden_dim": cfg.hidden_dim,
                "depth": cfg.depth,
                "dropout": cfg.dropout,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "pnl_weight": cfg.pnl_weight,
                "seed": cfg.seed,
            }
            row.update({f"train_{k}": v for k, v in result.final_metrics.items()})
            if sim_metrics:
                row.update({f"sim_{k}": v for k, v in sim_metrics.items()})
            summary_rows.append(row)

    summary_path = sweep_root / "sweep_summary.json"
    _save_json(summary_path, {"rows": summary_rows})

    try:
        import pandas as pd

        df = pd.DataFrame(summary_rows)
        df.to_csv(sweep_root / "sweep_summary.csv", index=False)
    except Exception:
        pass

    best_by_sortino = None
    best_by_pnl = None
    for row in summary_rows:
        sortino = row.get("sim_sortino")
        pnl = row.get("sim_pnl")
        if sortino is not None:
            if best_by_sortino is None or float(sortino) > float(best_by_sortino.get("sim_sortino", -1e9)):
                best_by_sortino = row
        if pnl is not None:
            if best_by_pnl is None or float(pnl) > float(best_by_pnl.get("sim_pnl", -1e12)):
                best_by_pnl = row

    _save_json(
        sweep_root / "best_models.json",
        {
            "best_by_sortino": best_by_sortino,
            "best_by_pnl": best_by_pnl,
        },
    )

    if best_by_sortino:
        print(f"Best Sortino: {best_by_sortino.get('run_name')} ({best_by_sortino.get('sim_sortino')})")
    if best_by_pnl:
        print(f"Best PnL: {best_by_pnl.get('run_name')} ({best_by_pnl.get('sim_pnl')})")


if __name__ == "__main__":
    main()
