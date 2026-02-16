#!/usr/bin/env python3
"""Comprehensive SUI hyperparameter sweep.

Sweeps: seed, return_weight, lr, epochs, seq_length, architecture.
Designed to run on 5090 (32GB VRAM) or local GPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent


def _python() -> str:
    venv = REPO_ROOT / ".venv313" / "bin" / "python3"
    return str(venv) if venv.exists() else sys.executable


@dataclass
class SweepConfig:
    symbol: str = "SUIUSDT"
    return_weight: float = 0.012
    epochs: int = 25
    sequence_length: int = 72
    learning_rate: float = 1e-4
    seed: int = 1337
    horizons: str = "1,4,24"
    batch_size: int = 16
    transformer_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    model_arch: str = "classic"
    maker_fee: float = 0.001
    weight_decay: float = 1e-4
    no_compile: bool = True


def run_one(cfg: SweepConfig, run_name: str) -> dict:
    if cfg.symbol == "SUIUSDT":
        cache_root = "binancechronossolexperiment/forecast_cache_sui_10bp"
    else:
        cache_root = "binanceneural/forecast_cache"

    cmd = [
        _python(), "-m", "binancechronossolexperiment.run_experiment",
        "--symbol", cfg.symbol,
        "--return-weight", str(cfg.return_weight),
        "--epochs", str(cfg.epochs),
        "--sequence-length", str(cfg.sequence_length),
        "--learning-rate", str(cfg.learning_rate),
        "--horizons", cfg.horizons,
        "--batch-size", str(cfg.batch_size),
        "--seed", str(cfg.seed),
        "--transformer-dim", str(cfg.transformer_dim),
        "--transformer-layers", str(cfg.transformer_layers),
        "--transformer-heads", str(cfg.transformer_heads),
        "--maker-fee", str(cfg.maker_fee),
        "--forecast-cache-root", cache_root,
        "--cache-only",
        "--no-plot",
        "--run-name", run_name,
    ]
    if cfg.no_compile:
        cmd.append("--no-compile")

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"  rw={cfg.return_weight} lr={cfg.learning_rate} ep={cfg.epochs} "
          f"seq={cfg.sequence_length} seed={cfg.seed} "
          f"dim={cfg.transformer_dim} L={cfg.transformer_layers} H={cfg.transformer_heads}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        env = dict(__import__("os").environ)
        # Override seed via env since run_experiment uses TrainingConfig default
        # We'll patch the training config seed through a temp config approach
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200, cwd=str(REPO_ROOT),
            env=env,
        )

        metrics_path = EXP_DIR / "results" / run_name / "simulation_metrics.json"
        elapsed = time.time() - t0
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            test_m = metrics.get("metrics", {}).get("test", {})
            out = {
                "name": run_name,
                "config": asdict(cfg),
                "test_return": test_m.get("total_return", 0),
                "test_sortino": test_m.get("sortino", 0),
                "test_max_dd": test_m.get("max_drawdown", 0),
                "test_trades": test_m.get("num_trades", 0),
                "final_equity": test_m.get("final_equity", 10000),
                "elapsed_s": elapsed,
            }
            print(f"  -> sortino={out['test_sortino']:.1f} return={out['test_return']:.4f} "
                  f"dd={out['test_max_dd']:.4f} trades={out['test_trades']} ({elapsed:.0f}s)")
            return out
        else:
            print(f"  -> NO METRICS FILE ({elapsed:.0f}s)")
            stderr_tail = result.stderr[-1000:] if result.stderr else ""
            return {"name": run_name, "error": "no metrics", "stderr": stderr_tail, "elapsed_s": elapsed}
    except Exception as e:
        return {"name": run_name, "error": str(e), "elapsed_s": time.time() - t0}


def build_sweep_configs(args) -> list[tuple[SweepConfig, str]]:
    configs = []
    base = SweepConfig(symbol=args.symbol)

    if args.sweep_type in ("all", "seed"):
        for seed in [1337, 42, 123, 2024, 7, 3407, 9999]:
            c = SweepConfig(symbol=args.symbol, seed=seed)
            configs.append((c, f"sweep_seed{seed}"))

    if args.sweep_type in ("all", "rw"):
        for rw in [0.005, 0.008, 0.010, 0.012, 0.014, 0.016, 0.020, 0.030]:
            c = SweepConfig(symbol=args.symbol, return_weight=rw)
            configs.append((c, f"sweep_rw{str(rw).replace('.', '')}"))

    if args.sweep_type in ("all", "lr"):
        for lr in [3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4]:
            c = SweepConfig(symbol=args.symbol, learning_rate=lr)
            configs.append((c, f"sweep_lr{lr:.0e}".replace("+", "")))

    if args.sweep_type in ("all", "epochs"):
        for ep in [15, 20, 25, 30, 40, 50]:
            c = SweepConfig(symbol=args.symbol, epochs=ep)
            configs.append((c, f"sweep_ep{ep}"))

    if args.sweep_type in ("all", "seq"):
        for sl in [32, 48, 72, 96, 128]:
            c = SweepConfig(symbol=args.symbol, sequence_length=sl)
            configs.append((c, f"sweep_seq{sl}"))

    if args.sweep_type in ("all", "arch"):
        arch_configs = [
            (128, 3, 4),
            (128, 4, 4),
            (256, 3, 8),
            (256, 4, 8),   # current
            (256, 6, 8),
            (384, 4, 8),
            (384, 6, 8),
            (512, 4, 8),
        ]
        for dim, layers, heads in arch_configs:
            c = SweepConfig(
                symbol=args.symbol,
                transformer_dim=dim,
                transformer_layers=layers,
                transformer_heads=heads,
            )
            configs.append((c, f"sweep_d{dim}_L{layers}_H{heads}"))

    if args.sweep_type == "fine":
        # Fine grid around current best (rw=0.012, lr=1e-4, ep25)
        for rw in [0.010, 0.011, 0.012, 0.013, 0.014]:
            for lr in [7e-5, 1e-4, 1.5e-4]:
                for ep in [20, 25, 30]:
                    c = SweepConfig(
                        symbol=args.symbol,
                        return_weight=rw,
                        learning_rate=lr,
                        epochs=ep,
                    )
                    name = f"fine_rw{str(rw).replace('.', '')}_lr{lr:.0e}_ep{ep}".replace("+", "")
                    configs.append((c, name))

    if args.sweep_type == "best_seeds":
        # Take best config and try all seeds
        for seed in [1337, 42, 123, 2024, 7, 3407, 9999, 314, 777, 1234]:
            c = SweepConfig(symbol=args.symbol, seed=seed)
            configs.append((c, f"bestseed_s{seed}"))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SUIUSDT")
    parser.add_argument("--sweep-type", default="all",
                        choices=["all", "seed", "rw", "lr", "epochs", "seq", "arch", "fine", "best_seeds"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip already-completed runs")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else EXP_DIR / f"sweep_{args.sweep_type}_{args.symbol.lower()}_{timestamp}.json"

    configs = build_sweep_configs(args)
    print(f"Sweep: {args.sweep_type} on {args.symbol}, {len(configs)} configs")

    completed = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for r in json.load(f):
                if "error" not in r:
                    completed[r["name"]] = r
        print(f"Resuming: {len(completed)} already done")

    results = list(completed.values())
    for cfg, name in configs:
        if name in completed:
            print(f"Skipping {name} (already done)")
            continue
        result = run_one(cfg, name)
        results.append(result)
        # Save after each run for crash resilience
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda x: x.get("test_sortino", 0), reverse=True)

    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE: {args.sweep_type} on {args.symbol}")
    print(f"{'='*70}")
    print(f"{'Name':40s} {'Sortino':>8s} {'Return':>10s} {'MaxDD':>8s} {'Trades':>7s}")
    print("-" * 75)
    for r in valid[:20]:
        print(f"{r['name']:40s} {r['test_sortino']:8.1f} {r['test_return']:10.4f} "
              f"{r.get('test_max_dd', 0):8.4f} {r['test_trades']:7d}")

    if valid:
        best = valid[0]
        print(f"\nBest: {best['name']} (Sortino {best['test_sortino']:.1f}, Return {best['test_return']:.4f})")


if __name__ == "__main__":
    main()
