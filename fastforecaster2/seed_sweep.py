from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import ContextManager, Optional

from wandboard import WandBoardLogger

from .config import FastForecaster2Config
from .run_training import build_config
from .trainer import FastForecaster2Trainer


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        value = int(token)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("Expected at least one seed in --seeds.")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastForecaster2 seed sweeps for reproducible MAE ranking.")

    parser.add_argument("--dataset", choices=["hourly", "daily", "custom"], default="hourly")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("fastforecaster2") / "seed_sweeps")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--max-symbols", type=int, default=24)
    parser.add_argument("--seeds", type=str, default="1337,1701,2026,4242")

    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--min-rows-per-symbol", type=int, default=1024)
    parser.add_argument("--max-train-windows-per-symbol", type=int, default=80000)
    parser.add_argument("--max-eval-windows-per-symbol", type=int, default=10000)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--return-loss-weight", type=float, default=0.20)
    parser.add_argument("--direction-loss-weight", type=float, default=0.02)
    parser.add_argument("--direction-margin-scale", type=float, default=16.0)
    parser.add_argument("--horizon-weight-power", type=float, default=0.35)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-eval", dest="use_ema_eval", action="store_true", default=True)
    parser.add_argument("--no-ema-eval", dest="use_ema_eval", action="store_false")

    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--qk-norm", dest="qk_norm", action="store_true", default=True)
    parser.add_argument("--no-qk-norm", dest="qk_norm", action="store_false")
    parser.add_argument("--qk-norm-eps", type=float, default=1e-6)

    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--torch-compile", dest="torch_compile", action="store_true", default=True)
    parser.add_argument("--no-torch-compile", dest="torch_compile", action="store_false")
    parser.add_argument("--compile-mode", type=str, default="max-autotune")
    parser.add_argument("--fused-optim", dest="use_fused_optimizer", action="store_true", default=True)
    parser.add_argument("--no-fused-optim", dest="use_fused_optimizer", action="store_false")
    parser.add_argument("--use-cpp-kernels", action="store_true", default=False)
    parser.add_argument("--build-cpp-extension", action="store_true", default=False)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")

    return parser.parse_args()


def _logger_context(
    *,
    cfg: FastForecaster2Config,
    run_name: str,
) -> ContextManager[Optional[WandBoardLogger]]:
    if not cfg.wandb_project:
        return nullcontext()
    return WandBoardLogger(
        run_name=run_name,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        tags=cfg.wandb_tags,
        log_dir="tensorboard_logs",
        tensorboard_subdir=f"fastforecaster2/seed_sweeps/{run_name}",
        enable_wandb=True,
        log_metrics=True,
        config={"fastforecaster2": cfg.as_dict()},
    )


def _seed_run_name(prefix: str, seed: int) -> str:
    return f"{prefix}_seed{seed}"


def _seed_config(base_cfg: FastForecaster2Config, *, sweep_dir: Path, seed: int) -> FastForecaster2Config:
    # Use dataclass replace so __post_init__ recomputes derived output paths.
    return replace(base_cfg, seed=seed, output_dir=(sweep_dir / f"seed_{seed}"))


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.output_dir / f"seed_sweep_{run_stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Build a baseline config using the same parser contract as run_training.py.
    args.output_dir = sweep_dir
    args.seed = seeds[0]
    base_cfg = build_config(args)

    run_name_prefix = base_cfg.wandb_run_name or f"fastforecaster2_seed_sweep_{run_stamp}"

    print(f"[fastforecaster2-seed-sweep] Running {len(seeds)} seeds: {', '.join(str(s) for s in seeds)}")
    results: list[dict[str, object]] = []

    for index, seed in enumerate(seeds, start=1):
        cfg = _seed_config(base_cfg, sweep_dir=sweep_dir, seed=seed)
        run_name = _seed_run_name(run_name_prefix, seed)
        print(f"[fastforecaster2-seed-sweep] Seed {index}/{len(seeds)}: {seed}")

        try:
            with _logger_context(cfg=cfg, run_name=run_name) as metrics_logger:
                trainer = FastForecaster2Trainer(cfg, metrics_logger=metrics_logger)
                summary = trainer.train()

            row = {
                "seed": seed,
                "status": "ok",
                "config": cfg.as_dict(),
                "summary": summary,
            }
            results.append(row)
            print(
                "[fastforecaster2-seed-sweep] Completed "
                f"seed={seed} val_mae={summary['best_val_mae']:.6f} test_mae={summary['test_mae']:.6f}"
            )
        except Exception as exc:
            row = {
                "seed": seed,
                "status": "failed",
                "config": cfg.as_dict(),
                "error": str(exc),
            }
            results.append(row)
            print(f"[fastforecaster2-seed-sweep] Seed {seed} failed: {exc}")

    ok_results = [row for row in results if row.get("status") == "ok"]
    ok_results.sort(key=lambda row: float(row["summary"]["best_val_mae"]))

    payload = {
        "run_timestamp_utc": run_stamp,
        "seed_count": len(seeds),
        "successful_runs": len(ok_results),
        "base_config": base_cfg.as_dict(),
        "results": results,
        "ranked_by_best_val_mae": ok_results,
    }
    out_path = sweep_dir / "seed_sweep_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if ok_results:
        best = ok_results[0]
        (sweep_dir / "best_seed.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(
            "[fastforecaster2-seed-sweep] Best seed: "
            f"{best['seed']} val_mae={best['summary']['best_val_mae']:.6f} "
            f"test_mae={best['summary']['test_mae']:.6f}"
        )

    print(f"[fastforecaster2-seed-sweep] Results written to {out_path}")


if __name__ == "__main__":
    main()
