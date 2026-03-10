from __future__ import annotations

import argparse
import itertools
import json
import random
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import ContextManager, Optional

from wandboard import WandBoardLogger

from .config import FastForecasterConfig
from .run_training import _default_data_dir, _parse_symbols, _parse_tags
from .trainer import FastForecasterTrainer


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Expected at least one float value.")
    return tuple(values)


def _parse_bool_list(raw: str) -> tuple[bool, ...]:
    values: list[bool] = []
    mapping = {
        "1": True,
        "true": True,
        "yes": True,
        "on": True,
        "0": False,
        "false": False,
        "no": False,
        "off": False,
    }
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token not in mapping:
            raise ValueError(f"Unsupported bool token '{item}'. Use on/off or true/false.")
        values.append(mapping[token])
    if not values:
        raise ValueError("Expected at least one bool value.")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-sweep FastForecaster objective hyperparameters.")
    parser.add_argument("--dataset", choices=["hourly", "daily", "custom"], default="hourly")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("FastForecaster") / "sweeps")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--max-symbols", type=int, default=8)

    parser.add_argument("--lookback", type=int, default=192)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--min-rows-per-symbol", type=int, default=1024)
    parser.add_argument("--max-train-windows-per-symbol", type=int, default=4000)
    parser.add_argument("--max-eval-windows-per-symbol", type=int, default=800)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rates", type=str, default="")
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--weight-decays", type=str, default="")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=4)

    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--dropouts", type=str, default="")

    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="max-autotune")
    parser.add_argument("--no-fused-optim", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--return-loss-weights", type=str, default="0.0,0.1,0.2,0.3")
    parser.add_argument("--direction-loss-weights", type=str, default="0.0,0.01,0.02")
    parser.add_argument("--direction-margin-scales", type=str, default="8,16")
    parser.add_argument("--horizon-weight-powers", type=str, default="0.0,0.2,0.35")
    parser.add_argument("--qk-norm-options", type=str, default="on,off")
    parser.add_argument("--ema-eval-options", type=str, default="on")
    parser.add_argument("--ema-decays", type=str, default="0.999")
    parser.add_argument("--limit-trials", type=int, default=12)

    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")

    return parser.parse_args()


def build_base_config(args: argparse.Namespace, sweep_output: Path) -> FastForecasterConfig:
    data_dir = args.data_dir if args.data_dir is not None else _default_data_dir(args.dataset)
    return FastForecasterConfig(
        data_dir=data_dir,
        output_dir=sweep_output,
        symbols=_parse_symbols(args.symbols),
        max_symbols=args.max_symbols,
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        min_rows_per_symbol=args.min_rows_per_symbol,
        max_train_windows_per_symbol=args.max_train_windows_per_symbol,
        max_eval_windows_per_symbol=args.max_eval_windows_per_symbol,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        grad_accum_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        early_stopping_patience=args.early_stopping_patience,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        qk_norm=True,
        qk_norm_eps=1e-6,
        precision=args.precision,
        torch_compile=not args.no_torch_compile,
        compile_mode=args.compile_mode,
        use_fused_optimizer=not args.no_fused_optim,
        use_cpp_kernels=False,
        build_cpp_extension=False,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        seed=args.seed,
        device=args.device,
        wandb_project=None,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=_parse_tags(args.wandb_tags),
    )


def _trial_name(index: int, config: FastForecasterConfig) -> str:
    qk = "qk1" if config.qk_norm else "qk0"
    ema = "ema1" if config.use_ema_eval else "ema0"
    return (
        f"trial{index:03d}_rlw{config.return_loss_weight:.3f}_"
        f"dlw{config.direction_loss_weight:.3f}_"
        f"dms{config.direction_margin_scale:.1f}_"
        f"lr{config.learning_rate:.5f}_"
        f"wd{config.weight_decay:.4f}_"
        f"do{config.dropout:.3f}_"
        f"hwp{config.horizon_weight_power:.2f}_{qk}_{ema}"
    ).replace(".", "p")


def _trial_config(
    base_cfg: FastForecasterConfig,
    *,
    output_dir: Path,
    return_loss_weight: float,
    direction_loss_weight: float,
    direction_margin_scale: float,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    horizon_weight_power: float,
    qk_norm: bool,
    use_ema_eval: bool,
    ema_decay: float,
) -> FastForecasterConfig:
    # Use dataclass replace so __post_init__ recomputes derived output paths.
    return replace(
        base_cfg,
        output_dir=output_dir,
        return_loss_weight=float(return_loss_weight),
        direction_loss_weight=float(direction_loss_weight),
        direction_margin_scale=float(direction_margin_scale),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        dropout=float(dropout),
        horizon_weight_power=float(horizon_weight_power),
        qk_norm=bool(qk_norm),
        use_ema_eval=bool(use_ema_eval),
        ema_decay=float(ema_decay),
    )


def _logger_context(
    *,
    args: argparse.Namespace,
    run_name: str,
    cfg: FastForecasterConfig,
) -> ContextManager[Optional[WandBoardLogger]]:
    if not args.wandb_project:
        return nullcontext()
    return WandBoardLogger(
        run_name=run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=cfg.wandb_tags,
        log_dir="tensorboard_logs",
        tensorboard_subdir=f"fastforecaster/sweeps/{run_name}",
        enable_wandb=True,
        log_metrics=True,
        config={"fastforecaster": cfg.as_dict()},
    )


def main() -> None:
    args = parse_args()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_output = args.output_dir / f"sweep_{run_stamp}"
    sweep_output.mkdir(parents=True, exist_ok=True)

    base_cfg = build_base_config(args, sweep_output)

    return_loss_weights = _parse_float_list(args.return_loss_weights)
    direction_loss_weights = _parse_float_list(args.direction_loss_weights)
    direction_margin_scales = _parse_float_list(args.direction_margin_scales)
    horizon_weight_powers = _parse_float_list(args.horizon_weight_powers)
    qk_options = _parse_bool_list(args.qk_norm_options)
    ema_options = _parse_bool_list(args.ema_eval_options)
    ema_decays = _parse_float_list(args.ema_decays)
    learning_rates = _parse_float_list(args.learning_rates) if args.learning_rates.strip() else (args.learning_rate,)
    weight_decays = _parse_float_list(args.weight_decays) if args.weight_decays.strip() else (args.weight_decay,)
    dropouts = _parse_float_list(args.dropouts) if args.dropouts.strip() else (args.dropout,)

    combos = list(
        itertools.product(
            return_loss_weights,
            direction_loss_weights,
            direction_margin_scales,
            learning_rates,
            weight_decays,
            dropouts,
            horizon_weight_powers,
            qk_options,
            ema_options,
            ema_decays,
        )
    )
    sampler = random.Random(args.seed)
    sampler.shuffle(combos)
    combos = combos[: max(1, args.limit_trials)]

    results: list[dict[str, object]] = []

    print(f"[fastforecaster-sweep] Running {len(combos)} trials")

    for trial_idx, (rlw, dlw, dms, lr, wd, dropout, hwp, qk, ema_eval, ema_decay) in enumerate(combos, start=1):
        trial_cfg = _trial_config(
            base_cfg,
            output_dir=(sweep_output / f"trial_{trial_idx:03d}"),
            return_loss_weight=float(rlw),
            direction_loss_weight=float(dlw),
            direction_margin_scale=float(dms),
            learning_rate=float(lr),
            weight_decay=float(wd),
            dropout=float(dropout),
            horizon_weight_power=float(hwp),
            qk_norm=bool(qk),
            use_ema_eval=bool(ema_eval),
            ema_decay=float(ema_decay),
        )
        name = _trial_name(trial_idx, trial_cfg)
        trial_cfg = replace(trial_cfg, output_dir=(sweep_output / name))

        print(f"[fastforecaster-sweep] Trial {trial_idx}/{len(combos)}: {name}")

        try:
            with _logger_context(args=args, run_name=name, cfg=trial_cfg) as metrics_logger:
                trainer = FastForecasterTrainer(trial_cfg, metrics_logger=metrics_logger)
                summary = trainer.train()

            result = {
                "trial_index": trial_idx,
                "trial_name": name,
                "status": "ok",
                "config": trial_cfg.as_dict(),
                "summary": summary,
            }
            results.append(result)

            print(
                "[fastforecaster-sweep] Completed "
                f"val_mae={summary['best_val_mae']:.6f} test_mae={summary['test_mae']:.6f}"
            )
        except Exception as exc:
            result = {
                "trial_index": trial_idx,
                "trial_name": name,
                "status": "failed",
                "config": trial_cfg.as_dict(),
                "error": str(exc),
            }
            results.append(result)
            print(f"[fastforecaster-sweep] Trial failed: {exc}")

    ok_results = [row for row in results if row.get("status") == "ok"]
    ok_results.sort(key=lambda row: float(row["summary"]["best_val_mae"]))

    payload = {
        "run_timestamp_utc": run_stamp,
        "base_config": base_cfg.as_dict(),
        "total_trials": len(combos),
        "successful_trials": len(ok_results),
        "results": results,
        "ranked_by_best_val_mae": ok_results,
    }

    results_path = sweep_output / "sweep_results.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if ok_results:
        best = ok_results[0]
        (sweep_output / "best_trial.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(
            "[fastforecaster-sweep] Best trial: "
            f"{best['trial_name']} val_mae={best['summary']['best_val_mae']:.6f} "
            f"test_mae={best['summary']['test_mae']:.6f}"
        )

    print(f"[fastforecaster-sweep] Results written to {results_path}")


if __name__ == "__main__":
    main()
