#!/usr/bin/env python3
"""Run repeated hourly trainer experiments across seeds and summarize market-sim robustness."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_csv_tokens(raw: str, *, cast=str) -> list[Any]:
    values: list[Any] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(cast(token))
    return values


def sanitize_token(value: Any) -> str:
    text = str(value).strip()
    return text.replace("-", "m").replace(".", "p").replace(",", "_")


def combo_label(combo: dict[str, Any]) -> str:
    return "_".join(f"{key}_{sanitize_token(value)}" for key, value in combo.items())


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "sem": float("nan"),
            "ci95_halfwidth": float("nan"),
        }
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values)) if len(values) > 0 else float("nan")
    return {
        "count": float(len(values)),
        "mean": float(mean),
        "std": float(std),
        "min": float(min(values)),
        "max": float(max(values)),
        "sem": float(sem),
        "ci95_halfwidth": float(1.96 * sem),
    }


def compare_metric(
    *,
    candidate: list[float],
    baseline: list[float],
    higher_is_better: bool,
    abs_tolerance: float,
) -> dict[str, float | bool]:
    cand = summarize(candidate)
    base = summarize(baseline)
    delta = float(cand["mean"] - base["mean"])
    direction_delta = delta if higher_is_better else -delta
    pooled_sem = math.sqrt(float(cand["sem"]) ** 2 + float(base["sem"]) ** 2)
    significance_margin = max(float(abs_tolerance), 1.96 * pooled_sem)
    signal_to_noise = direction_delta / max(significance_margin, 1e-9)
    return {
        "delta_mean": delta,
        "directional_delta": direction_delta,
        "pooled_sem": pooled_sem,
        "significance_margin": significance_margin,
        "signal_to_noise": signal_to_noise,
        "significant": bool(direction_delta > significance_margin),
    }


def robust_market_score(
    *,
    sortino_mean: float,
    sortino_std: float,
    return_mean: float,
    return_std: float,
    drawdown_mean: float,
    drawdown_std: float,
    return_weight: float = 0.10,
    std_penalty: float = 1.0,
    drawdown_penalty: float = 0.25,
) -> float:
    robust_sortino = sortino_mean - std_penalty * sortino_std
    robust_return = return_mean - std_penalty * return_std
    robust_drawdown = abs(drawdown_mean) + std_penalty * abs(drawdown_std)
    return robust_sortino + return_weight * robust_return - drawdown_penalty * robust_drawdown


@dataclass(frozen=True)
class ExperimentCombo:
    learning_rate: float
    weight_decay: float
    return_weight: float
    fill_temperature: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "lr": self.learning_rate,
            "wd": self.weight_decay,
            "rw": self.return_weight,
            "ft": self.fill_temperature,
        }


def build_combos(args: argparse.Namespace) -> list[ExperimentCombo]:
    lrs = parse_csv_tokens(args.learning_rates, cast=float)
    wds = parse_csv_tokens(args.weight_decays, cast=float)
    rws = parse_csv_tokens(args.return_weights, cast=float)
    fts = parse_csv_tokens(args.fill_temperatures, cast=float)
    combos = [
        ExperimentCombo(
            learning_rate=lr,
            weight_decay=wd,
            return_weight=rw,
            fill_temperature=ft,
        )
        for lr, wd, rw, ft in itertools.product(lrs, wds, rws, fts)
    ]
    if not combos:
        raise ValueError("At least one combo is required")
    return combos


def build_run_name(*, prefix: str, combo: ExperimentCombo, seed: int) -> str:
    return f"{prefix}_{combo_label(combo.as_dict())}_seed_{seed}"


def train_command(args: argparse.Namespace, combo: ExperimentCombo, seed: int, run_name: str) -> list[str]:
    cmd = [
        sys.executable,
        "unified_hourly_experiment/train_unified_policy.py",
        "--symbols",
        args.symbols,
        "--crypto-symbols",
        args.crypto_symbols,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--lr",
        str(combo.learning_rate),
        "--weight-decay",
        str(combo.weight_decay),
        "--return-weight",
        str(combo.return_weight),
        "--fill-temperature",
        str(combo.fill_temperature),
        "--forecast-horizons",
        args.forecast_horizons,
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--log-dir",
        str(args.log_dir),
        "--checkpoint-metric",
        args.checkpoint_metric,
        "--checkpoint-gap-penalty",
        str(args.checkpoint_gap_penalty),
        "--top-k-checkpoints",
        str(args.top_k_checkpoints),
        "--wandb-mode",
        args.wandb_mode,
    ]
    if args.preload:
        cmd.extend(["--preload", str(args.preload)])
    if args.dry_train_steps is not None:
        cmd.extend(["--dry-train-steps", str(args.dry_train_steps)])
    if args.no_compile:
        cmd.append("--no-compile")
    if args.no_amp:
        cmd.append("--no-amp")
    if args.wandb_project:
        cmd.extend(["--wandb-project", args.wandb_project])
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])
    if args.wandb_group:
        cmd.extend(["--wandb-group", args.wandb_group])
    if args.wandb_tags:
        cmd.extend(["--wandb-tags", args.wandb_tags])
    if args.wandb_notes:
        cmd.extend(["--wandb-notes", args.wandb_notes])
    return cmd


def backtest_command(args: argparse.Namespace, checkpoint: Path) -> list[str]:
    return [
        sys.executable,
        "unified_hourly_experiment/backtest_portfolio.py",
        "--checkpoint",
        str(checkpoint),
        "--symbols",
        args.backtest_symbols or args.symbols,
        "--data-root",
        str(args.backtest_data_root),
        "--cache-root",
        str(args.backtest_cache_root),
        "--maker-fee",
        str(args.maker_fee),
        "--validation-days",
        str(args.backtest_validation_days),
    ]


def resolve_best_checkpoint(checkpoint_dir: Path) -> Path:
    alias = checkpoint_dir / "best.pt"
    if alias.exists():
        return alias.resolve() if alias.is_symlink() else alias
    progress_path = checkpoint_dir / "training_progress.json"
    if progress_path.exists():
        payload = json.loads(progress_path.read_text())
        best = payload.get("best_checkpoint")
        if best:
            return Path(best)
    raise FileNotFoundError(f"No best checkpoint found in {checkpoint_dir}")


def load_training_progress(checkpoint_dir: Path) -> dict[str, Any]:
    payload = json.loads((checkpoint_dir / "training_progress.json").read_text())
    return payload


def load_backtest_summary(checkpoint_dir: Path) -> dict[str, Any]:
    payload = json.loads((checkpoint_dir / "portfolio_backtest.json").read_text())
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"portfolio_backtest.json empty for {checkpoint_dir}")
        return dict(payload[-1])
    raise TypeError(f"Unexpected backtest payload type for {checkpoint_dir}: {type(payload)!r}")


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO), check=True)


def maybe_run(args: argparse.Namespace, cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    if not args.dry_run:
        run_cmd(cmd)


def summarize_config_runs(
    *,
    combo: ExperimentCombo,
    runs: list[dict[str, Any]],
    baseline_runs: list[dict[str, Any]] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_metrics = [float(run["train"]["checkpoint_metric"]) for run in runs]
    sortinos = [float(run["backtest"]["avg_sortino"]) for run in runs]
    returns = [float(run["backtest"]["avg_return"]) for run in runs]
    drawdowns = [float(run["backtest"]["avg_max_dd"]) for run in runs]

    train_summary = summarize(train_metrics)
    sortino_summary = summarize(sortinos)
    return_summary = summarize(returns)
    drawdown_summary = summarize(drawdowns)
    robust_score = robust_market_score(
        sortino_mean=float(sortino_summary["mean"]),
        sortino_std=float(sortino_summary["std"]),
        return_mean=float(return_summary["mean"]),
        return_std=float(return_summary["std"]),
        drawdown_mean=float(drawdown_summary["mean"]),
        drawdown_std=float(drawdown_summary["std"]),
        return_weight=float(args.market_return_weight),
        std_penalty=float(args.market_std_penalty),
        drawdown_penalty=float(args.market_drawdown_penalty),
    )
    summary: dict[str, Any] = {
        "combo": combo.as_dict(),
        "label": combo_label(combo.as_dict()),
        "n_runs": len(runs),
        "train_checkpoint_metric": train_summary,
        "market_avg_sortino": sortino_summary,
        "market_avg_return": return_summary,
        "market_avg_max_dd": drawdown_summary,
        "market_robust_score": robust_score,
        "runs": runs,
    }
    if baseline_runs:
        baseline_sortinos = [float(run["backtest"]["avg_sortino"]) for run in baseline_runs]
        baseline_returns = [float(run["backtest"]["avg_return"]) for run in baseline_runs]
        baseline_drawdowns = [float(run["backtest"]["avg_max_dd"]) for run in baseline_runs]
        summary["baseline_comparison"] = {
            "sortino": compare_metric(
                candidate=sortinos,
                baseline=baseline_sortinos,
                higher_is_better=True,
                abs_tolerance=float(args.sortino_tolerance),
            ),
            "return": compare_metric(
                candidate=returns,
                baseline=baseline_returns,
                higher_is_better=True,
                abs_tolerance=float(args.return_tolerance_pct),
            ),
            "max_drawdown": compare_metric(
                candidate=drawdowns,
                baseline=baseline_drawdowns,
                higher_is_better=True,
                abs_tolerance=float(args.drawdown_tolerance_pct),
            ),
        }
    return summary


def print_summary(report: dict[str, Any]) -> None:
    print("\n=== Multi-Seed Summary ===", flush=True)
    baseline_label = report.get("baseline_label")
    if baseline_label:
        print(f"Baseline: {baseline_label}", flush=True)
    print(
        f"{'label':<34} {'robust':>8} {'sortino':>9} {'return%':>9} {'maxdd%':>9} {'train':>9}",
        flush=True,
    )
    print("-" * 86, flush=True)
    for row in report["configs"]:
        sortino = row["market_avg_sortino"]["mean"]
        ret = row["market_avg_return"]["mean"]
        dd = row["market_avg_max_dd"]["mean"]
        train = row["train_checkpoint_metric"]["mean"]
        print(
            f"{row['label']:<34} "
            f"{row['market_robust_score']:>8.2f} "
            f"{sortino:>9.2f} "
            f"{ret:>+8.2f} "
            f"{dd:>8.2f} "
            f"{train:>9.2f}",
            flush=True,
        )
    best = report["configs"][0]
    print(
        f"\nBest config: {best['label']} "
        f"(robust={best['market_robust_score']:.2f}, "
        f"sortino={best['market_avg_sortino']['mean']:.2f}, "
        f"return={best['market_avg_return']['mean']:+.2f}%)",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a multi-seed hourly trainer scout and summarize market-sim robustness.")
    p.add_argument("--symbols", default="AAPL,TSLA,NVDA")
    p.add_argument("--crypto-symbols", default="SOLUSD,AVAXUSD,ETHUSD,UNIUSD")
    p.add_argument("--backtest-symbols", default=None)
    p.add_argument("--forecast-horizons", default="1,24")
    p.add_argument("--seeds", default="1337,42,7")
    p.add_argument("--learning-rates", default="1e-4,3e-4")
    p.add_argument("--weight-decays", default="0.0,0.05")
    p.add_argument("--return-weights", default="0.08")
    p.add_argument("--fill-temperatures", default="0.0005")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--dry-train-steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--sequence-length", type=int, default=48)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--preload", type=Path, default=Path("unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt"))
    p.add_argument("--checkpoint-root", type=Path, default=Path("unified_hourly_experiment/checkpoints"))
    p.add_argument("--log-dir", type=Path, default=Path("tensorboard_logs") / "binanceneural")
    p.add_argument("--checkpoint-metric", default="robust_score")
    p.add_argument("--checkpoint-gap-penalty", type=float, default=0.25)
    p.add_argument("--top-k-checkpoints", type=int, default=10)
    p.add_argument("--maker-fee", type=float, default=0.001)
    p.add_argument("--backtest-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    p.add_argument("--backtest-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    p.add_argument("--backtest-validation-days", type=int, default=30)
    p.add_argument("--run-prefix", default=f"hourly_multiseed_{time.strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--baseline-label", default="")
    p.add_argument("--output", type=Path, default=Path("analysis") / f"hourly_multiseed_{time.strftime('%Y%m%d_%H%M%S')}.json")
    p.add_argument("--sortino-tolerance", type=float, default=1.0)
    p.add_argument("--return-tolerance-pct", type=float, default=0.5)
    p.add_argument("--drawdown-tolerance-pct", type=float, default=0.25)
    p.add_argument("--market-return-weight", type=float, default=0.10)
    p.add_argument("--market-std-penalty", type=float, default=1.0)
    p.add_argument("--market-drawdown-penalty", type=float, default=0.25)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-tags", default="")
    p.add_argument("--wandb-notes", default=None)
    p.add_argument("--wandb-mode", default="online")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    combos = build_combos(args)
    seeds = parse_csv_tokens(args.seeds, cast=int)
    baseline_label = args.baseline_label or combo_label(combos[0].as_dict())
    by_label: dict[str, list[dict[str, Any]]] = {}

    for combo in combos:
        label = combo_label(combo.as_dict())
        combo_runs: list[dict[str, Any]] = []
        for seed in seeds:
            run_name = build_run_name(prefix=args.run_prefix, combo=combo, seed=seed)
            checkpoint_dir = args.checkpoint_root / run_name
            maybe_run(args, train_command(args, combo, seed, run_name))
            if args.dry_run:
                continue
            best_checkpoint = resolve_best_checkpoint(checkpoint_dir)
            maybe_run(args, backtest_command(args, best_checkpoint))
            progress = load_training_progress(checkpoint_dir)
            backtest = load_backtest_summary(checkpoint_dir)
            combo_runs.append(
                {
                    "seed": seed,
                    "run_name": run_name,
                    "checkpoint_dir": str(checkpoint_dir),
                    "best_checkpoint": str(best_checkpoint),
                    "train": {
                        "checkpoint_metric_name": progress["checkpoint_metric_name"],
                        "checkpoint_metric": float(progress["checkpoint_metric"]),
                        "generalization_gap": float(progress["generalization_gap"]),
                        "best_metric": float(progress["best_metric"]),
                    },
                    "backtest": {
                        "epoch": int(backtest["epoch"]),
                        "avg_return": float(backtest["avg_return"]),
                        "avg_sortino": float(backtest["avg_sortino"]),
                        "avg_max_dd": float(backtest["avg_max_dd"]),
                    },
                }
            )
        by_label[label] = combo_runs

    if args.dry_run:
        return 0

    report_rows: list[dict[str, Any]] = []
    baseline_runs = by_label.get(baseline_label)
    if baseline_runs is None:
        raise KeyError(f"Unknown baseline label {baseline_label!r}")
    for combo in combos:
        label = combo_label(combo.as_dict())
        report_rows.append(
            summarize_config_runs(
                combo=combo,
                runs=by_label[label],
                baseline_runs=baseline_runs if label != baseline_label else None,
                args=args,
            )
        )
    report_rows.sort(key=lambda row: float(row["market_robust_score"]), reverse=True)

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_label": baseline_label,
        "seeds": seeds,
        "configs": report_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print_summary(report)
    print(f"\nSaved report to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
