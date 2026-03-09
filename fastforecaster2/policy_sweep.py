from __future__ import annotations

import argparse
import itertools
import json
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import ContextManager, Iterable, Iterator, Optional

import torch

from wandboard import WandBoardLogger

from .config import FastForecaster2Config
from .trainer import FastForecaster2Trainer


@dataclass(frozen=True)
class PolicySweepSpec:
    buy_threshold: float
    sell_threshold: float
    entry_score_threshold: float
    top_k: int
    ema_alpha: float
    max_hold_hours: int | None
    max_trade_intensity: float
    min_trade_intensity: float
    switch_score_gap: float


def _summary_rank_key(summary: dict[str, object]) -> tuple[float, float, float, float]:
    if "sim_goodness_score" in summary:
        return (
            float(summary.get("sim_goodness_score", float("-inf"))),
            float(summary.get("sim_pnl", float("-inf"))),
            -float(summary.get("sim_max_drawdown", 1.0)),
            float(summary.get("sim_smoothness", 0.0)),
        )
    return (
        float(summary.get("sim_pnl", float("-inf"))),
        -float(summary.get("sim_max_drawdown", 1.0)),
        float(summary.get("sim_smoothness", 0.0)),
        float(summary.get("sim_sortino", float("-inf"))),
    )


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one float value.")
    return tuple(values)


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one int value.")
    return tuple(values)


def _parse_optional_int_list(raw: str) -> tuple[int | None, ...]:
    values: list[int | None] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"none", "null"}:
            values.append(None)
        else:
            values.append(int(token))
    if not values:
        raise ValueError("Expected at least one max-hold value.")
    return tuple(values)


def _parse_tags(raw: str) -> tuple[str, ...]:
    return tuple(sorted({item.strip() for item in raw.split(",") if item.strip()}))


def _iter_specs(
    *,
    buy_thresholds: Iterable[float],
    sell_thresholds: Iterable[float],
    entry_score_thresholds: Iterable[float],
    top_ks: Iterable[int],
    ema_alphas: Iterable[float],
    max_hold_hours_values: Iterable[int | None],
    max_trade_intensities: Iterable[float],
    min_trade_intensities: Iterable[float],
    switch_score_gaps: Iterable[float],
) -> Iterator[PolicySweepSpec]:
    for (
        buy_threshold,
        sell_threshold,
        entry_score_threshold,
        top_k,
        ema_alpha,
        max_hold_hours,
        max_trade_intensity,
        min_trade_intensity,
        switch_score_gap,
    ) in (
        itertools.product(
            buy_thresholds,
            sell_thresholds,
            entry_score_thresholds,
            top_ks,
            ema_alphas,
            max_hold_hours_values,
            max_trade_intensities,
            min_trade_intensities,
            switch_score_gaps,
        )
    ):
        if sell_threshold > buy_threshold:
            continue
        if entry_score_threshold < 0:
            continue
        if min_trade_intensity > max_trade_intensity:
            continue
        if switch_score_gap < 0:
            continue
        yield PolicySweepSpec(
            buy_threshold=float(buy_threshold),
            sell_threshold=float(sell_threshold),
            entry_score_threshold=float(entry_score_threshold),
            top_k=int(top_k),
            ema_alpha=float(ema_alpha),
            max_hold_hours=max_hold_hours,
            max_trade_intensity=float(max_trade_intensity),
            min_trade_intensity=float(min_trade_intensity),
            switch_score_gap=float(switch_score_gap),
        )


def _trial_name(spec: PolicySweepSpec) -> str:
    def format_float(value: float) -> str:
        return f"{value:.8g}"

    hold_label = "none" if spec.max_hold_hours is None else str(spec.max_hold_hours)
    return (
        f"b{format_float(spec.buy_threshold)}_"
        f"s{format_float(spec.sell_threshold)}_"
        f"es{format_float(spec.entry_score_threshold)}_"
        f"k{spec.top_k}_"
        f"ema{format_float(spec.ema_alpha)}_"
        f"h{hold_label}_"
        f"maxi{format_float(spec.max_trade_intensity)}_"
        f"mini{format_float(spec.min_trade_intensity)}_"
        f"sgap{format_float(spec.switch_score_gap)}"
    ).replace("+0", "").replace(".", "p")


def _logger_context(
    *,
    args: argparse.Namespace,
    cfg: FastForecaster2Config,
    run_name: str,
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
        tensorboard_subdir=f"fastforecaster2/policy_sweeps/{run_name}",
        enable_wandb=True,
        log_metrics=True,
        config={"fastforecaster2_policy_sweep": cfg.as_dict()},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint-only policy sweep for FastForecaster2 market simulation.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("fastforecaster2") / "policy_sweeps")
    parser.add_argument("--buy-thresholds", type=str, default="3e-5")
    parser.add_argument("--sell-thresholds", type=str, default="1.5e-5")
    parser.add_argument("--entry-score-thresholds", type=str, default="")
    parser.add_argument("--top-ks", type=str, default="1")
    parser.add_argument("--ema-alphas", type=str, default="0.55")
    parser.add_argument("--max-hold-hours-values", type=str, default="6")
    parser.add_argument("--max-trade-intensities", type=str, default="8,10,12")
    parser.add_argument("--min-trade-intensities", type=str, default="")
    parser.add_argument("--switch-score-gaps", type=str, default="")
    parser.add_argument("--limit-trials", type=int, default=0)

    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-run-prefix", type=str, default="fastforecaster2_policy_sweep")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    base_cfg = FastForecaster2Config(**checkpoint["config"])

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"policy_sweep_{run_stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    min_trade_intensity_tokens = (
        _parse_float_list(args.min_trade_intensities)
        if args.min_trade_intensities.strip()
        else (float(base_cfg.market_sim_min_trade_intensity),)
    )
    entry_score_threshold_tokens = (
        _parse_float_list(args.entry_score_thresholds)
        if args.entry_score_thresholds.strip()
        else (float(base_cfg.market_sim_entry_score_threshold),)
    )
    switch_score_gap_tokens = (
        _parse_float_list(args.switch_score_gaps)
        if args.switch_score_gaps.strip()
        else (float(base_cfg.market_sim_switch_score_gap),)
    )
    specs = list(
        _iter_specs(
            buy_thresholds=_parse_float_list(args.buy_thresholds),
            sell_thresholds=_parse_float_list(args.sell_thresholds),
            entry_score_thresholds=entry_score_threshold_tokens,
            top_ks=_parse_int_list(args.top_ks),
            ema_alphas=_parse_float_list(args.ema_alphas),
            max_hold_hours_values=_parse_optional_int_list(args.max_hold_hours_values),
            max_trade_intensities=_parse_float_list(args.max_trade_intensities),
            min_trade_intensities=min_trade_intensity_tokens,
            switch_score_gaps=switch_score_gap_tokens,
        )
    )
    if args.limit_trials > 0:
        specs = specs[: args.limit_trials]
    if not specs:
        raise ValueError("No valid policy sweep trials generated.")

    trainer_cfg = replace(
        base_cfg,
        output_dir=sweep_dir / "base",
        wandb_project=None,
        wandb_run_name=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_tags=(),
    )
    trainer = FastForecaster2Trainer(trainer_cfg, metrics_logger=None)
    trainer.model.load_state_dict(checkpoint["model_state"])
    trainer.model.eval()

    results: list[dict[str, object]] = []
    tags = _parse_tags(args.wandb_tags)

    print(f"[fastforecaster2-policy-sweep] Running {len(specs)} policy trials from {checkpoint_path}")
    for index, spec in enumerate(specs, start=1):
        trial_name = _trial_name(spec)
        cfg = replace(
            trainer_cfg,
            output_dir=sweep_dir / trial_name,
            market_sim_buy_threshold=spec.buy_threshold,
            market_sim_sell_threshold=spec.sell_threshold,
            market_sim_entry_score_threshold=spec.entry_score_threshold,
            market_sim_top_k=spec.top_k,
            market_sim_signal_ema_alpha=spec.ema_alpha,
            market_sim_max_hold_hours=spec.max_hold_hours,
            market_sim_max_trade_intensity=spec.max_trade_intensity,
            market_sim_min_trade_intensity=spec.min_trade_intensity,
            market_sim_switch_score_gap=spec.switch_score_gap,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            wandb_tags=tags,
        )
        cfg.ensure_output_dirs()
        trainer.config = cfg

        run_name = f"{args.wandb_run_prefix}_{trial_name}_{run_stamp}"
        with _logger_context(args=args, cfg=cfg, run_name=run_name) as metrics_logger:
            summary = trainer._run_market_sim_eval()
            if metrics_logger is not None:
                metrics_logger.log({f"summary/{k}": v for k, v in summary.items()}, step=index)

        row = {
            "trial_index": index,
            "trial_name": trial_name,
            "checkpoint_path": str(checkpoint_path),
            "config": {
                "market_sim_buy_threshold": spec.buy_threshold,
                "market_sim_sell_threshold": spec.sell_threshold,
                "market_sim_entry_score_threshold": spec.entry_score_threshold,
                "market_sim_top_k": spec.top_k,
                "market_sim_signal_ema_alpha": spec.ema_alpha,
                "market_sim_max_hold_hours": spec.max_hold_hours,
                "market_sim_max_trade_intensity": spec.max_trade_intensity,
                "market_sim_min_trade_intensity": spec.min_trade_intensity,
                "market_sim_switch_score_gap": spec.switch_score_gap,
            },
            "summary": summary,
        }
        results.append(row)
        compact = {
            "trial_name": trial_name,
            "sim_goodness_score": summary.get("sim_goodness_score"),
            "sim_pnl": summary.get("sim_pnl"),
            "sim_max_drawdown": summary.get("sim_max_drawdown"),
            "sim_smoothness": summary.get("sim_smoothness"),
            "sim_trades": summary.get("sim_trades"),
        }
        print(json.dumps(compact))

    results.sort(key=lambda row: _summary_rank_key(row["summary"]), reverse=True)
    payload = {
        "run_timestamp_utc": run_stamp,
        "checkpoint_path": str(checkpoint_path),
        "trial_count": len(results),
        "results": results,
        "best_trial": results[0] if results else None,
    }
    out_path = sweep_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if results:
        (sweep_dir / "best_policy.json").write_text(json.dumps(results[0], indent=2), encoding="utf-8")
        print(
            "[fastforecaster2-policy-sweep] Best trial: "
            f"{results[0]['trial_name']} sim_goodness={results[0]['summary'].get('sim_goodness_score')} "
            f"sim_pnl={results[0]['summary'].get('sim_pnl')}"
        )
    print(f"[fastforecaster2-policy-sweep] Results written to {out_path}")


if __name__ == "__main__":
    main()
