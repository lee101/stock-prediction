from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from differentiable_market.differentiable_utils import soft_drawdown

from .config import E2EDataConfig, E2EModelConfig, E2ETrainConfig
from .data import StockDataset, load_stock_dataset, sample_start_indices, split_dataset
from .model import ChronosTradingPolicy


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _make_run_dir(cfg: E2ETrainConfig) -> Path:
    stamp = cfg.run_name or time.strftime("%Y%m%d_%H%M%S")
    return _ensure_dir(cfg.save_root / stamp)


def _build_optimizer(model: ChronosTradingPolicy, cfg: E2ETrainConfig) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)
    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(cfg.backbone_learning_rate),
                "weight_decay": float(cfg.weight_decay),
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": float(cfg.learning_rate),
                "weight_decay": float(cfg.weight_decay),
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer")
    return torch.optim.AdamW(param_groups)


def _portfolio_terms(
    *,
    step_weights: torch.Tensor,
    prev_weights: torch.Tensor,
    next_returns: torch.Tensor,
    transaction_cost_bps: float,
    include_cash: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    returns_with_cash = next_returns.to(step_weights)
    if include_cash:
        returns_with_cash = torch.cat(
            [
                returns_with_cash,
                torch.zeros(1, device=step_weights.device, dtype=step_weights.dtype),
            ],
            dim=0,
        )
    if step_weights.shape != prev_weights.shape:
        raise ValueError("step_weights and prev_weights must have matching shapes")
    if step_weights.numel() != returns_with_cash.numel():
        raise ValueError("Portfolio weights must align with tradable returns")
    turnover = torch.abs(step_weights - prev_weights).sum()
    cost = turnover * (float(transaction_cost_bps) / 10000.0)
    gross = torch.sum(step_weights * returns_with_cash)
    net = torch.clamp(gross - cost, min=-0.95)
    log_growth = torch.log1p(net)
    return net, log_growth


def _rollout_loss(
    *,
    model: ChronosTradingPolicy,
    close: torch.Tensor,
    start_idx: int,
    model_cfg: E2EModelConfig,
    train_cfg: E2ETrainConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    context_length = int(model_cfg.context_length)
    rollout_length = int(train_cfg.rollout_length)
    asset_count = close.shape[1]
    portfolio_size = asset_count + (1 if model_cfg.include_cash else 0)
    prev_weights = torch.zeros(portfolio_size, device=device, dtype=torch.float32)
    if model_cfg.include_cash:
        prev_weights[-1] = 1.0
    else:
        prev_weights.fill_(1.0 / max(asset_count, 1))

    portfolio_returns = []
    log_growth = []
    forecast_losses = []
    last_weights = prev_weights

    for step in range(rollout_length):
        context_start = start_idx + step
        context_end = context_start + context_length
        next_idx = context_end
        context_close = close[context_start:context_end].transpose(0, 1)
        actual_next_close = close[next_idx]

        output = model(context_close, actual_next_close, prev_weights=prev_weights)
        next_returns = (actual_next_close / close[next_idx - 1].clamp_min(1e-8)) - 1.0
        step_return, step_log = _portfolio_terms(
            step_weights=output.weights,
            prev_weights=prev_weights,
            next_returns=next_returns,
            transaction_cost_bps=train_cfg.transaction_cost_bps,
            include_cash=model_cfg.include_cash,
        )
        prev_weights = output.weights
        last_weights = output.weights
        portfolio_returns.append(step_return)
        log_growth.append(step_log)
        forecast_losses.append(output.aux_forecast_loss)

    returns_tensor = torch.stack(portfolio_returns)
    log_growth_tensor = torch.stack(log_growth)
    forecast_loss = torch.stack(forecast_losses).mean()

    downside = torch.clamp_min(-returns_tensor, 0.0)
    downside_std = torch.sqrt(torch.mean(downside.square()) + 1e-8)
    sortino = returns_tensor.mean() / downside_std
    _, drawdown = soft_drawdown(log_growth_tensor.unsqueeze(0))
    drawdown_penalty = drawdown.mean()

    objective = (
        log_growth_tensor.mean()
        + float(train_cfg.sortino_weight) * sortino
        - float(train_cfg.drawdown_weight) * drawdown_penalty
    )
    loss = -objective + float(train_cfg.forecast_loss_weight) * forecast_loss

    return {
        "loss": loss,
        "objective": objective.detach(),
        "mean_return": returns_tensor.mean().detach(),
        "sortino": sortino.detach(),
        "drawdown": drawdown_penalty.detach(),
        "forecast_loss": forecast_loss.detach(),
        "cash_weight": (
            last_weights[-1].detach()
            if model_cfg.include_cash
            else torch.zeros((), device=device, dtype=last_weights.dtype)
        ),
    }


def _evaluate(
    *,
    model: ChronosTradingPolicy,
    dataset: StockDataset,
    model_cfg: E2EModelConfig,
    train_cfg: E2ETrainConfig,
    device: torch.device,
) -> dict[str, float]:
    close = dataset.close.to(device=device, dtype=torch.float32)
    max_rollouts = min(4, max(1, (close.shape[0] - model_cfg.context_length - train_cfg.rollout_length - 1) // max(train_cfg.rollout_length, 1)))
    starts = []
    total = close.shape[0] - model_cfg.context_length - train_cfg.rollout_length - 1
    if total <= 0:
        raise ValueError("Validation dataset is too short for the requested context/rollout")
    stride = max(1, total // max_rollouts)
    for idx in range(max_rollouts):
        starts.append(min(idx * stride, total))

    aggregates = {"objective": 0.0, "mean_return": 0.0, "sortino": 0.0, "drawdown": 0.0, "forecast_loss": 0.0}
    model.eval()
    with torch.no_grad():
        for start in starts:
            metrics = _rollout_loss(
                model=model,
                close=close,
                start_idx=int(start),
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=device,
            )
            for key in aggregates:
                aggregates[key] += float(metrics[key])
    model.train()
    count = float(len(starts))
    return {f"val_{key}": value / count for key, value in aggregates.items()}


def run_training(
    data_cfg: E2EDataConfig,
    model_cfg: E2EModelConfig,
    train_cfg: E2ETrainConfig,
) -> Path:
    _set_seed(train_cfg.seed)
    run_dir = _make_run_dir(train_cfg)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoints_dir = _ensure_dir(run_dir / "checkpoints")

    dataset = load_stock_dataset(data_cfg)
    train_ds, val_ds = split_dataset(dataset, data_cfg.train_ratio)

    device = torch.device(train_cfg.device if train_cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ChronosTradingPolicy(model_cfg, device=str(device), torch_dtype=train_cfg.torch_dtype)
    model.to(device)
    optimizer = _build_optimizer(model, train_cfg)
    generator = torch.Generator().manual_seed(train_cfg.seed)

    config_snapshot = {
        "data": asdict(data_cfg),
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
        "symbols": dataset.symbols,
        "train_timesteps": len(train_ds.index),
        "val_timesteps": len(val_ds.index),
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2, default=_json_default) + "\n")

    best_objective = float("-inf")
    train_close = train_ds.close.to(device=device, dtype=torch.float32)
    for step in range(1, train_cfg.steps + 1):
        starts = sample_start_indices(
            total_steps=train_close.shape[0],
            context_length=model_cfg.context_length,
            rollout_length=train_cfg.rollout_length,
            batch_size=train_cfg.batch_size,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        batch_metrics: dict[str, torch.Tensor] = {}
        batch_loss = torch.zeros((), device=device, dtype=torch.float32)
        for start in starts.tolist():
            metrics = _rollout_loss(
                model=model,
                close=train_close,
                start_idx=int(start),
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=device,
            )
            batch_loss = batch_loss + metrics["loss"]
            for key, value in metrics.items():
                batch_metrics[key] = batch_metrics.get(key, torch.zeros_like(value)) + value.detach()
        batch_loss = batch_loss / len(starts)
        batch_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=float(train_cfg.grad_clip))
        optimizer.step()

        train_record = {
            "phase": "train",
            "step": step,
            "loss": float(batch_loss.detach()),
        }
        for key, value in batch_metrics.items():
            train_record[key] = float(value / len(starts))
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(train_record) + "\n")

        if step % max(1, train_cfg.log_every) == 0 or step == 1:
            print(
                f"[train step {step}] loss={train_record['loss']:.4f} "
                f"obj={train_record['objective']:.4f} ret={train_record['mean_return']:.4f} "
                f"sortino={train_record['sortino']:.4f} dd={train_record['drawdown']:.4f}",
                flush=True,
            )

        if step % max(1, train_cfg.eval_every) == 0 or step == train_cfg.steps:
            eval_metrics = _evaluate(
                model=model,
                dataset=val_ds,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=device,
            )
            eval_record = {"phase": "eval", "step": step, **eval_metrics}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_record) + "\n")
            print(
                f"[eval step {step}] obj={eval_metrics['val_objective']:.4f} "
                f"ret={eval_metrics['val_mean_return']:.4f} sortino={eval_metrics['val_sortino']:.4f} "
                f"dd={eval_metrics['val_drawdown']:.4f} forecast={eval_metrics['val_forecast_loss']:.4f}",
                flush=True,
            )
            checkpoint = {
                "model_state": model.state_dict(),
                "data_cfg": asdict(data_cfg),
                "model_cfg": asdict(model_cfg),
                "train_cfg": asdict(train_cfg),
                "eval_metrics": eval_metrics,
                "symbols": dataset.symbols,
            }
            torch.save(checkpoint, checkpoints_dir / "latest.pt")
            if eval_metrics["val_objective"] > best_objective:
                best_objective = eval_metrics["val_objective"]
                torch.save(checkpoint, checkpoints_dir / "best.pt")

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end Chronos2 + portfolio training on stock daily data.")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    parser.add_argument("--universe-file", type=Path, default=Path("available_stocks_with_data.json"))
    parser.add_argument("--max-assets", type=int, default=64)
    parser.add_argument("--include-symbols", default=None, help="Comma-separated symbols to include")
    parser.add_argument("--exclude-symbols", default=None, help="Comma-separated symbols to exclude")
    parser.add_argument("--min-timesteps", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument(
        "--cross-learning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Chronos cross-learning across assets",
    )
    parser.add_argument("--no-cash", action="store_true")
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--rollout-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--forecast-loss-weight", type=float, default=0.25)
    parser.add_argument("--sortino-weight", type=float, default=0.10)
    parser.add_argument("--drawdown-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-root", type=Path, default=Path("e2etraining") / "runs")
    parser.add_argument("--log-every", type=int, default=5)
    return parser


def _parse_symbol_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    values = []
    for token in str(raw).split(","):
        symbol = token.strip().upper()
        if symbol:
            values.append(symbol)
    return tuple(values)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_cfg = E2EDataConfig(
        data_root=args.data_root,
        universe_file=args.universe_file,
        include_symbols=_parse_symbol_csv(args.include_symbols),
        exclude_symbols=_parse_symbol_csv(args.exclude_symbols),
        max_assets=int(args.max_assets),
        min_timesteps=int(args.min_timesteps),
        train_ratio=float(args.train_ratio),
    )
    model_cfg = E2EModelConfig(
        model_id=str(args.model_id),
        context_length=int(args.context_length),
        prediction_length=int(args.prediction_length),
        cross_learning=bool(args.cross_learning),
        include_cash=not bool(args.no_cash),
        lora_enabled=not bool(args.disable_lora),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
    )
    train_cfg = E2ETrainConfig(
        run_name=args.run_name,
        save_root=args.save_root,
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        seed=int(args.seed),
        steps=int(args.steps),
        eval_every=int(args.eval_every),
        rollout_length=int(args.rollout_length),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        backbone_learning_rate=float(args.backbone_learning_rate),
        weight_decay=float(args.weight_decay),
        transaction_cost_bps=float(args.transaction_cost_bps),
        forecast_loss_weight=float(args.forecast_loss_weight),
        sortino_weight=float(args.sortino_weight),
        drawdown_weight=float(args.drawdown_weight),
        grad_clip=float(args.grad_clip),
        log_every=int(args.log_every),
    )
    run_dir = run_training(data_cfg, model_cfg, train_cfg)
    print(f"Run saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
