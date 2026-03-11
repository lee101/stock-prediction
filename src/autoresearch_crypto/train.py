"""Crypto RL training script for autoresearch optimization.

This file is MODIFIABLE by LLM agents. The evaluation harness in prepare.py
is fixed and must not be changed.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from binanceneural.config import PolicyConfig
from binanceneural.model import BinancePolicyBase, build_policy
from differentiable_loss_utils import (
    HOURLY_PERIODS_PER_YEAR,
    compute_loss_by_type,
    simulate_hourly_trades,
    simulate_hourly_trades_binary,
)

from .prepare import (
    TIME_BUDGET,
    CryptoTaskConfig,
    evaluate_model,
    prepare_task,
    print_metrics,
    resolve_task_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Crypto RL autoresearch training")
    p.add_argument("--symbols", type=str, default=None)
    p.add_argument("--sequence-length", type=int, default=72)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--model-arch", type=str, default="classic")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["none", "cosine", "linear"])
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss-type", type=str, default="multiwindow_dd")
    p.add_argument("--return-weight", type=float, default=0.15)
    p.add_argument("--dd-penalty", type=float, default=1.0)
    p.add_argument("--smoothness-penalty", type=float, default=0.0)
    p.add_argument("--fill-temperature", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--maker-fee", type=float, default=0.001)
    p.add_argument("--max-leverage", type=float, default=2.0)
    p.add_argument("--margin-rate", type=float, default=0.0625)
    p.add_argument("--fill-buffer-pct", type=float, default=0.0005)
    p.add_argument("--decision-lag", type=int, default=1)
    p.add_argument("--holdout-days", type=int, default=30)
    p.add_argument("--val-days", type=int, default=70)
    p.add_argument("--eval-windows", type=str, default="72,168,336,720")
    p.add_argument("--forecast-horizons", type=str, default="1")
    p.add_argument("--trade-amount-scale", type=float, default=100.0)
    return p.parse_args(argv)


def build_cosine_lr_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(float(step + 1) / float(max(warmup_steps, 1)), min_ratio)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def main(argv=None) -> int:
    run_start = time.perf_counter()
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s seed=%d budget=%ds", device, args.seed, TIME_BUDGET)

    task_config = resolve_task_config(
        symbols=args.symbols,
        sequence_length=args.sequence_length,
        maker_fee=args.maker_fee,
        max_leverage=args.max_leverage,
        margin_annual_rate=args.margin_rate,
        fill_buffer_pct=args.fill_buffer_pct,
        decision_lag_bars=args.decision_lag,
        holdout_days=args.holdout_days,
        val_days=args.val_days,
        eval_windows=args.eval_windows,
        forecast_horizons=args.forecast_horizons,
        batch_size=args.batch_size,
    )

    task = prepare_task(task_config)
    n_features = len(task.feature_columns)
    logger.info("features=%d symbols=%s", n_features, ",".join(task_config.symbols))

    policy_cfg = PolicyConfig(
        input_dim=n_features,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        model_arch=args.model_arch,
        max_len=args.sequence_length,
        trade_amount_scale=args.trade_amount_scale,
    )
    model = build_policy(policy_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model params=%d arch=%s", n_params, args.model_arch)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loaders = {}
    val_loaders = {}
    for sym, module in task.train_modules.items():
        train_loaders[sym] = module.train_dataloader(args.batch_size)
        val_loaders[sym] = module.val_dataloader(args.batch_size)

    estimated_steps_per_epoch = sum(len(tl) for tl in train_loaders.values())
    estimated_total_steps = int(TIME_BUDGET / max(estimated_steps_per_epoch * 0.05, 1)) * estimated_steps_per_epoch
    estimated_total_steps = max(estimated_total_steps, 1000)

    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            build_cosine_lr_lambda(args.warmup_steps, estimated_total_steps),
        )

    best_val_score = float("-inf")
    best_state = None
    best_val_loss = float("inf")
    best_val_sortino = 0.0
    best_val_return = 0.0
    step_count = 0
    epoch_count = 0
    train_start = time.perf_counter()
    eval_reserve = 30.0

    while (time.perf_counter() - train_start) < (TIME_BUDGET - eval_reserve):
        epoch_count += 1
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for sym in task_config.symbols:
            loader = train_loaders[sym]
            for batch in loader:
                if (time.perf_counter() - train_start) >= (TIME_BUDGET - eval_reserve):
                    break

                features = batch["features"].to(device)
                highs = batch["high"].to(device)
                lows = batch["low"].to(device)
                closes = batch["close"].to(device)
                opens = batch["open"].to(device)
                ref_close = batch["reference_close"].to(device)
                ch_high = batch["chronos_high"].to(device)
                ch_low = batch["chronos_low"].to(device)

                outputs = model(features)
                actions = model.decode_actions(
                    outputs,
                    reference_close=ref_close,
                    chronos_high=ch_high,
                    chronos_low=ch_low,
                )

                scale = args.trade_amount_scale
                sim = simulate_hourly_trades(
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    opens=opens,
                    buy_prices=actions["buy_price"],
                    sell_prices=actions["sell_price"],
                    trade_intensity=actions["trade_amount"] / scale,
                    buy_trade_intensity=actions["buy_amount"] / scale,
                    sell_trade_intensity=actions["sell_amount"] / scale,
                    maker_fee=task_config.maker_fee,
                    initial_cash=1.0,
                    max_leverage=task_config.max_leverage,
                    can_short=task_config.can_short,
                    can_long=True,
                    temperature=args.fill_temperature,
                    decision_lag_bars=task_config.decision_lag_bars,
                    fill_buffer_pct=task_config.fill_buffer_pct,
                    margin_annual_rate=task_config.margin_annual_rate,
                )

                loss, score, sortino, annual_ret = compute_loss_by_type(
                    sim.returns.float(),
                    args.loss_type,
                    periods_per_year=HOURLY_PERIODS_PER_YEAR,
                    return_weight=args.return_weight,
                    smoothness_penalty=args.smoothness_penalty,
                    dd_penalty=args.dd_penalty,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                step_count += 1
                epoch_loss += float(loss.detach().item())
                epoch_steps += 1

        if epoch_steps == 0:
            break

        avg_train_loss = epoch_loss / epoch_steps

        model.eval()
        val_loss_sum = 0.0
        val_score_sum = 0.0
        val_sortino_sum = 0.0
        val_return_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for sym in task_config.symbols:
                for batch in val_loaders[sym]:
                    features = batch["features"].to(device)
                    highs = batch["high"].to(device)
                    lows = batch["low"].to(device)
                    closes = batch["close"].to(device)
                    opens = batch["open"].to(device)
                    ref_close = batch["reference_close"].to(device)
                    ch_high = batch["chronos_high"].to(device)
                    ch_low = batch["chronos_low"].to(device)

                    outputs = model(features)
                    actions = model.decode_actions(
                        outputs,
                        reference_close=ref_close,
                        chronos_high=ch_high,
                        chronos_low=ch_low,
                    )

                    scale = args.trade_amount_scale
                    sim = simulate_hourly_trades_binary(
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        opens=opens,
                        buy_prices=actions["buy_price"],
                        sell_prices=actions["sell_price"],
                        trade_intensity=actions["trade_amount"] / scale,
                        buy_trade_intensity=actions["buy_amount"] / scale,
                        sell_trade_intensity=actions["sell_amount"] / scale,
                        maker_fee=task_config.maker_fee,
                        initial_cash=1.0,
                        max_leverage=task_config.max_leverage,
                        can_short=task_config.can_short,
                        can_long=True,
                        decision_lag_bars=task_config.decision_lag_bars,
                        fill_buffer_pct=task_config.fill_buffer_pct,
                        margin_annual_rate=task_config.margin_annual_rate,
                    )

                    loss_v, score_v, sort_v, ret_v = compute_loss_by_type(
                        sim.returns.float(),
                        args.loss_type,
                        periods_per_year=HOURLY_PERIODS_PER_YEAR,
                        return_weight=args.return_weight,
                        dd_penalty=args.dd_penalty,
                    )
                    val_loss_sum += float(loss_v.detach().item())
                    val_score_sum += float(score_v.detach().mean().item())
                    val_sortino_sum += float(sort_v.detach().mean().item())
                    val_return_sum += float(ret_v.detach().mean().item())
                    val_steps += 1

        if val_steps > 0:
            avg_val_loss = val_loss_sum / val_steps
            avg_val_score = val_score_sum / val_steps
            avg_val_sortino = val_sortino_sum / val_steps
            avg_val_return = val_return_sum / val_steps
        else:
            avg_val_loss = float("inf")
            avg_val_score = float("-inf")
            avg_val_sortino = 0.0
            avg_val_return = 0.0

        logger.info(
            "ep%d steps=%d train_loss=%.4f val_loss=%.4f val_sort=%.2f val_ret=%.2f%%",
            epoch_count, step_count, avg_train_loss, avg_val_loss,
            avg_val_sortino, avg_val_return * 100,
        )

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            best_val_loss = avg_val_loss
            best_val_sortino = avg_val_sortino
            best_val_return = avg_val_return
            best_state = copy.deepcopy(model.state_dict())

    training_seconds = time.perf_counter() - train_start
    logger.info("training done: %d epochs, %d steps, %.1fs", epoch_count, step_count, training_seconds)

    if best_state is not None:
        model.load_state_dict(best_state)

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    eval_result = evaluate_model(model, task, device=device)
    total_seconds = time.perf_counter() - run_start

    print_metrics(
        eval_result["summary"],
        val_loss=best_val_loss,
        val_sortino=best_val_sortino,
        val_return_pct=best_val_return * 100,
        training_seconds=training_seconds,
        total_seconds=total_seconds,
        peak_vram_mb=peak_vram_mb,
        num_steps=step_count,
        num_epochs=epoch_count,
        symbols=",".join(task_config.symbols),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
