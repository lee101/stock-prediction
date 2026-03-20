#!/usr/bin/env python3
"""Train neural daily work-steal policy with multi-step rollout simulator."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from differentiable_loss_utils import (
    DAILY_PERIODS_PER_YEAR_CRYPTO,
    approx_buy_fill_probability,
    approx_sell_fill_probability,
    compute_loss_by_type,
)
from binance_worksteal.model import DailyWorkStealPolicy, PerSymbolWorkStealPolicy
from binance_worksteal.data import (
    build_datasets,
    build_sequential_datasets,
    build_dataloader,
    FEATURE_NAMES,
    N_MARKET_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_EPS = 1e-8


def simulate_daily_trades(
    *,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    intensity: torch.Tensor,
    maker_fee: float = 0.001,
    initial_cash: float = 1.0,
    temperature: float = 5e-4,
    max_positions: int = 5,
):
    """Differentiable daily trade simulator for one step."""
    B, S = highs.shape
    device = highs.device
    dtype = highs.dtype

    fee = torch.tensor(maker_fee, dtype=dtype, device=device)

    intensity_sum = intensity.sum(dim=-1, keepdim=True).clamp(min=_EPS)
    max_alloc = float(max_positions) / max(S, 1)
    norm_intensity = intensity * torch.clamp(max_alloc / intensity_sum, max=1.0)

    buy_prob = approx_buy_fill_probability(
        buy_prices, lows, closes, temperature=temperature
    )
    sell_prob = approx_sell_fill_probability(
        sell_prices, highs, closes, temperature=temperature
    )

    cash_per_symbol = initial_cash * norm_intensity

    buy_qty = torch.where(
        buy_prices > _EPS,
        cash_per_symbol / (buy_prices * (1.0 + fee)).clamp(min=_EPS),
        torch.zeros_like(cash_per_symbol),
    )
    executed_buy_qty = buy_qty * buy_prob
    buy_cost = executed_buy_qty * buy_prices * (1.0 + fee)

    executed_sell_qty = executed_buy_qty * sell_prob
    sell_proceeds = executed_sell_qty * sell_prices * (1.0 - fee)

    unsold_qty = executed_buy_qty - executed_sell_qty
    unsold_value = unsold_qty * closes

    undeployed = initial_cash * (1.0 - norm_intensity.sum(dim=-1))

    portfolio_value = undeployed + sell_proceeds.sum(dim=-1) + unsold_value.sum(dim=-1) + \
                      (initial_cash * norm_intensity - buy_cost).sum(dim=-1)

    returns = (portfolio_value - initial_cash) / initial_cash

    return {
        "portfolio_value": portfolio_value,
        "returns": returns,
        "buy_prob": buy_prob,
        "sell_prob": sell_prob,
        "executed_buy_qty": executed_buy_qty,
        "executed_sell_qty": executed_sell_qty,
    }


def run_sequential_sim(
    *,
    features_seq: torch.Tensor,
    ohlcv_seq: torch.Tensor,
    target_highs: torch.Tensor,
    target_lows: torch.Tensor,
    target_closes: torch.Tensor,
    current_closes: torch.Tensor,
    model,
    maker_fee: float = 0.001,
    initial_cash: float = 10000.0,
    temperature: float = 5e-4,
    max_positions: int = 5,
):
    """Single-step sim (backward compat)."""
    actions = model(features_seq)

    buy_prices = current_closes * (1.0 - actions["buy_offset"])
    sell_prices = current_closes * (1.0 + actions["sell_offset"])

    sim = simulate_daily_trades(
        highs=target_highs,
        lows=target_lows,
        closes=target_closes,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        intensity=actions["intensity"],
        maker_fee=maker_fee,
        initial_cash=initial_cash,
        temperature=temperature,
        max_positions=max_positions,
    )
    return sim, actions


def _rollout_step(
    feat, cur_close, t_high, t_low, t_close,
    has_pos, entry_price, hold_days, cash,
    model, fee, temperature, max_positions, max_hold_days,
):
    """Single step of multi-step rollout. Extracted for gradient checkpointing."""
    actions = model(feat)

    buy_target = cur_close * (1.0 - actions["buy_offset"])

    tp_price = entry_price * (1.0 + actions["sell_offset"])
    tp_prob = approx_sell_fill_probability(
        tp_price, t_high, t_close, temperature=temperature,
    )
    force_close = (hold_days >= max_hold_days).float()
    exit_prob = has_pos * (tp_prob + force_close - tp_prob * force_close)

    exit_price = torch.where(force_close > 0.5, t_close, tp_price)
    exit_pnl_per_sym = exit_prob * (
        (exit_price - entry_price) / entry_price.clamp(min=_EPS) - 2.0 * fee
    )

    no_pos = 1.0 - has_pos
    buy_prob = approx_buy_fill_probability(
        buy_target, t_low, t_close, temperature=temperature,
    )
    effective_buy = no_pos * buy_prob * actions["intensity"]

    n_held = (has_pos * (1.0 - exit_prob)).sum(dim=-1, keepdim=True)
    n_new = effective_buy.sum(dim=-1, keepdim=True)
    scale = torch.clamp(float(max_positions) / (n_held + n_new).clamp(min=_EPS), max=1.0)
    effective_buy = effective_buy * scale

    alloc_sum = effective_buy.sum(dim=-1, keepdim=True).clamp(min=_EPS)
    alloc_frac = effective_buy / alloc_sum
    deployable = cash.unsqueeze(-1) * alloc_frac
    buy_qty = torch.where(
        buy_target > _EPS,
        deployable / (buy_target * (1.0 + fee)).clamp(min=_EPS),
        torch.zeros_like(buy_target),
    )
    buy_cost = buy_qty * buy_target * (1.0 + fee)

    new_has_pos = (has_pos * (1.0 - exit_prob) + effective_buy).clamp(0.0, 1.0)
    new_entry = torch.where(effective_buy > 0.01, buy_target, entry_price * (1.0 - exit_prob))
    new_hold_days = torch.where(
        effective_buy > 0.01,
        torch.zeros_like(hold_days),
        (hold_days + 1.0) * (1.0 - exit_prob),
    )

    mtm_change = has_pos * (1.0 - exit_prob) * (
        (t_close - cur_close) / cur_close.clamp(min=_EPS)
    )
    step_ret = (exit_pnl_per_sym.sum(dim=-1) + mtm_change.sum(dim=-1)) / cash.clamp(min=_EPS)

    new_unrealized_pnl = torch.where(
        new_has_pos > 0.01,
        (t_close - new_entry) / new_entry.clamp(min=_EPS),
        torch.zeros_like(has_pos),
    )
    exit_proceeds = (exit_prob * entry_price * (1.0 + (exit_price - entry_price) / entry_price.clamp(min=_EPS)) * (1.0 - fee)).sum(dim=-1)
    new_cash = cash - buy_cost.sum(dim=-1) + exit_proceeds

    return step_ret, new_has_pos, new_entry, new_hold_days, new_unrealized_pnl, new_cash


def run_multistep_rollout(
    *,
    features_seq: torch.Tensor,
    target_highs: torch.Tensor,
    target_lows: torch.Tensor,
    target_closes: torch.Tensor,
    current_closes: torch.Tensor,
    model,
    maker_fee: float = 0.001,
    initial_cash: float = 10000.0,
    temperature: float = 5e-4,
    max_positions: int = 5,
    max_hold_days: int = 14,
    use_grad_checkpoint: bool = False,
):
    """Multi-step rollout with position state tracking.

    Args:
        features_seq:  [B, R, T, S, F]
        target_highs:  [B, R, S]
        target_lows:   [B, R, S]
        target_closes: [B, R, S]
        current_closes:[B, R, S]
        use_grad_checkpoint: recompute activations in backward to save memory
    """
    B, R, T, S, F = features_seq.shape
    device = features_seq.device
    dtype = features_seq.dtype

    fee = torch.tensor(maker_fee, dtype=dtype, device=device)
    hold_days_scale = 1.0 / max(max_hold_days, 1)

    has_pos = torch.zeros(B, S, dtype=dtype, device=device)
    entry_price = torch.zeros(B, S, dtype=dtype, device=device)
    hold_days = torch.zeros(B, S, dtype=dtype, device=device)
    unrealized_pnl = torch.zeros(B, S, dtype=dtype, device=device)
    cash = torch.full((B,), initial_cash, dtype=dtype, device=device)

    # Pre-split along rollout dim to avoid repeated indexing
    feat_steps = features_seq.unbind(dim=1)
    cur_close_steps = current_closes.unbind(dim=1)
    t_high_steps = target_highs.unbind(dim=1)
    t_low_steps = target_lows.unbind(dim=1)
    t_close_steps = target_closes.unbind(dim=1)

    step_returns = []

    for step in range(R):
        feat = feat_steps[step].clone()
        feat[:, -1, :, N_MARKET_FEATURES] = has_pos
        feat[:, -1, :, N_MARKET_FEATURES + 1] = unrealized_pnl.clamp(-1.0, 1.0)
        feat[:, -1, :, N_MARKET_FEATURES + 2] = (hold_days * hold_days_scale).clamp(0.0, 1.0)

        cur_close = cur_close_steps[step]
        t_high = t_high_steps[step]
        t_low = t_low_steps[step]
        t_close = t_close_steps[step]

        if use_grad_checkpoint and torch.is_grad_enabled():
            step_ret, has_pos, entry_price, hold_days, unrealized_pnl, cash = grad_checkpoint(
                _rollout_step,
                feat, cur_close, t_high, t_low, t_close,
                has_pos, entry_price, hold_days, cash,
                model, fee, temperature, max_positions, max_hold_days,
                use_reentrant=False,
            )
        else:
            step_ret, has_pos, entry_price, hold_days, unrealized_pnl, cash = _rollout_step(
                feat, cur_close, t_high, t_low, t_close,
                has_pos, entry_price, hold_days, cash,
                model, fee, temperature, max_positions, max_hold_days,
            )
        step_returns.append(step_ret)

    returns_tensor = torch.stack(step_returns, dim=-1)  # [B, R]
    return {
        "returns": returns_tensor,
        "final_has_pos": has_pos,
        "final_cash": cash,
    }


def _backward_step(loss, optimizer, model, grad_clip, scaler):
    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


def train_epoch(model, loader, optimizer, device, config, scaler=None):
    model.train()
    total_loss = 0.0
    total_return = 0.0
    steps = 0
    use_amp = scaler is not None

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        target_high = batch["target_high"].to(device, non_blocking=True)
        target_low = batch["target_low"].to(device, non_blocking=True)
        target_close = batch["target_close"].to(device, non_blocking=True)
        current_close = batch["current_close"].to(device, non_blocking=True)
        ohlcv = batch["ohlcv"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            sim, actions = run_sequential_sim(
                features_seq=features,
                ohlcv_seq=ohlcv,
                target_highs=target_high,
                target_lows=target_low,
                target_closes=target_close,
                current_closes=current_close,
                model=model,
                maker_fee=config["maker_fee"],
                initial_cash=config["initial_cash"],
                temperature=config["temperature"],
                max_positions=config["max_positions"],
            )
            returns = sim["returns"]

        loss, score, sortino, annual_ret = compute_loss_by_type(
            returns.float().unsqueeze(0),
            config["loss_type"],
            periods_per_year=DAILY_PERIODS_PER_YEAR_CRYPTO,
            return_weight=config["return_weight"],
        )

        _backward_step(loss, optimizer, model, config["grad_clip"], scaler)

        total_loss += loss.item()
        total_return += returns.mean().item()
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "mean_return": total_return / max(steps, 1),
    }


def train_epoch_multistep(model, loader, optimizer, device, config, scaler=None):
    model.train()
    total_loss = 0.0
    total_return = 0.0
    steps = 0
    use_amp = scaler is not None
    use_gc = config.get("grad_checkpoint", False)

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        target_high = batch["target_high"].to(device, non_blocking=True)
        target_low = batch["target_low"].to(device, non_blocking=True)
        target_close = batch["target_close"].to(device, non_blocking=True)
        current_close = batch["current_close"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            result = run_multistep_rollout(
                features_seq=features,
                target_highs=target_high,
                target_lows=target_low,
                target_closes=target_close,
                current_closes=current_close,
                model=model,
                maker_fee=config["maker_fee"],
                initial_cash=config["initial_cash"],
                temperature=config["temperature"],
                max_positions=config["max_positions"],
                max_hold_days=config.get("max_hold_days", 14),
                use_grad_checkpoint=use_gc,
            )
            returns = result["returns"]

        loss, score, sortino, annual_ret = compute_loss_by_type(
            returns.float(),
            config["loss_type"],
            periods_per_year=DAILY_PERIODS_PER_YEAR_CRYPTO,
            return_weight=config["return_weight"],
        )

        _backward_step(loss, optimizer, model, config["grad_clip"], scaler)

        total_loss += loss.item()
        total_return += returns.mean().item()
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "mean_return": total_return / max(steps, 1),
    }


@torch.no_grad()
def eval_epoch(model, loader, device, config):
    model.eval()
    total_loss = 0.0
    total_return = 0.0
    total_sortino = 0.0
    steps = 0

    for batch in loader:
        features = batch["features"].to(device)
        target_high = batch["target_high"].to(device)
        target_low = batch["target_low"].to(device)
        target_close = batch["target_close"].to(device)
        current_close = batch["current_close"].to(device)
        ohlcv = batch["ohlcv"].to(device)

        sim, actions = run_sequential_sim(
            features_seq=features,
            ohlcv_seq=ohlcv,
            target_highs=target_high,
            target_lows=target_low,
            target_closes=target_close,
            current_closes=current_close,
            model=model,
            maker_fee=config["maker_fee"],
            initial_cash=config["initial_cash"],
            temperature=config["temperature"],
            max_positions=config["max_positions"],
        )

        returns = sim["returns"]
        loss, score, sortino, annual_ret = compute_loss_by_type(
            returns.unsqueeze(0),
            config["loss_type"],
            periods_per_year=DAILY_PERIODS_PER_YEAR_CRYPTO,
            return_weight=config["return_weight"],
        )

        total_loss += loss.item()
        total_return += returns.mean().item()
        total_sortino += sortino.mean().item()
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "mean_return": total_return / max(steps, 1),
        "sortino": total_sortino / max(steps, 1),
    }


@torch.no_grad()
def eval_epoch_multistep(model, loader, device, config):
    model.eval()
    total_loss = 0.0
    total_return = 0.0
    total_sortino = 0.0
    steps = 0

    for batch in loader:
        features = batch["features"].to(device)
        target_high = batch["target_high"].to(device)
        target_low = batch["target_low"].to(device)
        target_close = batch["target_close"].to(device)
        current_close = batch["current_close"].to(device)

        result = run_multistep_rollout(
            features_seq=features,
            target_highs=target_high,
            target_lows=target_low,
            target_closes=target_close,
            current_closes=current_close,
            model=model,
            maker_fee=config["maker_fee"],
            initial_cash=config["initial_cash"],
            temperature=config["temperature"],
            max_positions=config["max_positions"],
            max_hold_days=config.get("max_hold_days", 14),
        )

        returns = result["returns"]
        loss, score, sortino, annual_ret = compute_loss_by_type(
            returns,
            config["loss_type"],
            periods_per_year=DAILY_PERIODS_PER_YEAR_CRYPTO,
            return_weight=config["return_weight"],
        )

        total_loss += loss.item()
        total_return += returns.mean().item()
        total_sortino += sortino.mean().item()
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "mean_return": total_return / max(steps, 1),
        "sortino": total_sortino / max(steps, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train neural work-steal daily policy")
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-days", type=int, default=14)
    parser.add_argument("--loss-type", default="sortino", choices=["sortino", "sharpe", "calmar", "log_wealth", "sortino_dd"])
    parser.add_argument("--return-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("binance_worksteal/checkpoints"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--model-type", default="persymbol", choices=["flat", "persymbol"])
    parser.add_argument("--multistep", action="store_true")
    parser.add_argument("--rollout-len", type=int, default=10)
    parser.add_argument("--cosine-lr", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision (FP16)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Gradient checkpointing to save memory")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        logger.warning("AMP requested but no CUDA device, falling back to FP32")

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info("Loading data from %s", args.data_dir)

    if args.multistep:
        train_ds, val_ds, test_ds, loaded_symbols = build_sequential_datasets(
            data_dir=args.data_dir,
            symbols=args.symbols,
            seq_len=args.seq_len,
            rollout_len=args.rollout_len,
            test_days=args.test_days,
            val_days=args.val_days,
        )
        from binance_worksteal.data import build_datasets as _build_single
        _, val_ds_single, _, _ = _build_single(
            data_dir=args.data_dir,
            symbols=args.symbols,
            seq_len=args.seq_len,
            test_days=args.test_days,
            val_days=args.val_days,
        )
    else:
        train_ds, val_ds, test_ds, loaded_symbols = build_datasets(
            data_dir=args.data_dir,
            symbols=args.symbols,
            seq_len=args.seq_len,
            test_days=args.test_days,
            val_days=args.val_days,
        )

    logger.info("Loaded %d symbols, train=%d val=%d test=%d samples",
                len(loaded_symbols), len(train_ds), len(val_ds), len(test_ds))

    n_symbols = len(loaded_symbols)
    n_features = len(FEATURE_NAMES)

    if args.model_type == "persymbol":
        num_temporal = max(1, args.num_layers // 2)
        num_cross = max(1, args.num_layers - num_temporal)
        model = PerSymbolWorkStealPolicy(
            n_features=n_features,
            n_symbols=n_symbols,
            hidden_dim=args.hidden_dim,
            num_temporal_layers=num_temporal,
            num_cross_layers=num_cross,
            num_heads=args.num_heads,
            seq_len=args.seq_len,
            dropout=args.dropout,
        ).to(device)
    else:
        model = DailyWorkStealPolicy(
            n_features=n_features,
            n_symbols=n_symbols,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            seq_len=args.seq_len,
            dropout=args.dropout,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s, params: %d, amp=%s, grad_ckpt=%s",
                args.model_type, n_params, use_amp, args.grad_checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        logger.info("Using cosine LR schedule")

    dl_pin = device.type == "cuda"
    dl_workers = args.num_workers
    dl_prefetch = 2 if dl_workers > 0 else None
    train_loader = build_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=dl_workers, pin_memory=dl_pin, prefetch_factor=dl_prefetch,
    )
    val_loader = build_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=dl_workers, pin_memory=dl_pin, prefetch_factor=dl_prefetch,
    )
    if args.multistep:
        val_loader_single = build_dataloader(
            val_ds_single, batch_size=args.batch_size, shuffle=False,
            num_workers=dl_workers, pin_memory=dl_pin, prefetch_factor=dl_prefetch,
        )
    else:
        val_loader_single = val_loader

    run_name = args.run_name or time.strftime("neural_worksteal_%Y%m%d_%H%M%S")
    ckpt_dir = args.checkpoint_dir / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "maker_fee": args.maker_fee,
        "initial_cash": args.initial_cash,
        "temperature": args.temperature,
        "max_positions": args.max_positions,
        "max_hold_days": args.max_hold_days,
        "loss_type": args.loss_type,
        "return_weight": args.return_weight,
        "grad_clip": args.grad_clip,
        "grad_checkpoint": args.grad_checkpoint,
    }

    best_val_sortino = float("-inf")
    best_epoch = 0
    history = []

    train_fn = train_epoch_multistep if args.multistep else train_epoch

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_fn(model, train_loader, optimizer, device, config, scaler=scaler)
        val_metrics = eval_epoch(model, val_loader_single, device, config)
        elapsed = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d lr=%.2e %.1fs | Train loss=%.4f ret=%.4f | Val loss=%.4f ret=%.4f sort=%.4f",
            epoch, args.epochs, lr_now, elapsed,
            train_metrics["loss"], train_metrics["mean_return"],
            val_metrics["loss"], val_metrics["mean_return"], val_metrics["sortino"],
        )

        entry = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_metrics["loss"],
            "train_return": train_metrics["mean_return"],
            "val_loss": val_metrics["loss"],
            "val_return": val_metrics["mean_return"],
            "val_sortino": val_metrics["sortino"],
            "elapsed_s": elapsed,
        }
        history.append(entry)

        ckpt_data = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": val_metrics,
            "config": {
                "n_features": n_features,
                "n_symbols": n_symbols,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "seq_len": args.seq_len,
                "dropout": args.dropout,
                "symbols": loaded_symbols,
                "model_type": args.model_type,
                "multistep": args.multistep,
                "rollout_len": args.rollout_len,
            },
        }
        torch.save(ckpt_data, ckpt_dir / f"epoch_{epoch:03d}.pt")

        if val_metrics["sortino"] > best_val_sortino:
            best_val_sortino = val_metrics["sortino"]
            best_epoch = epoch
            torch.save(ckpt_data, ckpt_dir / "best.pt")

    meta = {
        "run_name": run_name,
        "symbols": loaded_symbols,
        "n_features": n_features,
        "n_symbols": n_symbols,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "model_type": args.model_type,
        "multistep": args.multistep,
        "rollout_len": args.rollout_len,
        "cosine_lr": args.cosine_lr,
        "amp": use_amp,
        "grad_checkpoint": args.grad_checkpoint,
        "best_epoch": best_epoch,
        "best_val_sortino": best_val_sortino,
        "history": history,
    }
    with open(ckpt_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Training complete. Best epoch=%d sortino=%.4f", best_epoch, best_val_sortino)
    logger.info("Checkpoints: %s", ckpt_dir)


if __name__ == "__main__":
    main()
