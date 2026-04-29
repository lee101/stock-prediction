"""Train the cross-attention transformer on the daily-stock panel.

Usage::

    .venv/bin/python -m models.cross_attn_train \\
        --panel analysis/cross_attn_transformer/panel_v1.npz \\
        --out analysis/cross_attn_transformer/v1_seed0.pt \\
        --seed 0 --epochs 10 --max-steps 0 --lr 3e-4

Trains one seed end-to-end with bf16 + AdamW + cosine schedule. The training
sampler picks one date per step (all active symbols on that day form the
batch) and computes BCE loss vs target_oc_up, masking invalid symbols.

Validation: the last 60 trading days of the training span are held out as a
validation set; validation loss + AUC printed each epoch. Early-stopping
patience is 3 epochs.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.cross_attn_data import fit_feature_stats
from models.cross_attn_transformer_v1 import CrossAttnConfig, CrossAttnTransformerV1

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Optional cap on total training steps (0 = unlimited within epochs)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-temporal", type=int, default=6)
    p.add_argument("--n-cross", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--val-days", type=int, default=60,
                   help="Number of trailing training days reserved as validation")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--max-train-secs", type=float, default=4 * 3600.0,
                   help="Wall-clock cap; training stops early if exceeded")
    return p.parse_args()


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def gather_window(X_norm_t: torch.Tensor, day_idx: int, seq_len: int) -> torch.Tensor:
    """Gather (S, T, F) tensor: window of last seq_len days ending at day_idx-1.

    The model predicts target on day_idx using features available through end
    of day_idx-1. Features at panel[day_idx, :, :] are already strict-no-
    lookahead (use D-1 close). To form a 256-day temporal context the model
    sees, we take panel[day_idx-seq_len+1 : day_idx+1, :, :] — the last entry
    is the row whose target we predict.
    """
    start = day_idx - seq_len + 1
    end = day_idx + 1                  # exclusive
    # X_norm_t is (D, S, F); transpose to (S, T, F)
    window = X_norm_t[start:end].transpose(0, 1).contiguous()
    return window


def evaluate_loss_and_auc(
    model: CrossAttnTransformerV1,
    X_norm_t: torch.Tensor,
    valid_t: torch.Tensor,
    y_t: torch.Tensor,
    day_indices: list[int],
    seq_len: int,
    device: torch.device,
) -> dict:
    model.eval()
    losses: list[float] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for d in day_indices:
            mask = valid_t[d]
            if mask.sum().item() < 5:
                continue
            window = gather_window(X_norm_t, d, seq_len)         # (S, T, F)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(window, valid_mask=mask)           # (S,)
            sel = mask
            sel_logits = logits[sel].float()
            sel_labels = y_t[d][sel].float()
            loss = F.binary_cross_entropy_with_logits(sel_logits, sel_labels)
            losses.append(loss.item())
            all_logits.append(sel_logits.detach().cpu().numpy())
            all_labels.append(sel_labels.detach().cpu().numpy())
    if not losses:
        return {"loss": float("nan"), "auc": float("nan"), "n_days": 0}
    flat_l = np.concatenate(all_logits)
    flat_y = np.concatenate(all_labels)
    auc = _binary_auc(flat_l, flat_y)
    return {
        "loss": float(np.mean(losses)),
        "auc": float(auc),
        "n_days": len(day_indices),
    }


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    sum_pos_ranks = float(ranks[pos].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("loading panel %s", args.panel)
    panel = np.load(args.panel, allow_pickle=True)
    X = panel["X"]
    valid = panel["valid"]
    y = panel["y"]
    dates = panel["dates"]
    train_end = np.datetime64(str(panel["train_end"]))
    train_end_idx = int((dates <= train_end).sum())   # exclusive boundary
    logger.info("panel shape D=%d S=%d F=%d  train_end_idx=%d",
                X.shape[0], X.shape[1], X.shape[2], train_end_idx)

    # Compute z-score stats on training data only (robust to outliers).
    mean, std = fit_feature_stats(X, valid, train_end_idx)
    logger.info("feature mean[0..3]=%s std[0..3]=%s", mean[:3], std[:3])

    # Normalize once. Then clip post-norm to ±5σ to neutralise outlier rows
    # from stock splits / data glitches. The clip is symmetric since the
    # robust stats already centred each feature near 0 within typical range.
    X_norm = ((X - mean) / std).astype(np.float32)
    np.clip(X_norm, -5.0, 5.0, out=X_norm)
    # Force invalid rows to zero so they don't contribute anything via attention
    X_norm[~valid] = 0.0

    device = torch.device("cuda")
    X_norm_t = torch.from_numpy(X_norm).to(device)         # (D, S, F)
    valid_t = torch.from_numpy(valid).to(device)
    y_t = torch.from_numpy(y).to(device)

    # Train / val split: hold out the last `val_days` of training.
    val_start = max(args.seq_len, train_end_idx - args.val_days)
    val_idx_range = list(range(val_start, train_end_idx))
    train_idx_range = list(range(args.seq_len, val_start))
    logger.info("train days=%d val days=%d", len(train_idx_range), len(val_idx_range))

    cfg = CrossAttnConfig(
        n_features=X.shape[2],
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_temporal_layers=args.n_temporal,
        n_cross_layers=args.n_cross,
        n_heads=args.n_heads,
        grad_checkpoint=not args.no_grad_checkpoint,
    )
    model = CrossAttnTransformerV1(cfg).to(device)
    logger.info("model params: %.2fM", model.num_params() / 1e6)

    # bf16 weights for speed/memory; AdamW master in fp32
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    n_steps_per_epoch = len(train_idx_range)
    total_steps = n_steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    logger.info("scheduling %d total steps", total_steps)

    rng = np.random.default_rng(args.seed)
    best_val = math.inf
    bad_epochs = 0
    history: list[dict] = []
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    global_step = 0
    stopped = False
    for epoch in range(args.epochs):
        if stopped:
            break
        model.train()
        perm = rng.permutation(train_idx_range)
        epoch_losses: list[float] = []
        t_ep = time.time()
        for d in perm:
            if global_step >= total_steps:
                stopped = True
                break
            if time.time() - t_start > args.max_train_secs:
                logger.info("max_train_secs reached")
                stopped = True
                break
            mask = valid_t[int(d)]
            if mask.sum().item() < 5:
                continue
            window = gather_window(X_norm_t, int(d), args.seq_len)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(window, valid_mask=mask)
                sel = mask
                sel_logits = logits[sel].float()
                sel_labels = y_t[int(d)][sel].float()
                if args.label_smoothing > 0:
                    eps = args.label_smoothing
                    sel_labels = sel_labels * (1 - eps) + 0.5 * eps
                loss = F.binary_cross_entropy_with_logits(sel_logits, sel_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            cur_lr = cosine_lr(global_step, total_steps, args.lr, args.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            opt.step()
            epoch_losses.append(loss.item())
            global_step += 1
            if global_step % 100 == 0:
                logger.info(
                    "ep=%d step=%d/%d lr=%.2e loss=%.4f mem=%.1fGB elapsed=%.0fs",
                    epoch, global_step, total_steps, cur_lr,
                    float(np.mean(epoch_losses[-100:])),
                    torch.cuda.max_memory_allocated() / 1e9,
                    time.time() - t_start,
                )

        # Evaluate validation
        val_metrics = evaluate_loss_and_auc(
            model, X_norm_t, valid_t, y_t, val_idx_range, args.seq_len, device,
        )
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        elapsed = time.time() - t_start
        logger.info(
            "EPOCH %d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  elapsed=%.0fs",
            epoch, train_loss, val_metrics["loss"], val_metrics["auc"], elapsed,
        )
        history.append(dict(
            epoch=epoch, train_loss=train_loss,
            val_loss=val_metrics["loss"], val_auc=val_metrics["auc"],
            elapsed_secs=elapsed,
        ))
        improved = val_metrics["loss"] < best_val - 1e-5
        if improved:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": asdict(cfg),
                "feature_mean": mean,
                "feature_std": std,
                "feature_cols": panel["feature_cols"],
                "train_end_idx": int(train_end_idx),
                "best_val_loss": float(best_val),
                "best_val_auc": float(val_metrics["auc"]),
                "epoch": int(epoch),
                "args": vars(args),
                "history": history,
            }, args.out)
            logger.info("saved checkpoint to %s (val_loss=%.4f)", args.out, best_val)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info("early stopping after %d non-improving epochs", bad_epochs)
                stopped = True

    # Save final history
    history_path = args.out.with_suffix(".history.json")
    history_path.write_text(json.dumps({
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "history": history,
        "best_val_loss": float(best_val) if best_val < math.inf else None,
        "elapsed_secs": time.time() - t_start,
    }, indent=2))
    logger.info("training done; history saved to %s", history_path)


if __name__ == "__main__":
    main()
