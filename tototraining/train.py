#!/usr/bin/env python3
"""
Fine-tune the Toto foundation model on local price series with efficiency tweaks
suited for the RTX 3090 workstation.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toto.inference.forecaster import TotoForecaster  # noqa: E402
from toto.model.toto import Toto  # noqa: E402

from tototraining.data import WindowConfig, build_dataloaders  # noqa: E402


def _bool_flag(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "1"}:
        return True
    if lowered in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean flag: {value}")


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True, help="Directory or file with training series.")
    parser.add_argument("--val-root", type=Path, default=None, help="Optional directory/file for validation series.")
    parser.add_argument("--context-length", type=int, default=4096, help="Number of past steps provided to the model.")
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=64,
        help="Number of future steps to predict (should align with patch size).",
    )
    parser.add_argument("--stride", type=int, default=64, help="Sliding window stride when building datasets.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", type=_bool_flag, default=True)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--output-dir", type=Path, default=Path("tototraining/checkpoints"))
    parser.add_argument("--checkpoint-name", default="toto-open-base-finetuned")
    parser.add_argument("--num-workers", type=int, default=max(os.cpu_count() - 2, 2))
    return parser


def _create_masks(series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padding_mask = torch.ones_like(series, dtype=torch.bool)
    id_mask = torch.zeros_like(series, dtype=torch.int)
    return padding_mask, id_mask


def _save_model(model: Toto, output_dir: Path, checkpoint_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / checkpoint_name
    model.save_pretrained(save_path)


def train() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    device = torch.device(args.device)

    window_cfg = WindowConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
    )
    train_loader, val_loader = build_dataloaders(
        args.train_root,
        args.val_root,
        window_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
    model.to(device)

    if args.compile and hasattr(model, "compile"):
        model.compile(mode=args.compile_mode)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_loss = math.inf
    best_epoch = -1
    def forward_pass(context: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        series = torch.cat([context, target], dim=-1)
        padding_mask, id_mask = _create_masks(series)
        base_distr, loc, scale = model.model(
            inputs=series,
            input_padding_mask=padding_mask,
            id_mask=id_mask,
            kv_cache=None,
            scaling_prefix_length=context.shape[-1],
        )
        distr = TotoForecaster.create_affine_transformed(base_distr, loc, scale)
        return distr, series

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()

        for step, (context, target) in enumerate(train_loader, start=1):
            context = context.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                distr, series = forward_pass(context, target)
                log_probs = distr.log_prob(series)
                target_log_probs = log_probs[:, :, -args.prediction_length :]
                loss = -(target_log_probs.mean()) / args.grad_accum

            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                if args.clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * args.grad_accum
            step_count += 1

        train_time = time.time() - start_time
        avg_train_loss = epoch_loss / max(step_count, 1)
        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.6f} time={train_time:.1f}s compiled={args.compile}")

        if val_loader is None:
            continue

        model.eval()
        val_loss = 0.0
        val_mape = 0.0
        val_steps = 0
        with torch.no_grad():
            for context, target in val_loader:
                context = context.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.float32)
                distr, series = forward_pass(context, target)
                log_probs = distr.log_prob(series)
                target_log_probs = log_probs[:, :, -args.prediction_length :]
                batch_loss = -target_log_probs.mean()
                val_loss += batch_loss.item()
                mean_forecast = distr.mean[:, :, -args.prediction_length :]
                ape = torch.abs(mean_forecast - target) / (torch.abs(target) + 1e-6)
                val_mape += ape.mean().item() * 100.0
                val_steps += 1

        val_loss /= max(val_steps, 1)
        val_mape /= max(val_steps, 1)
        print(f"[Epoch {epoch}] val_loss={val_loss:.6f} val_mape={val_mape:.3f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            _save_model(model, args.output_dir, args.checkpoint_name)

    if best_epoch > 0:
        print(f"Best validation loss {best_val_loss:.6f} achieved at epoch {best_epoch}.")
    else:
        _save_model(model, args.output_dir, args.checkpoint_name)


if __name__ == "__main__":
    train()
