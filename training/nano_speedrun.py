#!/usr/bin/env python3
"""
Nanochat-style speedrun training loop for stock forecasting.

This script mirrors the fast defaults used in `karpathy/nanochat`:
  * unified optimizer factory (AdamW, Lion, Muon, etc.) via traininglib.make_optimizer
  * bf16 autocast + TF32 matmuls + Flash/SDPA attention through enable_fast_kernels
  * torch.compile with graceful fallback
  * cosine LR schedule with warmup measured in steps
  * markdown report summarising the run

The goal is to give the training/ directory a minimal, reproducible entry point
that experiments can reuse during benchmarking or CI smoke tests.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from traininglib import (
    enable_fast_kernels,
    bf16_supported,
    maybe_compile,
    make_optimizer,
    WarmupCosine,
    write_report_markdown,
)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


def load_price_matrix(data_root: Path, limit: int | None = None, max_rows: int | None = None) -> np.ndarray:
    """
    Load OHLC price data from CSV files under `data_root`.

    The loader favours `trainingdata/train/*.csv` (matching the existing HF scripts)
    and falls back to `trainingdata/*.csv`.  If neither exists we synthesise a
    random walk so the script remains runnable in CI.
    """
    candidates = []
    if (data_root / "train").exists():
        candidates.extend(sorted((data_root / "train").glob("*.csv")))
    candidates.extend(sorted(data_root.glob("*.csv")))
    if not candidates:
        return generate_synthetic_data(num_days=max_rows or 8192)

    rows: list[np.ndarray] = []
    for path in candidates[:limit] if limit else candidates:
        try:
            import pandas as pd

            df = pd.read_csv(path)
            cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
            if len(cols) < 4:
                continue
            arr = (
                df[cols]
                .apply(pd.to_numeric, errors="coerce")
                .ffill()
                .dropna()
                .to_numpy(dtype=np.float32)
            )
            if max_rows:
                arr = arr[:max_rows]
            if len(arr) > 0:
                rows.append(arr)
        except Exception:
            continue

    if not rows:
        return generate_synthetic_data(num_days=max_rows or 8192)

    return np.concatenate(rows, axis=0)


def generate_synthetic_data(num_days: int = 8192) -> np.ndarray:
    """Generate a simple geometric random walk as a fallback dataset."""
    rng = np.random.default_rng(1337)
    prices = [100.0]
    for _ in range(1, num_days):
        prices.append(prices[-1] * float(1 + rng.normal(0.0005, 0.02)))
    prices = np.array(prices, dtype=np.float32)

    highs = prices * (1 + rng.normal(0.01, 0.005, size=num_days))
    lows = prices * (1 - rng.normal(0.01, 0.005, size=num_days))
    opens = prices * (1 + rng.normal(0.0, 0.003, size=num_days))
    return np.stack([opens, highs, lows, prices], axis=1).astype(np.float32)


class SequenceDataset(Dataset):
    """Sliding-window dataset producing (context, horizon) pairs."""

    def __init__(self, matrix: np.ndarray, sequence_length: int, horizon: int):
        self.sequence_length = int(sequence_length)
        self.horizon = int(horizon)
        self.matrix = torch.from_numpy(matrix.astype(np.float32))

    def __len__(self) -> int:
        return max(0, self.matrix.size(0) - self.sequence_length - self.horizon + 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.matrix[idx : idx + self.sequence_length]
        target = self.matrix[idx + self.sequence_length : idx + self.sequence_length + self.horizon, -1]
        return {
            "inputs": window,
            "targets": target,
            "mask": torch.ones(self.sequence_length, dtype=torch.float32),
        }


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class PriceForecaster(nn.Module):
    """Simple transformer-style forecaster for demonstration purposes."""

    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.horizon = horizon
        self.embed = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embed(inputs)
        x = self.encoder(x)
        pooled = x[:, -1]
        return self.head(pooled)


# --------------------------------------------------------------------------------------
# Training utilities
# --------------------------------------------------------------------------------------


@dataclass
class SpeedrunConfig:
    data_dir: str = "trainingdata"
    output_dir: str = "runs/speedrun"
    report_path: str = "runs/speedrun/report.md"
    sequence_length: int = 64
    prediction_horizon: int = 8
    device_batch_size: int = 64
    grad_accum: int = 2
    epochs: int = 5
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    min_learning_rate: float = 0.0
    compile: bool = True
    seed: int = 1337
    max_training_rows: int | None = None
    max_symbols: int | None = 12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: SpeedrunConfig) -> Tuple[DataLoader, DataLoader, int]:
    matrix = load_price_matrix(Path(cfg.data_dir), limit=cfg.max_symbols, max_rows=cfg.max_training_rows)
    split = int(len(matrix) * 0.9)
    train_mat, val_mat = matrix[:split], matrix[split:]
    train_ds = SequenceDataset(train_mat, cfg.sequence_length, cfg.prediction_horizon)
    val_ds = SequenceDataset(val_mat, cfg.sequence_length, cfg.prediction_horizon)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.device_batch_size,
        shuffle=True,
        pin_memory=pin_mem,
        num_workers=4 if pin_mem else 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.device_batch_size,
        shuffle=False,
        pin_memory=pin_mem,
        num_workers=2 if pin_mem else 0,
    )
    return train_loader, val_loader, matrix.shape[1]


def train_speedrun(cfg: SpeedrunConfig) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, feature_dim = build_dataloaders(cfg)
    model = PriceForecaster(
        input_dim=feature_dim,
        hidden_dim=512,
        horizon=cfg.prediction_horizon,
    ).to(device)

    stack = contextlib.ExitStack()
    stack.enter_context(enable_fast_kernels())

    try:
        model = maybe_compile(model, do_compile=cfg.compile)
        optimizer = make_optimizer(
            model,
            name=cfg.optimizer,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        steps_per_epoch = math.ceil(len(train_loader) / max(1, cfg.grad_accum))
        total_steps = steps_per_epoch * cfg.epochs
        scheduler = WarmupCosine(
            optimizer,
            warmup_steps=cfg.warmup_steps,
            total_steps=max(total_steps, cfg.warmup_steps + 1),
            min_lr=cfg.min_learning_rate,
        )

        autocast_dtype = torch.bfloat16 if bf16_supported() else None
        report_metrics: Dict[str, float] = {}
        global_step = 0
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            epoch_loss = 0.0
            iter_start = time.time()
            for it, batch in enumerate(train_loader):
                inputs = batch["inputs"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)

                context = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
                with context:
                    pred = model(inputs)
                    loss = nn.functional.mse_loss(pred, targets)
                    loss = loss / cfg.grad_accum

                loss.backward()
                epoch_loss += float(loss.detach()) * cfg.grad_accum

                if (it + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
            steps_per_sec = global_step / max(1e-6, time.time() - iter_start)

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch in val_loader:
                    inputs = batch["inputs"].to(device, non_blocking=True)
                    targets = batch["targets"].to(device, non_blocking=True)
                    context = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
                    with context:
                        pred = model(inputs)
                        val_loss += float(nn.functional.mse_loss(pred, targets).detach())
                val_loss /= max(1, len(val_loader))

            report_metrics[f"epoch_{epoch}_train_loss"] = epoch_loss / max(1, len(train_loader))
            report_metrics[f"epoch_{epoch}_val_loss"] = val_loss
            report_metrics[f"epoch_{epoch}_steps_per_sec"] = steps_per_sec
            print(
                f"[epoch {epoch}] train_loss={report_metrics[f'epoch_{epoch}_train_loss']:.4f} "
                f"val_loss={val_loss:.4f} steps/s={steps_per_sec:.2f}"
            )

        args_dict = asdict(cfg)
        write_report_markdown(
            cfg.report_path,
            title="Nano Speedrun Training",
            args=args_dict,
            train_metrics=report_metrics,
            eval_metrics=None,
            notes=f"Finished in {cfg.epochs} epochs with optimizer '{cfg.optimizer}'.",
        )
        print(f"Report written to {cfg.report_path}")
    finally:
        stack.close()


def parse_args(argv: Iterable[str] | None = None) -> SpeedrunConfig:
    parser = argparse.ArgumentParser(description="Nanochat-style speedrun trainer for stock forecasts.")
    parser.add_argument("--data-dir", type=str, default="trainingdata")
    parser.add_argument("--output-dir", type=str, default="runs/speedrun")
    parser.add_argument("--report", type=str, default="runs/speedrun/report.md")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--device-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-training-rows", type=int, default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args(args=argv)

    return SpeedrunConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        sequence_length=args.sequence_length,
        prediction_horizon=args.horizon,
        device_batch_size=args.device_batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_learning_rate=args.min_lr,
        compile=args.compile and not args.no_compile,
        seed=args.seed,
        max_training_rows=args.max_training_rows,
        max_symbols=args.max_symbols,
    )


def main() -> None:
    cfg = parse_args()
    train_speedrun(cfg)


if __name__ == "__main__":
    main()
