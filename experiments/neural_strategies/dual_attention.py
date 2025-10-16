#!/usr/bin/env python3
"""
Prototype dual-attention experiment.

This approximates a lightweight dual-attention architecture by combining an
input projection with a transformer encoder. The goal is to benchmark sequence
models under bf16 compute without requiring a full-blown order-book simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from hftraining.data_utils import StockDataProcessor
from .base import StrategyExperiment
from .registry import register


@dataclass
class SequenceDataset:
    train: TensorDataset
    val: TensorDataset
    input_dim: int
    context_length: int


class DualAttentionModel(nn.Module):
    """Minimal transformer-style model with optional checkpointing."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        x = self.input_proj(x)
        if use_checkpoint:
            for layer in self.encoder.layers:
                x = gradient_checkpoint(layer, x)
            if self.encoder.norm is not None:
                x = self.encoder.norm(x)
        else:
            x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


@register("dual_attention_prototype")
class DualAttentionPrototype(StrategyExperiment):
    """Sequence model harness built for GPU benchmarking."""

    def prepare_data(self) -> SequenceDataset:
        cfg = self.config.get("data", {})
        csv_path = Path(cfg.get("csv_path", "WIKI-AAPL.csv")).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV path '{csv_path}' does not exist")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()

        context = int(cfg.get("context_length", 32))
        horizon = int(cfg.get("prediction_horizon", 5))

        processor = StockDataProcessor(
            sequence_length=context,
            prediction_horizon=horizon,
            use_toto_forecasts=True,
        )
        features = processor.prepare_features(df)
        features = np.nan_to_num(features, copy=False)

        close = df["close"].astype(np.float32).to_numpy()
        future = np.roll(close, -horizon)
        target = (future - close) / (close + 1e-6)

        valid_length = len(features) - context - horizon
        if valid_length <= 0:
            raise ValueError("Not enough data to create sequences; reduce context length.")

        seqs = []
        labels = []
        for i in range(valid_length):
            start = i
            end = i + context
            seqs.append(features[start:end])
            labels.append(target[end - 1])

        seqs = np.stack(seqs).astype(np.float32)
        labels = np.array(labels, dtype=np.float32)

        splits = self._train_val_split(len(seqs))
        train_x = torch.tensor(seqs[: splits["train"]])
        train_y = torch.tensor(labels[: splits["train"]])
        val_x = torch.tensor(seqs[splits["train"] : splits["val"]])
        val_y = torch.tensor(labels[splits["train"] : splits["val"]])

        train_ds = TensorDataset(train_x, train_y)
        val_ds = TensorDataset(val_x, val_y)
        return SequenceDataset(
            train=train_ds,
            val=val_ds,
            input_dim=train_x.shape[-1],
            context_length=context,
        )

    def build_model(
        self, dataset: SequenceDataset
    ) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
        model_cfg = self.config.get("model", {})
        embed_dim = int(model_cfg.get("embed_dim", 128))
        num_heads = int(model_cfg.get("num_heads", 4))
        num_layers = int(model_cfg.get("num_layers", 2))
        dropout = float(model_cfg.get("dropout", 0.1))

        model = DualAttentionModel(
            input_dim=dataset.input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        model = model.to(self.device, dtype=self.dtype)

        train_cfg = self.config.get("training", {})
        lr = float(train_cfg.get("learning_rate", 2e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.SmoothL1Loss()
        return model, optimizer, criterion

    def train_and_evaluate(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        dataset: SequenceDataset,
    ) -> Dict[str, float]:
        train_cfg = self.config.get("training", {})
        epochs = int(train_cfg.get("epochs", 4))
        batch_size = int(train_cfg.get("batch_size", 32))
        val_batch = int(train_cfg.get("val_batch_size", batch_size))

        train_loader = DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset.val, batch_size=val_batch, shuffle=False)

        scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp())

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for seqs, labels in train_loader:
                seqs = seqs.to(self.device, dtype=self.dtype)
                labels = labels.to(self.device, dtype=self.dtype).unsqueeze(-1)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self._use_amp(), dtype=self._amp_dtype()):
                    preds = model(seqs, use_checkpoint=self.gradient_checkpointing)
                    loss = criterion(preds.float(), labels.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            print(
                f"[Epoch {epoch+1}/{epochs}] train_loss={total_loss / max(len(train_loader),1):.6f}"
            )

        return self._evaluate(model, criterion, val_loader)

    # ------------------------------------------------------------------ #
    def _use_amp(self) -> bool:
        return self.device.type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}

    def _amp_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == torch.bfloat16 else torch.float16

    def _evaluate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        loader: DataLoader,
    ) -> Dict[str, float]:
        model.eval()
        mse_sum = 0.0
        mae_sum = 0.0
        win_sum = 0
        total = 0
        with torch.inference_mode():
            for seqs, labels in loader:
                seqs = seqs.to(self.device, dtype=self.dtype)
                labels = labels.to(self.device, dtype=self.dtype).unsqueeze(-1)
                preds = model(seqs, use_checkpoint=False)
                mse_sum += torch.mean((preds.float() - labels.float()) ** 2).item() * len(labels)
                mae_sum += torch.mean(torch.abs(preds.float() - labels.float())).item() * len(
                    labels
                )
                win_sum += (torch.sign(preds) == torch.sign(labels)).sum().item()
                total += len(labels)
        return {
            "val_mse": mse_sum / total if total else float("nan"),
            "val_mae": mae_sum / total if total else float("nan"),
            "directional_accuracy": win_sum / total if total else float("nan"),
        }

    def _train_val_split(self, length: int) -> Dict[str, int]:
        train_ratio = float(self.config.get("data", {}).get("train_split", 0.7))
        val_ratio = float(self.config.get("data", {}).get("val_split", 0.15))
        train_end = int(length * train_ratio)
        val_end = int(length * (train_ratio + val_ratio))
        train_end = max(train_end, 1)
        val_end = min(max(val_end, train_end + 1), length)
        return {"train": train_end, "val": val_end}
