#!/usr/bin/env python3
"""
Toto distillation baseline that keeps memory use in check for 3090-class GPUs.

The experiment runs a shallow feed-forward student model that learns to predict
future returns using Toto-enhanced features. It is intentionally lightweight so
multiple configs can be benchmarked side-by-side.
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

from hftraining.data_utils import StockDataProcessor
from .base import StrategyExperiment
from .registry import register


@dataclass
class PreparedDataset:
    train: TensorDataset
    val: TensorDataset
    input_dim: int


@register("toto_distillation")
class TotoDistillationExperiment(StrategyExperiment):
    """Lightweight student network for Toto-derived features."""

    def prepare_data(self) -> PreparedDataset:
        cfg = self.config.get("data", {})
        csv_path = Path(cfg.get("csv_path", "WIKI-AAPL.csv")).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV path '{csv_path}' does not exist")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()
        if "close" not in df.columns:
            raise ValueError("Dataframe must contain a 'close' column for targets")

        seq_len = int(cfg.get("sequence_length", 60))
        horizon = int(cfg.get("prediction_horizon", 5))

        processor = StockDataProcessor(
            sequence_length=seq_len,
            prediction_horizon=horizon,
            use_toto_forecasts=True,
        )
        features = processor.prepare_features(df)
        features = np.nan_to_num(features, copy=False)

        close = df["close"].astype(np.float32).to_numpy()
        future = np.roll(close, -horizon)
        target = (future - close) / (close + 1e-6)

        valid_length = len(target) - horizon
        features = features[:valid_length].astype(np.float32)
        target = target[:valid_length].astype(np.float32)

        splits = self._train_val_split(valid_length)
        train_x = torch.tensor(features[: splits["train"]])
        train_y = torch.tensor(target[: splits["train"]])
        val_x = torch.tensor(features[splits["train"] : splits["val"]])
        val_y = torch.tensor(target[splits["train"] : splits["val"]])

        train_ds = TensorDataset(train_x, train_y)
        val_ds = TensorDataset(val_x, val_y)

        return PreparedDataset(train=train_ds, val=val_ds, input_dim=train_x.shape[1])

    def build_model(
        self, dataset: PreparedDataset
    ) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
        model_cfg = self.config.get("model", {})
        hidden = int(model_cfg.get("hidden_size", 128))
        depth = int(model_cfg.get("num_layers", 2))
        dropout = float(model_cfg.get("dropout", 0.1))

        layers = []
        in_dim = dataset.input_dim
        for layer_idx in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))

        model = nn.Sequential(*layers)
        model = model.to(self.device)
        model = model.to(dtype=self.dtype)

        optim_cfg = self.config.get("training", {})
        lr = float(optim_cfg.get("learning_rate", 1e-3))
        weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        return model, optimizer, criterion

    def train_and_evaluate(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        dataset: PreparedDataset,
    ) -> Dict[str, float]:
        train_cfg = self.config.get("training", {})
        epochs = int(train_cfg.get("epochs", 3))
        batch_size = int(train_cfg.get("batch_size", 64))
        val_batch = int(train_cfg.get("val_batch_size", batch_size))

        train_loader = DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset.val, batch_size=val_batch, shuffle=False)

        scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp())

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                features, target = batch
                features = features.to(self.device, dtype=self.dtype)
                target = target.to(self.device, dtype=self.dtype).unsqueeze(-1)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self._use_amp(), dtype=self._amp_dtype()):
                    preds = model(features)
                    loss = criterion(preds.float(), target.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            avg_loss = running_loss / max(len(train_loader), 1)
            print(f"[Epoch {epoch+1}/{epochs}] train_mse={avg_loss:.6f}")

        metrics = self._evaluate(model, criterion, val_loader)
        return metrics

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
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
        directional_correct = 0
        total = 0
        with torch.no_grad():
            for features, target in loader:
                features = features.to(self.device, dtype=self.dtype)
                target = target.to(self.device, dtype=self.dtype).unsqueeze(-1)
                preds = model(features)
                mse_sum += criterion(preds.float(), target.float()).item() * len(target)
                mae_sum += torch.mean(torch.abs(preds.float() - target.float())).item() * len(
                    target
                )
                directional_correct += (
                    (torch.sign(preds) == torch.sign(target)).sum().item()
                )
                total += len(target)

        return {
            "val_mse": mse_sum / total if total else float("nan"),
            "val_mae": mae_sum / total if total else float("nan"),
            "directional_accuracy": directional_correct / total if total else float("nan"),
        }

    def _train_val_split(self, length: int) -> Dict[str, int]:
        train_ratio = float(self.config.get("data", {}).get("train_split", 0.7))
        val_ratio = float(self.config.get("data", {}).get("val_split", 0.15))
        train_end = int(length * train_ratio)
        val_end = int(length * (train_ratio + val_ratio))
        train_end = max(train_end, 1)
        val_end = min(max(val_end, train_end + 1), length)
        return {"train": train_end, "val": val_end}
