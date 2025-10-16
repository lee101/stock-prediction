from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import KronosTrainingConfig
from .data_utils import ALL_FEATURES, iter_symbol_dataframes
from .dataset import KronosMultiTickerDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_KRONOS = REPO_ROOT / "external" / "kronos"
if str(EXTERNAL_KRONOS) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_KRONOS))

from external.kronos.model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    _ = rng.random()  # ensure generator initialised


class KronosTrainer:
    def __init__(self, config: KronosTrainingConfig) -> None:
        self.config = config
        self.config.ensure_output_dirs()
        self.device = torch.device(self.config.resolved_device())
        set_seed(self.config.seed)

        print(f"[kronos] Using device: {self.device}")

        self.tokenizer = KronosTokenizer.from_pretrained(self.config.tokenizer_name)
        self.model = Kronos.from_pretrained(self.config.model_name)
        self.tokenizer.to(self.device)
        self.model.to(self.device)

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = KronosMultiTickerDataset(
            data_dir=str(self.config.data_dir),
            split="train",
            lookback=self.config.lookback_window,
            prediction_length=self.config.prediction_length,
            validation_days=self.config.validation_days,
            clip=self.config.clip_value,
            min_symbol_length=self.config.min_symbol_length,
        )
        val_dataset = KronosMultiTickerDataset(
            data_dir=str(self.config.data_dir),
            split="val",
            lookback=self.config.lookback_window,
            prediction_length=self.config.prediction_length,
            validation_days=self.config.validation_days,
            clip=self.config.clip_value,
            min_symbol_length=self.config.min_symbol_length,
        )

        print(f"[kronos] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=len(train_dataset) > self.config.batch_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )
        return train_loader, val_loader

    def train(self) -> Dict[str, float]:
        train_loader, val_loader = self._build_dataloaders()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )

        best_val_loss = float("inf")
        best_epoch = -1

        total_steps = 0
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = self._train_one_epoch(train_loader, optimizer, epoch)
            val_loss = self._evaluate(val_loader)
            total_steps += len(train_loader)

            epoch_msg = (
                f"[kronos] Epoch {epoch}/{self.config.epochs} - "
                f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}"
            )
            print(epoch_msg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self._save_checkpoint(self.config.best_model_path)
                print(f"[kronos] Saved new best model at {self.config.best_model_path}")

            self._save_checkpoint(self.config.last_model_path)

        duration_min = (time.time() - start_time) / 60.0
        print(f"[kronos] Training finished in {duration_min:.2f} minutes. Best epoch: {best_epoch}")

        # Reload best weights for downstream evaluation
        if self.config.best_model_path.exists():
            self.model = Kronos.from_pretrained(str(self.config.best_model_path))
            self.model.to(self.device)

        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs": self.config.epochs,
            "steps": total_steps,
            "training_minutes": duration_min,
        }

    def _train_one_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        step_count = 0

        for step, (batch_x, batch_stamp) in enumerate(loader, start=1):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_stamp = batch_stamp.to(self.device, non_blocking=True)

            with torch.no_grad():
                token_seq_0, token_seq_1 = self.tokenizer.encode(batch_x, half=True)

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            logits = self.model(token_in[0], token_in[1], batch_stamp[:, :-1, :])
            loss, _, _ = self.model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

            if step % self.config.log_interval == 0:
                print(f"[kronos] Epoch {epoch} step {step}/{len(loader)} loss={loss.item():.4f}")

        return total_loss / max(step_count, 1)

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch_x, batch_stamp in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_stamp = batch_stamp.to(self.device, non_blocking=True)
                token_seq_0, token_seq_1 = self.tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = self.model(token_in[0], token_in[1], batch_stamp[:, :-1, :])
                val_loss, _, _ = self.model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                losses.append(val_loss.item())
        if not losses:
            return float("nan")
        return float(np.mean(losses))

    def _save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))

    def evaluate_holdout(self) -> Dict[str, object]:
        self.model.eval()
        max_context = min(512, self.config.lookback_window + self.config.prediction_length)
        predictor = KronosPredictor(
            self.model, self.tokenizer, device=str(self.device), max_context=max_context, clip=self.config.clip_value
        )

        per_symbol: List[Dict[str, float]] = []
        for symbol, df in iter_symbol_dataframes(self.config.data_dir):
            total_len = len(df)
            needed = self.config.lookback_window + self.config.validation_days
            if total_len < needed:
                continue

            context_start = total_len - needed
            history = df.iloc[context_start : context_start + self.config.lookback_window].copy()
            future = df.iloc[-self.config.validation_days :].copy()

            # Prepare inputs
            hist_df = history[list(ALL_FEATURES)].astype(np.float32)
            x_timestamp = history["timestamps"]
            y_timestamp = future["timestamps"]

            try:
                pred_df = predictor.predict(
                    df=hist_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=self.config.validation_days,
                    T=self.config.eval_temperature,
                    top_p=self.config.eval_top_p,
                    sample_count=self.config.eval_sample_count,
                    verbose=False,
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[kronos] Evaluation skipped for {symbol}: {exc}")
                continue

            actual_close = future["close"].to_numpy(dtype=np.float64)
            pred_close = pred_df["close"].to_numpy(dtype=np.float64)
            error = pred_close - actual_close

            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(np.mean(error ** 2)))
            mape = float(np.mean(np.abs(error) / (np.abs(actual_close) + 1e-5)) * 100.0)

            per_symbol.append(
                {
                    "symbol": symbol,
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                }
            )

        if not per_symbol:
            raise RuntimeError("No symbols produced evaluation metrics; ensure validation window is valid.")

        aggregate = {
            "symbols_evaluated": len(per_symbol),
            "mae": float(np.mean([m["mae"] for m in per_symbol])),
            "rmse": float(np.mean([m["rmse"] for m in per_symbol])),
            "mape": float(np.mean([m["mape"] for m in per_symbol])),
        }

        metrics = {"aggregate": aggregate, "per_symbol": per_symbol}
        with open(self.config.metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("[kronos] Validation metrics:")
        print(json.dumps(aggregate, indent=2))
        return metrics
