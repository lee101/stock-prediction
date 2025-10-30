from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:  # PyTorch >= 2.1
    from torch.amp import GradScaler as TorchGradScaler  # type: ignore[attr-defined]
    from torch.amp import autocast as torch_autocast  # type: ignore[attr-defined]
    _AMP_SUPPORTS_DEVICE = True
except ImportError:  # pragma: no cover - compatibility path
    from torch.cuda.amp import GradScaler as TorchGradScaler  # type: ignore
    from torch.cuda.amp import autocast as torch_autocast  # type: ignore
    _AMP_SUPPORTS_DEVICE = False

from .config import KronosTrainingConfig
from .data_utils import ALL_FEATURES, iter_symbol_dataframes
from .dataset import KronosMultiTickerDataset
from src.parameter_efficient import (
    LoraMetadata,
    freeze_module_parameters,
    inject_lora_adapters,
    save_lora_adapter,
)
from traininglib.dynamic_batcher import WindowBatcher

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

        self._configure_cuda()
        self._amp_dtype = self._resolve_amp_dtype(self.config.precision)
        self._amp_enabled = self._amp_dtype is not None and self.device.type == "cuda"
        self.scaler = TorchGradScaler(enabled=self._amp_enabled and self.config.precision == "fp16")
        self._compile_enabled = bool(self.config.torch_compile)
        self._compiled_steps: Dict[Tuple[int, int, int], Callable[..., torch.Tensor]] = {}
        self._bucket_warmup_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self._warmup_target = max(0, self.config.bucket_warmup_steps)
        self._train_dataset: Optional[KronosMultiTickerDataset] = None
        self._val_dataset: Optional[KronosMultiTickerDataset] = None

        print(f"[kronos] Using device: {self.device}")

        self.tokenizer = KronosTokenizer.from_pretrained(self.config.tokenizer_name)
        self.model = Kronos.from_pretrained(self.config.model_name)
        self.tokenizer.to(self.device)
        self.model.to(self.device)
        self._adapter_metadata: LoraMetadata | None = None
        self._adapter_targets: List[str] = []

        if self.config.adapter_type == "lora":
            if self.config.freeze_backbone:
                freeze_module_parameters(self.model)
            replacements = inject_lora_adapters(
                self.model,
                target_patterns=self.config.adapter_targets,
                rank=self.config.adapter_rank,
                alpha=self.config.adapter_alpha,
                dropout=self.config.adapter_dropout,
            )
            # Newly injected LoRA weights are created on the default device; ensure they follow the model device.
            self.model.to(self.device)
            self._adapter_targets = replacements
            if self.config.freeze_backbone:
                trainable = [p for p in self.model.parameters() if p.requires_grad]
            else:
                trainable = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable:
                raise RuntimeError("LoRA injection resulted in no trainable parameters.")
            self._adapter_metadata = LoraMetadata(
                adapter_type="lora",
                rank=self.config.adapter_rank,
                alpha=self.config.adapter_alpha,
                dropout=self.config.adapter_dropout,
                targets=replacements,
                base_model=self.config.model_name,
            )
            print(f"[kronos] Enabled LoRA adapters on {len(replacements)} modules.")

    def _resolve_amp_dtype(self, precision: str) -> Optional[torch.dtype]:
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        return None

    def _configure_cuda(self) -> None:
        if not torch.cuda.is_available():
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            from torch.backends.cuda import sdp_kernel

            sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        except (ImportError, AttributeError):  # pragma: no cover - optional acceleration
            pass

    def _autocast(self):
        if not self._amp_enabled or self._amp_dtype is None:
            return nullcontext()
        if _AMP_SUPPORTS_DEVICE:
            return torch_autocast(self.device.type, dtype=self._amp_dtype, enabled=True)
        return torch_autocast(dtype=self._amp_dtype, enabled=True)

    def _build_optimizer(self, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        fused = self.config.use_fused_optimizer and self.device.type == "cuda"
        try:
            optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
                fused=fused,
            )
            if fused:
                print("[kronos] Using fused AdamW optimizer.")
            return optimizer
        except TypeError:
            if fused:
                print("[kronos] Fused AdamW unavailable; falling back to standard AdamW.")
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )

    def _build_step_function(self) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        def step_fn(
            token_in0: torch.Tensor,
            token_in1: torch.Tensor,
            stamp_in: torch.Tensor,
            token_out0: torch.Tensor,
            token_out1: torch.Tensor,
        ) -> torch.Tensor:
            logits = self.model(token_in0, token_in1, stamp_in)
            loss, _, _ = self.model.head.compute_loss(logits[0], logits[1], token_out0, token_out1)
            return loss

        if not self._compile_enabled or not hasattr(torch, "compile"):
            return step_fn
        try:
            compiled = torch.compile(step_fn, fullgraph=True, mode=self.config.compile_mode)
            return compiled
        except Exception as exc:  # pragma: no cover - compilation fallback
            print(f"[kronos] torch.compile failed ({exc}); continuing in eager mode.")
            self._compile_enabled = False
            return step_fn

    def _build_data_interfaces(self) -> Tuple[WindowBatcher, DataLoader]:
        self._train_dataset = KronosMultiTickerDataset(
            data_dir=str(self.config.data_dir),
            split="train",
            lookback=self.config.lookback_window,
            prediction_length=self.config.prediction_length,
            validation_days=self.config.validation_days,
            clip=self.config.clip_value,
            min_symbol_length=self.config.min_symbol_length,
        )
        self._val_dataset = KronosMultiTickerDataset(
            data_dir=str(self.config.data_dir),
            split="val",
            lookback=self.config.lookback_window,
            prediction_length=self.config.prediction_length,
            validation_days=self.config.validation_days,
            clip=self.config.clip_value,
            min_symbol_length=self.config.min_symbol_length,
        )

        train_batcher = WindowBatcher(
            self._train_dataset,
            max_tokens_per_batch=self.config.max_tokens_per_batch,
            context_buckets=self.config.length_buckets,
            horizon_buckets=self.config.horizon_buckets,
            stride=self.config.window_stride,
            pack_windows=self.config.pack_windows,
        )
        print(
            "[kronos] Train windows: "
            f"{len(train_batcher)} across {len(self._train_dataset.series_ids)} series, "
            f"Val samples: {len(self._val_dataset)}"
        )

        val_loader = DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )
        return train_batcher, val_loader

    def train(self) -> Dict[str, float]:
        train_batcher, val_loader = self._build_data_interfaces()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters detected; check adapter/freezing configuration.")
        optimizer = self._build_optimizer(trainable_params)

        best_val_loss = float("inf")
        best_epoch = -1

        total_steps = 0
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_loss, epoch_steps = self._train_one_epoch(train_batcher, optimizer, epoch)
            val_loss = self._evaluate(val_loader)
            total_steps += epoch_steps

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
                self._save_adapter()

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

    def _train_one_epoch(
        self,
        batcher: WindowBatcher,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Tuple[float, int]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        grad_accum = self.config.grad_accum_steps
        accum_counter = 0
        optim_steps = 0
        log_interval = max(1, self.config.log_interval)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer.zero_grad(set_to_none=True)

        for step_idx, window_batch in enumerate(batcher, start=1):
            batch_x, batch_stamp = window_batch.batch
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_stamp = batch_stamp.to(self.device, non_blocking=True)

            token_seq_0, token_seq_1 = self.tokenizer.encode(batch_x, half=True)
            token_in0, token_in1 = token_seq_0[:, :-1], token_seq_1[:, :-1]
            token_out0, token_out1 = token_seq_0[:, 1:], token_seq_1[:, 1:]
            stamp_in = batch_stamp[:, :-1, :]

            compile_key = (window_batch.context, window_batch.horizon, batch_x.shape[0])
            step_fn = self._compiled_steps.get(compile_key)
            if step_fn is None:
                step_fn = self._build_step_function()
                self._compiled_steps[compile_key] = step_fn

            bucket_key = (window_batch.context, window_batch.horizon)
            warmups_done = self._bucket_warmup_counts.get(bucket_key, 0)
            if warmups_done < self._warmup_target:
                with torch.no_grad():
                    with self._autocast():
                        step_fn(token_in0, token_in1, stamp_in, token_out0, token_out1)
                self._bucket_warmup_counts[bucket_key] = warmups_done + 1

            with self._autocast():
                loss = step_fn(token_in0, token_in1, stamp_in, token_out0, token_out1)

            loss_value = float(loss.detach().item())
            total_loss += loss_value * window_batch.size
            total_samples += window_batch.size

            loss_for_backward = loss / grad_accum
            if self.scaler.is_enabled():
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            accum_counter += 1
            if accum_counter == grad_accum:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, self.config.grad_clip_norm)
                if self.scaler.is_enabled():
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                optim_steps += 1

            if step_idx % log_interval == 0:
                avg_loss = total_loss / max(total_samples, 1)
                print(
                    f"[kronos] Epoch {epoch} step {step_idx} "
                    f"ctx={window_batch.context} hor={window_batch.horizon} "
                    f"loss={loss_value:.4f} avg={avg_loss:.4f}"
                )

        if accum_counter > 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.grad_clip_norm)
            if self.scaler.is_enabled():
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optim_steps += 1

        avg_epoch_loss = total_loss / max(total_samples, 1)
        return avg_epoch_loss, optim_steps

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

    def _save_adapter(self) -> None:
        if self._adapter_metadata is None or self.config.adapter_type != "lora":
            return
        try:
            save_lora_adapter(self.model, self.config.adapter_file, metadata=self._adapter_metadata)
            print(f"[kronos] Saved LoRA adapter to {self.config.adapter_file}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[kronos] Adapter save failed: {exc}")

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
