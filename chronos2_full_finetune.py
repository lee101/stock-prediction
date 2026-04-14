#!/usr/bin/env python3
"""
Full Chronos2 fine-tuning on ALL available stock/crypto data.

Trains a single shared model on all symbols jointly (daily stocks + hourly crypto),
with online data augmentation. This is better than per-symbol LoRAs because:
  - Cross-symbol patterns are learned (market regimes, sector correlations)
  - Much more total training data (328 stocks × 700 days + 206 hourly × 50k bars)
  - Online augmentation (amplitude jitter, noise, time-dropout) prevents overfitting

After training, the finetuned model is evaluated on the held-out validation windows
and optionally saved as the new base for per-symbol LoRA training.

Usage:
    source .venv/bin/activate
    python chronos2_full_finetune.py \\
        --output-dir chronos2_finetuned/stocks_all_v1 \\
        --num-steps 50000 \\
        --batch-size 512 \\
        --learning-rate 5e-5 \\
        --context-length 512

Arguments (all optional, see --help):
    --daily-data-dir    Daily stock CSVs (default: trainingdata/)
    --hourly-data-dirs  Comma-separated hourly dirs (default: binance_spot_hourly)
    --output-dir        Where to save finetuned checkpoint
    --model-id          Base model to start from (default: amazon/chronos-2)
    --finetune-mode     full | lora (default: full)
    --lora-r            LoRA rank if finetune-mode=lora (default: 32)
    --num-steps         Training steps (default: 30000)
    --batch-size        Training batch size (default: 256)
    --learning-rate     Peak LR (default: 5e-5 for full, 1e-4 for lora)
    --context-length    Context window (default: 512)
    --prediction-length Forecast horizon (default: 1)
    --val-bars          Hold-out bars per series for validation (default: 60)
    --amp-log-std       Amplitude jitter log-std (default: 0.30)
    --noise-frac        Relative noise std fraction (default: 0.002)
    --dropout-rate      Time dropout rate (default: 0.02)
    --no-return-variants  Disable return-space variant series
    --no-sliding          Disable hourly sliding-window aggregations
    --seed              Random seed (default: 42)
    --torch-dtype       float32 | bfloat16 | float16 (default: bfloat16)
    --eval-mae-only     Skip training, only evaluate --model-id on val data
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Make local project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chronos2_stock_augmentation import (
    AugConfig,
    AugmentedChronos2Dataset,
    OHLC_COLS,
    load_ohlc_csv,
    prepare_all_training_series,
    split_series_list,
)

try:
    from src.r2_client import R2Client
    _R2_AVAILABLE = True
except ImportError:
    _R2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DAILY_DIR   = "trainingdata"
DEFAULT_HOURLY_DIRS = "binance_spot_hourly"
DEFAULT_OUTPUT_ROOT = "chronos2_finetuned"
DEFAULT_MODEL_ID    = "amazon/chronos-2"
DEFAULT_CONTEXT     = 512
DEFAULT_PRED_LEN    = 1
DEFAULT_STEPS       = 30_000
DEFAULT_BATCH       = 256
DEFAULT_LR_FULL     = 5e-5
DEFAULT_LR_LORA     = 1e-4
DEFAULT_VAL_BARS    = 60
DEFAULT_SEED        = 42


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_mae_on_series(
    pipeline: Any,
    series_list: List[dict],
    context_length: int,
    prediction_length: int,
    quantile_index: int = 10,  # index of 0.5 in the default 21-quantile grid
    max_total_windows: int = 500,  # cap total inference calls for speed
) -> Tuple[float, float, int]:
    """
    Sliding-window MAE evaluation on a list of held-out val/test series.

    For each series, step through it one prediction_length at a time using
    the last context_length bars as context. Returns:
        (mae, mae_percent, n_windows)
    """
    import torch
    all_actuals: List[float] = []
    all_preds:   List[float] = []

    try:
        quantiles = list(pipeline.model.chronos_config.quantiles)
        q_idx = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))
    except Exception:
        q_idx = quantile_index

    total_windows = 0
    for s in series_list:
        if total_windows >= max_total_windows:
            break
        arr = s["target"]  # (4, T)
        T = arr.shape[-1]
        if T < context_length + prediction_length:
            continue

        for start in range(context_length, T - prediction_length + 1, prediction_length):
            if total_windows >= max_total_windows:
                break
            ctx_arr = arr[:, start - context_length : start]
            fut_arr = arr[:, start : start + prediction_length]

            ctx_tensor = torch.from_numpy(ctx_arr).float()  # (4, ctx_len)
            try:
                # pipeline.predict accepts list of series; each series can be (n_variates, T)
                preds = pipeline.predict(
                    [ctx_tensor],
                    prediction_length=prediction_length,
                    batch_size=1,
                )
            except Exception:
                continue

            if not preds:
                continue

            pred_tensor = preds[0]  # (n_variates, n_quantiles, pred_len)
            try:
                pred_np = pred_tensor.detach().cpu().numpy()
            except Exception:
                continue

            # close is the 4th channel (index 3), median quantile
            if pred_np.ndim == 3 and pred_np.shape[0] >= 4:
                pred_close = float(pred_np[3, q_idx, 0])
            elif pred_np.ndim == 2 and pred_np.shape[0] >= 4:
                pred_close = float(pred_np[3, q_idx])
            else:
                continue

            actual_close = float(fut_arr[3, 0])
            all_actuals.append(actual_close)
            all_preds.append(pred_close)
            total_windows += 1

    if not all_actuals:
        return float("nan"), float("nan"), 0

    actuals = np.array(all_actuals, dtype=np.float64)
    preds   = np.array(all_preds,   dtype=np.float64)
    mae = float(np.mean(np.abs(actuals - preds)))
    scale = float(np.mean(np.abs(actuals)))
    mae_pct = (mae / scale * 100.0) if scale > 0 else float("nan")
    return mae, mae_pct, len(all_actuals)


# ---------------------------------------------------------------------------
# torch dtype helper
# ---------------------------------------------------------------------------

def _parse_dtype(s: Optional[str]):
    if s is None:
        return None
    import torch
    m = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    key = s.strip().lower()
    if key not in m:
        raise ValueError(f"Unknown torch dtype: {s!r}")
    return m[key]


# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------

def load_pipeline(model_id: str, device_map: str = "cuda", torch_dtype_str: Optional[str] = "bfloat16"):
    try:
        from chronos import Chronos2Pipeline
    except ImportError as e:
        raise RuntimeError("Install chronos-forecasting: pip install chronos-forecasting") from e
    dtype = _parse_dtype(torch_dtype_str)
    print(f"Loading {model_id} ...")
    return Chronos2Pipeline.from_pretrained(model_id, device_map=device_map, torch_dtype=dtype)


# ---------------------------------------------------------------------------
# Muon optimizer (wraps our existing pufferlib_market/muon.py)
# ---------------------------------------------------------------------------

def _make_muon_optimizer(model: Any, lr: float, weight_decay: float = 0.01):
    """
    Build a Muon+AdamW mixed optimizer.

    2D+ weight matrices (attention projections, FFN) use Muon with NS orthogonalization.
    1D parameters (biases, layer-norm) use AdamW as fallback.

    Muon lr should be lower than typical AdamW lr since NS orthogonalization
    amplifies the effective step size. We scale muon_lr = lr * 0.4 and use
    the same lr for the AdamW 1D-param fallback.

    Requires pufferlib_market/muon.py to be importable.
    """
    try:
        from pufferlib_market.muon import make_muon_optimizer
    except ImportError as e:
        raise RuntimeError(
            "Muon optimizer requires pufferlib_market/muon.py — ensure pufferlib_market is on sys.path"
        ) from e
    # make_muon_optimizer splits params: 2D+ -> Muon, 1D -> AdamW
    # muon_lr is the orthogonalised-update LR; adamw_lr is for biases/norms
    muon_lr  = lr * 0.4   # Muon updates are "amplified" by NS → use smaller base LR
    adamw_lr = lr          # biases, layer-norm: same as the requested lr
    return make_muon_optimizer(
        model,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        adamw_wd=weight_decay,
    )


# ---------------------------------------------------------------------------
# R2 checkpoint upload
# ---------------------------------------------------------------------------

def upload_ckpt_to_r2(ckpt_dir: Path, r2_prefix: str) -> bool:
    """
    Upload all files in ckpt_dir to R2 under r2_prefix.
    Returns True if upload succeeded, False if R2 is not configured or upload fails.
    """
    if not _R2_AVAILABLE:
        return False
    try:
        client = R2Client()
    except Exception as e:
        print(f"R2 not configured, skipping upload: {e}")
        return False
    try:
        client.upload_directory(ckpt_dir, r2_prefix)
        print(f"Uploaded {ckpt_dir} → R2:{r2_prefix}")
        return True
    except AttributeError:
        # Fallback: upload_file for each file in ckpt_dir
        import os
        for fname in os.listdir(ckpt_dir):
            local = ckpt_dir / fname
            if local.is_file():
                key = f"{r2_prefix}/{fname}"
                try:
                    client.upload_file(local, key)
                except Exception as ue:
                    print(f"R2 upload failed for {fname}: {ue}")
                    return False
        print(f"Uploaded {ckpt_dir} → R2:{r2_prefix}")
        return True
    except Exception as e:
        print(f"R2 upload failed: {e}")
        return False


# ---------------------------------------------------------------------------
# LoRA application
# ---------------------------------------------------------------------------

def apply_lora(model: Any, r: int = 32, alpha: int = 64, dropout: float = 0.05):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise RuntimeError("Install peft: uv pip install peft") from e
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q", "k", "v", "o"],
        bias="none",
    )
    return get_peft_model(model, cfg)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _make_print_loss_callback():
    """Build a TrainerCallback that prints loss to stdout at every logging step."""
    from transformers.trainer_callback import TrainerCallback

    class _PrintLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss = logs.get("loss", logs.get("train_loss"))
            eval_loss = logs.get("eval_loss")
            lr = logs.get("learning_rate")
            parts = [f"step={state.global_step}"]
            if loss is not None:
                parts.append(f"loss={loss:.4f}")
            if eval_loss is not None:
                parts.append(f"eval_loss={eval_loss:.4f}")
            if lr is not None:
                parts.append(f"lr={lr:.2e}")
            print(f"[train] {' | '.join(parts)}", flush=True)

    return _PrintLossCallback()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    pipeline: Any,
    train_series: List[dict],
    val_series: List[dict],
    *,
    output_dir: Path,
    context_length: int = DEFAULT_CONTEXT,
    prediction_length: int = DEFAULT_PRED_LEN,
    batch_size: int = DEFAULT_BATCH,
    num_steps: int = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LR_FULL,
    finetune_mode: str = "full",
    lora_r: int = 32,
    seed: int = DEFAULT_SEED,
    aug_config: Optional[AugConfig] = None,
    use_muon: bool = False,
    grad_accum: int = 1,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Any:
    """
    Fine-tune pipeline on train_series with early stopping via val_series.
    Returns the trained pipeline.
    """
    import torch
    from copy import deepcopy
    from chronos.chronos2 import Chronos2Model
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback
    from transformers.training_args import TrainingArguments

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # W&B init (before TrainingArguments so run name is set before trainer reads env)
    _wandb_project = wandb_project
    if wandb_project:
        try:
            import wandb
            import os
            if wandb_entity:
                os.environ["WANDB_ENTITY"] = wandb_entity
            run_name = wandb_run_name or f"chronos2_{output_dir.name}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                entity=wandb_entity,
                config={
                    "context_length": context_length,
                    "prediction_length": prediction_length,
                    "batch_size": batch_size,
                    "num_steps": num_steps,
                    "learning_rate": learning_rate,
                    "finetune_mode": finetune_mode,
                    "use_muon": use_muon,
                    "grad_accum": grad_accum,
                    "seed": seed,
                },
            )
            print(f"W&B run: {wandb.run.url}")
        except ImportError:
            print("Warning: wandb not installed, disabling W&B logging")
            _wandb_project = None

    base_model = pipeline.model
    model_cfg = deepcopy(base_model.config)
    model = Chronos2Model(model_cfg).to(base_model.device)
    model.load_state_dict(base_model.state_dict())

    if finetune_mode == "lora":
        model = apply_lora(model, r=lora_r, alpha=lora_r * 2)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"LoRA active: {trainable:,} / {total:,} params trainable ({trainable/total:.1%})")
    else:
        trainable = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tune: {trainable:,} trainable params")

    output_patch_size = pipeline.model_output_patch_size

    train_dataset = AugmentedChronos2Dataset(
        inputs=[{"target": s["target"]} for s in train_series],
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        output_patch_size=output_patch_size,
        min_past=prediction_length,
        mode=DatasetMode.TRAIN,
        aug_config=aug_config,
    )

    # Val dataset (no augmentation)
    val_dataset = None
    callbacks: List[Any] = [_make_print_loss_callback()]
    eval_kwargs: Dict[str, Any] = {}

    if val_series:
        # Val series need at least min_past + prediction_length bars
        min_val_len = max(prediction_length + 1, 2)
        val_series_clipped = [s for s in val_series if s["target"].shape[-1] >= min_val_len]
        if val_series_clipped:
            val_dataset = AugmentedChronos2Dataset(
                inputs=[{"target": s["target"]} for s in val_series_clipped],
                context_length=context_length,
                prediction_length=prediction_length,
                batch_size=batch_size,
                output_patch_size=output_patch_size,
                mode=DatasetMode.VALIDATION,
                aug_config=None,  # no augmentation on val
            )
            # save_steps must be an exact multiple of eval_steps for load_best_model_at_end
            eval_steps = max(200, num_steps // 10)
            eval_kwargs = dict(
                save_strategy="steps",
                save_steps=eval_steps,
                eval_strategy="steps",
                eval_steps=eval_steps,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                label_names=["future_target"],
            )
            callbacks.append(EvaluateAndSaveFinalStepCallback())

    # Muon: build optimizer outside TrainingArguments (use custom optim callback)
    # If use_muon, we override the Trainer.create_optimizer method after construction.
    optim_str = "adamw_torch_fused"
    if use_muon:
        optim_str = "adamw_torch_fused"  # placeholder; will be overridden

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim=optim_str,
        logging_strategy="steps",
        logging_steps=200,
        disable_tqdm=False,
        report_to="wandb" if _wandb_project else "none",
        max_steps=num_steps,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=2,
        tf32=True,
        bf16=False,
        save_only_model=True,
        prediction_loss_only=True,
        save_total_limit=2,
        save_strategy=eval_kwargs.get("save_strategy", "no"),
        save_steps=eval_kwargs.get("save_steps", 500),
        eval_strategy=eval_kwargs.get("eval_strategy", "no"),
        eval_steps=eval_kwargs.get("eval_steps", 500),
        load_best_model_at_end=eval_kwargs.get("load_best_model_at_end", False),
        metric_for_best_model=eval_kwargs.get("metric_for_best_model"),
        label_names=eval_kwargs.get("label_names"),
        use_cpu=(str(base_model.device) == "cpu"),
        seed=seed,
    )

    trainer = Chronos2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )

    # Patch optimizer to Muon if requested
    if use_muon:
        _lr = learning_rate
        _wd = 0.01

        def _muon_create_optimizer():
            # HF Trainer.create_optimizer_and_scheduler calls create_optimizer() then reads
            # self.optimizer — so we must set self.optimizer here too, matching base behavior.
            opt = _make_muon_optimizer(model, lr=_lr, weight_decay=_wd)
            trainer.optimizer = opt
            return opt

        trainer.create_optimizer = _muon_create_optimizer
        print("Using Muon optimizer (Newton-Schulz orthogonalized momentum)")

    print(f"Training for {num_steps:,} steps | batch={batch_size} | lr={learning_rate:.2e}")
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed/60:.1f} min")

    if finetune_mode == "lora":
        try:
            model = model.merge_and_unload()
        except Exception as e:
            print(f"Warning: LoRA merge failed: {e}")

    # Update max_output_patches
    model.chronos_config.max_output_patches = max(
        model.chronos_config.max_output_patches,
        math.ceil(prediction_length / output_patch_size),
    )

    return pipeline.__class__(model=model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full Chronos2 fine-tune on all stock/crypto data")
    p.add_argument("--daily-data-dir",  default=DEFAULT_DAILY_DIR,
                   help="Dir with daily stock CSVs (trainingdata/)")
    p.add_argument("--hourly-data-dirs", default=DEFAULT_HOURLY_DIRS,
                   help="Comma-separated dirs with hourly CSVs")
    p.add_argument("--output-dir",  default=None,
                   help="Where to save checkpoint (default: chronos2_finetuned/stocks_all_YYYYMMDD_HHMMSS)")
    p.add_argument("--model-id",    default=DEFAULT_MODEL_ID)
    p.add_argument("--device-map",  default="cuda")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--finetune-mode", default="full", choices=["full", "lora"])
    p.add_argument("--lora-r",      type=int, default=32)
    p.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT)
    p.add_argument("--prediction-length", type=int, default=DEFAULT_PRED_LEN)
    p.add_argument("--batch-size",  type=int, default=DEFAULT_BATCH)
    p.add_argument("--num-steps",   type=int, default=DEFAULT_STEPS)
    p.add_argument("--learning-rate", type=float, default=None,
                   help="Peak LR (default: 5e-5 for full, 1e-4 for lora)")
    p.add_argument("--val-bars",    type=int, default=DEFAULT_VAL_BARS)
    p.add_argument("--test-bars",   type=int, default=DEFAULT_VAL_BARS)
    p.add_argument("--amp-log-std", type=float, default=0.30)
    p.add_argument("--noise-frac",  type=float, default=0.002)
    p.add_argument("--dropout-rate", type=float, default=0.02)
    p.add_argument("--freq-subsample-prob", type=float, default=0.0,
                   help="Prob of stride-2 subsampling per batch (teaches multi-timescale). 0=off")
    p.add_argument("--detrend",     action="store_true",
                   help="Linear detrend context windows during training")
    p.add_argument("--channel-dropout-prob", type=float, default=0.0,
                   help="Prob of zeroing one random OHLC channel in context (robustness). 0=off")
    p.add_argument("--time-warp-prob", type=float, default=0.0,
                   help="Prob of random time-warp on context (temporal invariance). 0=off")
    p.add_argument("--outlier-inject-prob", type=float, default=0.0,
                   help="Prob of injecting 1-3 extreme bars into context (crash robustness). 0=off")
    p.add_argument("--outlier-magnitude", type=float, default=5.0,
                   help="Outlier magnitude in units of local std (default: 5.0)")
    p.add_argument("--gap-inject-prob", type=float, default=0.0,
                   help="Prob of injecting overnight price gap (level shift) into context. 0=off")
    p.add_argument("--gap-magnitude-frac", type=float, default=0.05,
                   help="Gap size as fraction of channel mean (default: 0.05 = 5%%)")
    p.add_argument("--trend-inject-prob", type=float, default=0.0,
                   help="Prob of injecting random linear trend into context. 0=off")
    p.add_argument("--trend-magnitude-frac", type=float, default=0.10,
                   help="Trend magnitude as fraction of channel mean (default: 0.10 = 10%%)")
    p.add_argument("--no-return-variants", action="store_true")
    p.add_argument("--no-sliding",  action="store_true",
                   help="Disable hourly sliding-window daily aggregations")
    p.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    p.add_argument("--eval-mae-only", action="store_true",
                   help="Skip training; just eval --model-id on val data")
    p.add_argument("--cache-path",  default=None,
                   help="Path to .npz cache of prepared series (saves/loads to avoid re-processing)")
    p.add_argument("--use-muon",    action="store_true",
                   help="Use Muon optimizer (Newton-Schulz) instead of AdamW")
    p.add_argument("--r2-prefix",   default=None,
                   help="R2 object key prefix for checkpoint upload (e.g. chronos2/finetune/v1)")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Parallel workers for data loading")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (default: 1). Effective batch = "
                        "batch_size × grad_accum. Steps are counted in optimizer updates.")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name. If set, logs training metrics to W&B.")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name (default: auto-generated from tag/timestamp)")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity/team name")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Path to a trainer checkpoint dir to resume training from "
                        "(e.g. chronos2_finetuned/stocks_all_v3/trainer_workspace/checkpoint-20000)")
    p.add_argument("--skip-baseline-eval", action="store_true",
                   help="Skip the baseline MAE eval (auto-set when --resume-from-checkpoint is given)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # --- Resolve output dir ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        tag = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(DEFAULT_OUTPUT_ROOT) / f"stocks_all_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # --- Augmentation config ---
    aug_config = AugConfig(
        amplitude_log_std=args.amp_log_std,
        noise_std_frac=args.noise_frac,
        time_dropout_rate=args.dropout_rate,
        freq_subsample_prob=args.freq_subsample_prob,
        detrend_context=args.detrend,
        channel_dropout_prob=args.channel_dropout_prob,
        time_warp_prob=args.time_warp_prob,
        outlier_inject_prob=getattr(args, "outlier_inject_prob", 0.0),
        outlier_magnitude=getattr(args, "outlier_magnitude", 5.0),
        gap_inject_prob=getattr(args, "gap_inject_prob", 0.0),
        gap_magnitude_frac=getattr(args, "gap_magnitude_frac", 0.05),
        trend_inject_prob=getattr(args, "trend_inject_prob", 0.0),
        trend_magnitude_frac=getattr(args, "trend_magnitude_frac", 0.10),
        add_return_variants=not args.no_return_variants,
        sliding_daily_offsets=[] if args.no_sliding else [0, 1, 2, 3, 4, 5, 6],
    )

    # --- Load data ---
    daily_dir = Path(args.daily_data_dir) if args.daily_data_dir else None
    hourly_dirs = [Path(d.strip()) for d in args.hourly_data_dirs.split(",") if d.strip()]
    hourly_dirs = [d for d in hourly_dirs if d.exists()]

    if daily_dir is None or not daily_dir.exists():
        print(f"Warning: daily data dir {daily_dir} not found, skipping")
        daily_dir = None
    if not hourly_dirs:
        print("Warning: no hourly data dirs found")

    cache_path = Path(args.cache_path) if args.cache_path else None
    all_series = prepare_all_training_series(
        daily_data_dir=daily_dir,
        hourly_data_dirs=hourly_dirs if hourly_dirs else None,
        aug_config=aug_config,
        cache_path=cache_path,
        num_workers=args.num_workers,
    )

    if not all_series:
        print("ERROR: No training series loaded. Check --daily-data-dir and --hourly-data-dirs.")
        return 1

    train_series, val_series, _test_series = split_series_list(
        all_series,
        val_bars=args.val_bars,
        test_bars=args.test_bars,
    )
    print(f"Split: train={len(train_series)}, val={len(val_series)}")

    # --- Load base model ---
    pipeline = load_pipeline(args.model_id, args.device_map, args.torch_dtype)

    # --- Baseline eval (use full-length series, not the 60-bar val slices) ---
    # Pick series long enough to have context_length + prediction_length bars
    min_eval_len = args.context_length + args.prediction_length
    eval_series = [s for s in all_series if s["target"].shape[-1] >= min_eval_len][:100]
    skip_eval = bool(getattr(args, "skip_baseline_eval", False)) or bool(
        getattr(args, "resume_from_checkpoint", None)
    )
    if args.eval_mae_only or not skip_eval:
        if not args.eval_mae_only:
            print(f"Evaluating baseline on {len(eval_series)} series ...")
        mae_base, mae_pct_base, n_base = _eval_mae_on_series(
            pipeline, eval_series,
            args.context_length, args.prediction_length
        )
        if n_base > 0:
            print(f"Baseline val MAE: {mae_base:.4f}  MAE%: {mae_pct_base:.2f}%  (n={n_base})")
        else:
            print("Baseline eval: no valid windows (series too short for context_length)")
    else:
        print("Skipping baseline eval (resuming from checkpoint)")
        mae_base, mae_pct_base, n_base = 0.0, 0.0, 0

    if args.eval_mae_only:
        return 0

    # --- Train ---
    lr = args.learning_rate
    if lr is None:
        lr = DEFAULT_LR_LORA if args.finetune_mode == "lora" else DEFAULT_LR_FULL

    finetuned = train(
        pipeline,
        train_series=train_series,
        val_series=val_series,
        output_dir=output_dir / "trainer_workspace",
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=lr,
        finetune_mode=args.finetune_mode,
        lora_r=args.lora_r,
        seed=args.seed,
        aug_config=aug_config,
        use_muon=args.use_muon,
        grad_accum=args.grad_accum,
        wandb_project=getattr(args, "wandb_project", None),
        wandb_run_name=getattr(args, "wandb_run_name", None),
        wandb_entity=getattr(args, "wandb_entity", None),
        resume_from_checkpoint=getattr(args, "resume_from_checkpoint", None),
    )

    # --- Post-train eval (same eval_series as baseline for fair comparison) ---
    print("Evaluating finetuned model ...")
    mae_ft, mae_pct_ft, n_ft = _eval_mae_on_series(
        finetuned, eval_series,
        args.context_length, args.prediction_length
    )
    print(f"Finetuned val MAE: {mae_ft:.4f}  MAE%: {mae_pct_ft:.2f}%  (n={n_ft})")
    delta_pct = mae_pct_base - mae_pct_ft
    print(f"MAE% improvement: {delta_pct:+.2f} pp  ({delta_pct/mae_pct_base*100:+.1f}%)")

    # --- Save locally ---
    ckpt_path = output_dir / "finetuned-ckpt"
    finetuned.save_pretrained(ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # --- Upload to R2 ---
    if args.r2_prefix:
        r2_key = args.r2_prefix.rstrip("/")
        ok = upload_ckpt_to_r2(ckpt_path, r2_key)
        if ok:
            summary["r2_ckpt_prefix"] = r2_key

    # Write summary JSON
    summary = {
        "model_id": args.model_id,
        "output_dir": str(output_dir),
        "ckpt_path": str(ckpt_path),
        "finetune_mode": args.finetune_mode,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "n_train_series": len(train_series),
        "n_val_series": len(val_series),
        "baseline_mae_pct": round(mae_pct_base, 4),
        "finetuned_mae_pct": round(mae_pct_ft, 4),
        "improvement_pp": round(delta_pct, 4),
        "aug_config": asdict(aug_config),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
