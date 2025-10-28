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
try:  # PyTorch ≥ 2.1 uses torch.amp
    from torch.amp import GradScaler as _GradScaler  # type: ignore[attr-defined]
    from torch.amp import autocast as _amp_autocast  # type: ignore[attr-defined]

    def autocast_context(device_type: str, *, enabled: bool = True):
        return _amp_autocast(device_type, enabled=enabled)

except ImportError:  # pragma: no cover - PyTorch < 2.1 fallback
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    from torch.cuda.amp import autocast as _amp_autocast  # type: ignore

    def autocast_context(device_type: str, *, enabled: bool = True):
        return _amp_autocast(device_type=device_type, enabled=enabled)
from torch.optim import AdamW
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.torch_backend import configure_tf32_backends, maybe_set_float32_precision  # noqa: E402
from toto.inference.forecaster import TotoForecaster  # noqa: E402
from toto.model.toto import Toto  # noqa: E402

from tototraining.data import WindowConfig, build_dataloaders  # noqa: E402
from traininglib.prof import maybe_profile  # noqa: E402
from traininglib.prefetch import CudaPrefetcher  # noqa: E402
from traininglib.ema import EMA  # noqa: E402
from traininglib.losses import huber_loss, heteroscedastic_gaussian_nll, pinball_loss  # noqa: E402
from src.parameter_efficient import (  # noqa: E402
    LoraMetadata,
    freeze_module_parameters,
    inject_lora_adapters,
    save_lora_adapter,
)


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
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-logdir", default="runs/prof/toto")
    parser.add_argument("--prefetch-to-gpu", dest="prefetch_to_gpu", action="store_true", default=True)
    parser.add_argument("--no-prefetch-to-gpu", dest="prefetch_to_gpu", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema-eval", dest="ema_eval", action="store_false")
    parser.add_argument("--ema-eval", dest="ema_eval", action="store_true", default=True)
    parser.add_argument("--loss", choices=["huber", "mse", "heteroscedastic", "quantile", "nll"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.01)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--cuda-graph-warmup", type=int, default=3)
    parser.add_argument("--global-batch", type=int, default=None)
    parser.add_argument("--adapter", choices=["none", "lora"], default="none", help="Adapter type (LoRA for PEFT).")
    parser.add_argument("--adapter-r", type=int, default=8, help="LoRA adapter rank.")
    parser.add_argument("--adapter-alpha", type=float, default=16.0, help="LoRA scaling factor.")
    parser.add_argument("--adapter-dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser.add_argument(
        "--adapter-targets",
        type=str,
        default="model.patch_embed.projection,attention.wQKV,attention.wO,mlp.0,model.unembed,output_distribution",
        help="Comma separated substrings of module names to LoRA wrap.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Directory root for saving adapter weights (defaults to output_dir/adapters).",
    )
    parser.add_argument("--adapter-name", type=str, default=None, help="Adapter identifier (e.g., ticker).")
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze Toto base parameters when adapters are enabled.",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Allow Toto base weights to train alongside adapters.",
    )
    parser.add_argument(
        "--train-head",
        action="store_true",
        help="Keep unembed/output distribution parameters trainable in addition to adapters.",
    )
    return parser


def _parse_targets(raw: str) -> tuple[str, ...]:
    items = [item.strip() for item in (raw or "").split(",")]
    return tuple(sorted({item for item in items if item}))


def _prepare_forecast_tensors(distr, context, target, prediction_length):
    forecast = distr.mean[:, :, -prediction_length:]
    preds = forecast.squeeze(1)
    targets = target.squeeze(1)
    return preds, targets


def compute_batch_loss(distr, context, target, args) -> torch.Tensor:
    preds, targets = _prepare_forecast_tensors(distr, context, target, args.prediction_length)

    if args.loss == "nll":
        series = torch.cat([context, target], dim=-1)
        log_probs = distr.log_prob(series)
        target_log_probs = log_probs[:, :, -args.prediction_length :]
        return -target_log_probs.mean()
    if args.loss == "huber":
        return huber_loss(preds, targets, delta=args.huber_delta)
    if args.loss == "mse":
        return F.mse_loss(preds, targets)
    if args.loss == "heteroscedastic":
        if hasattr(distr, "log_scale"):
            log_sigma = distr.log_scale[:, :, -args.prediction_length :].squeeze(1)
        elif hasattr(distr, "scale"):
            log_sigma = distr.scale[:, :, -args.prediction_length :].squeeze(1).clamp_min(1e-5).log()
        else:
            raise RuntimeError("Distribution must expose scale/log_scale for heteroscedastic loss.")
        return heteroscedastic_gaussian_nll(preds, log_sigma, targets)
    if args.loss == "quantile":
        levels = args.quantiles or [0.1, 0.5, 0.9]
        losses = []
        if hasattr(distr, "icdf"):
            for q in levels:
                prob = torch.full_like(preds, float(q))
                quant_pred = distr.icdf(prob.unsqueeze(1)).squeeze(1)
                losses.append(pinball_loss(quant_pred, targets, q))
        elif hasattr(distr, "quantiles"):
            quant_tensor = distr.quantiles[:, :, -args.prediction_length :, :]
            if quant_tensor.shape[-1] != len(levels):
                raise RuntimeError("Quantile tensor count mismatch.")
            for idx, q in enumerate(levels):
                losses.append(pinball_loss(quant_tensor[:, 0, :, idx], targets, q))
        else:
            raise RuntimeError("Distribution must provide icdf or quantile tensors for quantile loss.")
        return sum(losses) / len(losses)
    raise AssertionError(f"Unsupported loss '{args.loss}'")


def _create_masks(series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padding_mask = torch.ones_like(series, dtype=torch.bool)
    id_mask = torch.zeros_like(series, dtype=torch.int)
    return padding_mask, id_mask


def _save_model(model: Toto, output_dir: Path, checkpoint_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / checkpoint_name
    try:
        model.save_pretrained(save_path)
    except NotImplementedError:
        fallback = save_path.with_suffix(".pth")
        torch.save(model.state_dict(), fallback)
        (output_dir / f"{fallback.name}.meta").write_text(
            "Saved state_dict fallback because save_pretrained is not implemented.\n",
            encoding="utf-8",
        )


def _train_iterable(loader, device, args):
    if args.prefetch_to_gpu and device.type == "cuda":
        return CudaPrefetcher(loader, device=device)
    return loader


def run_standard_epoch(
    loader,
    forward_pass,
    model,
    optimizer,
    scaler,
    ema,
    args,
    device,
    amp_enabled: bool,
):
    optimizer.zero_grad(set_to_none=True)
    epoch_loss = 0.0
    step_count = 0
    start_time = time.time()
    iterable = _train_iterable(loader, device, args)
    for step, (context, target) in enumerate(iterable, start=1):
        context = context.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        with autocast_context(device.type, enabled=amp_enabled):
            distr = forward_pass(context, target)
            loss = compute_batch_loss(distr, context, target, args)
        loss = loss / args.grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.grad_accum == 0:
            if args.clip_grad is not None:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update(model)

        epoch_loss += loss.detach().item() * args.grad_accum
        step_count += 1
    train_time = time.time() - start_time
    avg_loss = epoch_loss / max(step_count, 1)
    return avg_loss, train_time


def setup_cuda_graph(train_loader, forward_pass, optimizer, args, device):
    example_iter = iter(train_loader)
    example_context, example_target = next(example_iter)
    example_context = example_context.to(device=device, dtype=torch.float32)
    example_target = example_target.to(device=device, dtype=torch.float32)

    torch.cuda.synchronize()
    for _ in range(max(0, args.cuda_graph_warmup)):
        optimizer.zero_grad(set_to_none=True)
        distr = forward_pass(example_context, example_target)
        loss = compute_batch_loss(distr, example_context, example_target, args)
        loss.backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    static_context = example_context.clone()
    static_target = example_target.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        distr = forward_pass(static_context, static_target)
        loss = compute_batch_loss(distr, static_context, static_target, args)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return graph, static_context, static_target, loss


def run_cuda_graph_epoch(train_loader, graph_state, model, ema, args, device):
    graph, static_context, static_target, loss_ref = graph_state
    epoch_loss = 0.0
    step_count = 0
    start_time = time.time()
    for context, target in train_loader:
        context = context.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        static_context.copy_(context)
        static_target.copy_(target)
        graph.replay()
        epoch_loss += loss_ref.item()
        step_count += 1
        if ema:
            ema.update(model)
    train_time = time.time() - start_time
    avg_loss = epoch_loss / max(step_count, 1)
    return avg_loss, train_time


def run_validation(val_loader, forward_pass, model, ema, args, device):
    if val_loader is None:
        return None

    using_ema = False
    if ema and args.ema_eval:
        ema.apply_to(model)
        using_ema = True

    model.eval()
    losses = []
    mapes = []
    with torch.no_grad():
        iterable = _train_iterable(val_loader, device, args)
        for context, target in iterable:
            context = context.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            distr = forward_pass(context, target)
            batch_loss = compute_batch_loss(distr, context, target, args)
            losses.append(batch_loss.detach())
            forecast = distr.mean[:, :, -args.prediction_length :].squeeze(1)
            ape = torch.abs(forecast - target.squeeze(1)) / (torch.abs(target.squeeze(1)) + 1e-6)
            mapes.append(ape.mean())
    model.train()
    if using_ema:
        ema.restore(model)

    val_loss = torch.stack(losses).mean().item() if losses else 0.0
    val_mape = torch.stack(mapes).mean().item() * 100 if mapes else 0.0
    return val_loss, val_mape


def run_with_namespace(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        configure_tf32_backends(torch)
        maybe_set_float32_precision(torch, mode="medium")

    device = torch.device(args.device)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch:
        denom = args.batch_size * world_size
        if denom == 0 or args.global_batch % denom != 0:
            raise ValueError("global-batch must be divisible by per-device batch_size * world size")
        args.grad_accum = max(1, args.global_batch // denom)

    if args.cuda_graphs:
        if device.type != "cuda":
            raise RuntimeError("CUDA graphs require a CUDA device.")
        if args.grad_accum != 1:
            raise RuntimeError("CUDA graphs path currently requires grad_accum=1.")
        if args.prefetch_to_gpu:
            args.prefetch_to_gpu = False

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
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch_factor,
    )

    base_model_id = "Datadog/Toto-Open-Base-1.0"
    model = Toto.from_pretrained(base_model_id).to(device)

    if args.compile and not args.cuda_graphs and hasattr(model, "compile"):
        model.compile(mode=args.compile_mode)

    adapter_targets = _parse_targets(args.adapter_targets)
    adapter_metadata: LoraMetadata | None = None
    adapter_save_path: Path | None = None

    if args.adapter == "lora":
        if args.freeze_backbone:
            freeze_module_parameters(model)

        def _model_filter(name: str, child: torch.nn.Module) -> bool:
            return name.startswith("model.")

        replacements = inject_lora_adapters(
            model,
            target_patterns=adapter_targets,
            rank=args.adapter_r,
            alpha=args.adapter_alpha,
            dropout=args.adapter_dropout,
            module_filter=_model_filter,
        )
        if args.train_head:
            for name, param in model.named_parameters():
                if not name.startswith("model."):
                    continue
                if any(
                    name.startswith(prefix)
                    for prefix in (
                        "model.unembed",
                        "model.output_distribution",
                    )
                ):
                    param.requires_grad_(True)

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("LoRA enabled but no parameters marked trainable.")
        adapter_root = args.adapter_dir or (args.output_dir / "adapters")
        adapter_name = args.adapter_name or args.checkpoint_name
        adapter_save_path = Path(adapter_root) / adapter_name / "adapter.pt"
        adapter_metadata = LoraMetadata(
            adapter_type="lora",
            rank=args.adapter_r,
            alpha=args.adapter_alpha,
            dropout=args.adapter_dropout,
            targets=replacements,
            base_model=base_model_id,
        )
        print(f"[toto] Injected LoRA adapters on {len(replacements)} modules.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        trainable_params = list(model.parameters())
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    amp_enabled = device.type == "cuda" and not args.cuda_graphs
    scaler = _GradScaler(enabled=amp_enabled)

    ema = None
    if args.ema_decay and 0.0 < args.ema_decay < 1.0:
        ema = EMA(model, decay=args.ema_decay)

    def forward_pass(context: torch.Tensor, target: torch.Tensor):
        series = torch.cat([context, target], dim=-1)
        padding_mask, id_mask = _create_masks(series)
        base_distr, loc, scale = model.model(
            inputs=series,
            input_padding_mask=padding_mask,
            id_mask=id_mask,
            kv_cache=None,
            scaling_prefix_length=context.shape[-1],
        )
        return TotoForecaster.create_affine_transformed(base_distr, loc, scale)

    graph_state = None
    if args.cuda_graphs:
        graph_state = setup_cuda_graph(train_loader, forward_pass, optimizer, args, device)

    best_val_loss = math.inf
    best_epoch = -1

    profile_ctx = maybe_profile(args.profile, args.profile_logdir)
    with profile_ctx:
        for epoch in range(1, args.epochs + 1):
            model.train()
            if graph_state:
                avg_train_loss, train_time = run_cuda_graph_epoch(train_loader, graph_state, model, ema, args, device)
            else:
                avg_train_loss, train_time = run_standard_epoch(
                    train_loader,
                    forward_pass,
                    model,
                    optimizer,
                    scaler,
                    ema,
                    args,
                    device,
                    amp_enabled,
                )
            print(
                f"[Epoch {epoch}] train_loss={avg_train_loss:.6f} time={train_time:.1f}s "
                f"compiled={args.compile and not args.cuda_graphs}"
            )

            val_metrics = run_validation(val_loader, forward_pass, model, ema, args, device)
            if val_metrics is None:
                continue
            val_loss, val_mape = val_metrics
            print(f"[Epoch {epoch}] val_loss={val_loss:.6f} val_mape={val_mape:.3f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                _save_model(model, args.output_dir, args.checkpoint_name)
                if adapter_metadata and adapter_save_path is not None:
                    try:
                        save_lora_adapter(model, adapter_save_path, metadata=adapter_metadata)
                        print(f"[toto] Saved LoRA adapter to {adapter_save_path}")
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[toto] Failed to save LoRA adapter: {exc}")

    if best_epoch > 0:
        print(f"Best validation loss {best_val_loss:.6f} achieved at epoch {best_epoch}.")
    else:
        _save_model(model, args.output_dir, args.checkpoint_name)
        if adapter_metadata and adapter_save_path is not None:
            try:
                save_lora_adapter(model, adapter_save_path, metadata=adapter_metadata)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[toto] Failed to save LoRA adapter: {exc}")


def train() -> None:
    parser = create_argparser()
    args = parser.parse_args()
    run_with_namespace(args)


if __name__ == "__main__":
    train()
