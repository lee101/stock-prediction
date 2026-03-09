from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import FastForecaster2Config
from .data import build_data_bundle
from .kernels import mae_loss, weighted_mae_loss
from .model import FastForecaster2Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro-benchmark FastForecaster2 forward+MAE throughput.")
    parser.add_argument("--dataset", choices=["hourly", "daily"], default="hourly")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--max-symbols", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--qk-norm", dest="qk_norm", action="store_true", default=True)
    parser.add_argument("--no-qk-norm", dest="qk_norm", action="store_false")
    parser.add_argument("--qk-norm-eps", type=float, default=1e-6)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--weighted-loss", action="store_true", default=False)
    parser.add_argument("--use-cpp-kernels", action="store_true", default=False)
    parser.add_argument("--build-cpp-extension", action="store_true", default=False)
    return parser.parse_args()


def _default_data_dir(dataset: str) -> Path:
    if dataset == "daily":
        return Path("trainingdata")
    return Path("trainingdatahourly") / "stocks"


def _parse_symbols(raw: str) -> tuple[str, ...] | None:
    symbols = tuple(sorted({s.strip().upper() for s in raw.split(",") if s.strip()}))
    return symbols or None


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return torch.no_grad()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", enabled=False)


def main() -> None:
    args = parse_args()
    cfg = FastForecaster2Config(
        data_dir=args.data_dir or _default_data_dir(args.dataset),
        symbols=_parse_symbols(args.symbols),
        max_symbols=args.max_symbols,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        epochs=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        qk_norm=args.qk_norm,
        qk_norm_eps=args.qk_norm_eps,
        precision=args.precision,
        torch_compile=False,
        num_workers=0,
        max_train_windows_per_symbol=5000,
        max_eval_windows_per_symbol=1000,
    )

    bundle = build_data_bundle(cfg)
    loader = DataLoader(bundle.train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    device = torch.device(cfg.resolved_device())
    model = FastForecaster2Model(
        input_dim=bundle.feature_dim,
        horizon=cfg.horizon,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_multiplier=cfg.ff_multiplier,
        dropout=cfg.dropout,
        max_symbols=len(bundle.symbols),
        qk_norm=cfg.qk_norm,
        qk_norm_eps=cfg.qk_norm_eps,
    ).to(device)
    model.eval()

    x, target_ret, target_close, base_close, symbol_idx = batch
    x = x.to(device)
    target_close = target_close.to(device)
    base_close = base_close.to(device)
    symbol_idx = symbol_idx.to(device)
    horizon_weights = torch.linspace(1.0, 0.75, cfg.horizon, device=device, dtype=target_close.dtype)

    def _loss_fn(pred_price: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if args.weighted_loss:
            return weighted_mae_loss(
                pred_price,
                target,
                horizon_weights,
                use_cpp=args.use_cpp_kernels,
                build_extension=args.build_cpp_extension,
            )
        return mae_loss(
            pred_price,
            target,
            use_cpp=args.use_cpp_kernels,
            build_extension=args.build_cpp_extension,
        )

    for _ in range(max(1, args.warmup)):
        with torch.no_grad():
            with _autocast_context(device, args.precision):
                pred_ret = model(x, symbol_idx)
                pred_price = base_close.unsqueeze(1) * (1.0 + pred_ret)
                _ = _loss_fn(pred_price, target_close)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(max(1, args.iters)):
        with torch.no_grad():
            with _autocast_context(device, args.precision):
                pred_ret = model(x, symbol_idx)
                pred_price = base_close.unsqueeze(1) * (1.0 + pred_ret)
                _ = _loss_fn(pred_price, target_close)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    batch_size = x.shape[0]
    iters = max(1, args.iters)
    per_iter_ms = (elapsed / iters) * 1000.0
    samples_per_sec = (batch_size * iters) / max(elapsed, 1e-9)

    print("[fastforecaster2] Benchmark")
    print(f"  device: {device}")
    print(f"  batch_size: {batch_size}")
    print(f"  lookback: {cfg.lookback}, horizon: {cfg.horizon}")
    print(f"  precision: {cfg.precision}")
    print(f"  weighted_loss: {args.weighted_loss}")
    print(f"  use_cpp_kernels: {args.use_cpp_kernels}")
    print(f"  build_cpp_extension: {args.build_cpp_extension}")
    print(f"  mean_latency_ms: {per_iter_ms:.3f}")
    print(f"  throughput_samples_per_sec: {samples_per_sec:.1f}")


if __name__ == "__main__":
    main()
