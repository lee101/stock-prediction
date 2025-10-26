from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.amp import GradScaler, autocast

from .runtime_flags import enable_fast_kernels


@dataclass
class TrainingRunResult:
    steps: int
    elapsed_seconds: float
    final_loss: float
    history: List[float]


class _AttentionToyModel(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_multiplier: int) -> None:
        super().__init__()
        self.project_in = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(ff_multiplier * embed_dim, embed_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.project_in(x)
        attn_out, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        return self.ff(attn_out)


def _run_single(
    *,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    ff_multiplier: int,
    lr: float,
    target_loss: float,
    max_steps: int,
    use_fast_kernels: bool,
    seed: int,
) -> TrainingRunResult:
    torch.manual_seed(seed)
    model = _AttentionToyModel(embed_dim, num_heads, ff_multiplier).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(device="cuda")
    inputs = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float16)
    history: List[float] = []
    context = enable_fast_kernels() if use_fast_kernels else contextlib.nullcontext()

    start_time = time.perf_counter()
    with context:
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                preds = model(inputs)
                loss = (preds ** 2).mean()
            history.append(loss.detach().item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if loss.detach().item() <= target_loss:
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    return TrainingRunResult(steps=step, elapsed_seconds=elapsed, final_loss=float(history[-1]), history=history)


def measure_flash_speedup(
    *,
    device: str = "cuda",
    batch_size: int = 32,
    seq_len: int = 512,
    embed_dim: int = 256,
    num_heads: int = 8,
    ff_multiplier: int = 4,
    lr: float = 3e-4,
    target_loss: float = 1e-4,
    max_steps: int = 400,
    seeds: Tuple[int, int] = (184, 184),
) -> Dict[str, TrainingRunResult]:
    """
    Compare plain SDPA vs. flash-attn accelerated training on a toy attention block.

    Returns a dictionary containing metrics for the baseline run and the fast-kernel run.
    """
    device_obj = torch.device(device)
    results = {
        "baseline": _run_single(
            device=device_obj,
            batch_size=batch_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            lr=lr,
            target_loss=target_loss,
            max_steps=max_steps,
            use_fast_kernels=False,
            seed=seeds[0],
        ),
        "fast_kernels": _run_single(
            device=device_obj,
            batch_size=batch_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            lr=lr,
            target_loss=target_loss,
            max_steps=max_steps,
            use_fast_kernels=True,
            seed=seeds[1],
        ),
    }
    return results


if __name__ == "__main__":  # pragma: no cover - manual benchmarking hook
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required to run the attention benchmark.")
    stats = measure_flash_speedup()
    for label, payload in stats.items():
        print(
            f"{label:>12}: steps={payload.steps:4d}  final_loss={payload.final_loss:.5f}  "
            f"time={payload.elapsed_seconds:.3f}s"
        )
