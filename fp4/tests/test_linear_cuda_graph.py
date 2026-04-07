"""CUDA-graph-safety test for NVFP4Linear.

Captures a forward+backward pass through a 256->128 NVFP4Linear into a
CUDAGraph, replays it 100 times, and checks that:
  * outputs and input-grads stay finite across replays,
  * captured outputs match an eager run within block-quant tolerance.

Skipped on non-CUDA boxes.
"""
from __future__ import annotations

import pytest
import torch

from fp4.linear import NVFP4Linear
from fp4.quant import prime_caches


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture requires a CUDA device"
)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).pow(2).mean().sqrt()
    den = b.pow(2).mean().sqrt().clamp(min=1e-8)
    return float((num / den).item())


def test_nvfp4_linear_cuda_graph_capture_replay():
    torch.manual_seed(0)
    device = torch.device("cuda")
    in_f, out_f, batch = 256, 128, 64

    # Prime per-(device, dtype) caches outside capture so the level tables
    # are already on-device when the graph records its ops.
    prime_caches(device, (torch.float32,))

    layer = NVFP4Linear(in_f, out_f, bias=True).to(device)
    # Static input/output/grad buffers — same addresses every replay.
    x = torch.randn(batch, in_f, device=device, requires_grad=True)
    gy = torch.randn(batch, out_f, device=device)

    # Eager reference on a *separate* tensor so we don't pollute x.grad's
    # autograd state before the capture.
    with torch.no_grad():
        x_ref = x.detach().clone().requires_grad_(True)
    y_eager = layer(x_ref)
    y_eager.backward(gy)
    y_eager_ref = y_eager.detach().clone()
    gx_eager_ref = x_ref.grad.detach().clone()
    layer.zero_grad(set_to_none=True)
    del x_ref, y_eager
    torch.cuda.synchronize()

    # Warm up on a private stream so allocator/autograd state is stable.
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            layer.zero_grad(set_to_none=True)
            x.grad = None
            y = layer(x)
            y.backward(gy)
    torch.cuda.current_stream().wait_stream(s)

    # Capture one fwd+bwd into a CUDAGraph on the same side stream.
    graph = torch.cuda.CUDAGraph()
    layer.zero_grad(set_to_none=True)
    x.grad = None
    with torch.cuda.graph(graph, stream=s):
        y_static = layer(x)
        y_static.backward(gy)

    # Replay 100x and assert we never produce non-finite tensors.
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(y_static).all().item(), "captured forward output has non-finite values"
    assert x.grad is not None and torch.isfinite(x.grad).all().item(), \
        "captured input grad has non-finite values"

    # Captured output should match the eager reference within block-quant tolerance.
    fwd_rel = _rel(y_static, y_eager_ref)
    assert fwd_rel < 0.25, f"captured fwd rel err too high: {fwd_rel}"

    # The captured backward uses stochastic rounding so grads have variance;
    # use a looser bound on x.grad but still require finiteness + same scale.
    gx_scale = x.grad.pow(2).mean().sqrt().item()
    ref_scale = gx_eager_ref.pow(2).mean().sqrt().item()
    assert 0.25 * ref_scale < gx_scale < 4.0 * ref_scale, (
        f"captured grad_x scale diverged: got {gx_scale}, ref {ref_scale}"
    )
