"""Regression test that reproduces cudagraph mutation skips under torch.compile."""

from __future__ import annotations

import contextlib

import pytest


@pytest.fixture(name="require_cuda")
def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():  # pragma: no cover - exercised only on CPU-only CI
        pytest.skip("CUDA device required for cudagraph mutation test")


@contextlib.contextmanager
def _temporarily_enable_cudagraph_debug():
    import torch._inductor.config as inductor_config

    prev_debug = inductor_config.debug
    prev_cudagraphs = inductor_config.triton.cudagraphs
    prev_error = inductor_config.triton.cudagraph_or_error
    try:
        inductor_config.debug = True
        inductor_config.triton.cudagraphs = True
        inductor_config.triton.cudagraph_or_error = True
        yield
    finally:  # pragma: no cover - defensive cleanup on failing platforms
        inductor_config.debug = prev_debug
        inductor_config.triton.cudagraphs = prev_cudagraphs
        inductor_config.triton.cudagraph_or_error = prev_error


def test_mutated_input_triggers_cudagraph_skip(require_cuda: None, caplog: pytest.LogCaptureFixture) -> None:
    import torch

    caplog.set_level("WARNING")

    def mutating_kernel(x: torch.Tensor) -> torch.Tensor:
        x.add_(1.0)  # in-place mutation should block cudagraph capture
        return x * 2.0

    with _temporarily_enable_cudagraph_debug():
        compiled = torch.compile(mutating_kernel, mode="reduce-overhead", fullgraph=True)
        with pytest.raises(RuntimeError) as excinfo:
            compiled(torch.ones(8, device="cuda"))

    message = str(excinfo.value)
    assert "mutated inputs" in message
    assert "skipping cudagraphs" in message
