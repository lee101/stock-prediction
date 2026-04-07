"""CUDA graph capture/replay helper.

Mirrors the pattern used by `pufferlib_market/train.py`'s `--cuda-graph-ppo`
path: build persistent static input/output tensors, warm up on a private
stream, then capture. Replay is just `graph.replay()` after copying new
inputs into the static tensors.

Public API:
    capture_step(fn, example_inputs)
        Returns a CapturedStep object with:
            .static_inputs   dict[str, Tensor]   write into these
            .static_outputs  dict[str, Tensor]   read these after replay
            .replay()                             actually run the captured graph

`fn(static_inputs) -> dict[str, Tensor]` must be functional in the sense
that it ALWAYS writes to the same output tensor addresses on each call
(use `.copy_` for that).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


class CapturedStep:
    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_inputs: Dict[str, torch.Tensor],
        static_outputs: Dict[str, torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> None:
        self.graph = graph
        self.static_inputs = static_inputs
        self.static_outputs = static_outputs
        self.stream = stream

    def copy_inputs(self, **kwargs: torch.Tensor) -> None:
        for k, v in kwargs.items():
            if k in self.static_inputs:
                self.static_inputs[k].copy_(v)

    def replay(self) -> Dict[str, torch.Tensor]:
        self.graph.replay()
        return self.static_outputs


def capture_step(
    fn: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    example_inputs: Dict[str, torch.Tensor],
    warmup: int = 3,
    stream: Optional[torch.cuda.Stream] = None,
) -> CapturedStep:
    """Capture a CUDA graph for `fn(static_inputs)`.

    Requirements (matching pufferlib pattern):
    - All inputs are CUDA tensors of fixed shape/dtype.
    - `fn` writes outputs into tensors that come back in the returned dict;
      these MUST be the same tensor objects (addresses) every call. The
      simplest way: allocate them once during the warmup call and reuse.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("capture_step requires CUDA")
    device = next(iter(example_inputs.values())).device
    if device.type != "cuda":
        raise RuntimeError(f"capture_step inputs must be on cuda, got {device}")

    # Persistent static inputs (allocated once, used as the graph's only input slots).
    static_inputs = {k: v.detach().clone() for k, v in example_inputs.items()}

    capture_stream = stream if stream is not None else torch.cuda.Stream(device=device)

    # Warmup on the private stream so allocator caches are populated and any
    # lazy initialization happens before capture.
    static_outputs: Dict[str, torch.Tensor] = {}
    with torch.cuda.stream(capture_stream):
        for _ in range(max(1, int(warmup))):
            out = fn(static_inputs)
            if not static_outputs:
                # Lock in output addresses on the first warmup pass.
                for k, v in out.items():
                    static_outputs[k] = v
            else:
                # Subsequent warmups must reuse the same output addresses.
                for k, v in out.items():
                    if static_outputs[k].data_ptr() != v.data_ptr():
                        static_outputs[k].copy_(v)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            out = fn(static_inputs)
            for k, v in out.items():
                if static_outputs[k].data_ptr() != v.data_ptr():
                    static_outputs[k].copy_(v)
    capture_stream.synchronize()

    return CapturedStep(graph=graph, static_inputs=static_inputs,
                        static_outputs=static_outputs, stream=capture_stream)
