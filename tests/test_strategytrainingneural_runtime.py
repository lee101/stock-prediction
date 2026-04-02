from __future__ import annotations

import numpy as np
import torch

from strategytrainingneural.runtime import NeuralStrategyEvaluator


class _DummyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moved_to: list[str] = []

    def to(self, device):
        self.moved_to.append(torch.device(device).type)
        return self

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.full((tensor.shape[0],), 0.25, dtype=torch.float32, device=tensor.device)


def test_predict_weights_auto_falls_back_to_cpu_on_cuda_pressure(monkeypatch):
    evaluator = NeuralStrategyEvaluator.__new__(NeuralStrategyEvaluator)
    evaluator.requested_device = None
    evaluator.device = torch.device("cuda")
    evaluator._model = _DummyPolicy()

    original_to = torch.Tensor.to

    def _fake_to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        if device is not None and torch.device(device).type == "cuda":
            raise torch.OutOfMemoryError("CUDA out of memory")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _fake_to, raising=False)

    weights = evaluator._predict_weights(np.array([[1.0, 2.0]], dtype=np.float32))

    assert evaluator.device == torch.device("cpu")
    assert weights.tolist() == [0.25]
    assert evaluator._model.moved_to == ["cpu"]
