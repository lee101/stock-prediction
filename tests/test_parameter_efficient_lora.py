from __future__ import annotations

import json

import torch
from torch import nn

from src.parameter_efficient import (
    LoraMetadata,
    freeze_module_parameters,
    inject_lora_adapters,
    save_lora_adapter,
)


class _ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def test_lora_injection_preserves_forward(tmp_path) -> None:
    model = _ToyNet()
    x = torch.randn(8, 4)
    baseline = model(x)

    freeze_module_parameters(model)
    replaced = inject_lora_adapters(
        model,
        target_patterns=("block.0",),
        rank=4,
        alpha=8.0,
        dropout=0.0,
    )

    assert replaced == ["block.0"]
    adapted = model(x)
    torch.testing.assert_close(baseline, adapted, atol=1e-6, rtol=1e-6)

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "LoRA injection should create trainable parameters."
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)

    adapter_path = tmp_path / "adapter.pt"
    metadata = LoraMetadata(
        adapter_type="lora",
        rank=4,
        alpha=8.0,
        dropout=0.0,
        targets=replaced,
        base_model="toy-model",
    )
    save_lora_adapter(model, adapter_path, metadata=metadata)

    payload = torch.load(adapter_path, map_location="cpu")
    assert "state_dict" in payload and payload["state_dict"], "Adapter payload must contain LoRA weights."

    meta = json.loads(adapter_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["rank"] == 4
    assert meta["base_model"] == "toy-model"
