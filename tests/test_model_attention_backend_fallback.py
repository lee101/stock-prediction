from __future__ import annotations

import torch

from binanceneural.config import PolicyConfig
from binanceneural.model import BinanceHourlyPolicy


def test_classic_policy_falls_back_after_attention_kernel_error(monkeypatch) -> None:
    policy = BinanceHourlyPolicy(
        PolicyConfig(
            input_dim=4,
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            max_len=16,
        )
    )
    features = torch.randn(2, 8, 4)
    hidden_dim = int(policy.norm.normalized_shape[0])
    calls = {"fast": 0, "fallback": 0}

    def _raise_no_kernel(hidden: torch.Tensor) -> torch.Tensor:
        calls["fast"] += 1
        raise RuntimeError("No available kernel. Aborting execution.")

    def _fallback(hidden: torch.Tensor) -> torch.Tensor:
        calls["fallback"] += 1
        return torch.zeros(hidden.shape[0], hidden.shape[1], hidden_dim, dtype=hidden.dtype)

    monkeypatch.setattr(policy, "_encode_once", _raise_no_kernel)
    monkeypatch.setattr(policy, "_encode_math_fallback", _fallback)

    out_first = policy(features)
    out_second = policy(features)

    assert policy._attention_backend_fallback is True
    assert calls == {"fast": 1, "fallback": 2}
    assert out_first["buy_price_logits"].shape == (2, 8, 1)
    assert out_second["sell_price_logits"].shape == (2, 8, 1)
