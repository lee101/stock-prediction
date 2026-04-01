"""Tests for BinanceHourlyPolicyMamba SSM-based trading policy."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binanceneural.config import PolicyConfig
from binanceneural.model import (
    BinanceHourlyPolicyMamba,
    PureTorchSSMBlock,
    build_policy,
)


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
    if _is_cuda_resource_pressure_error(exc):
        pytest.skip(f"Mamba CUDA test skipped under shared-GPU resource pressure: {exc}")


def _cuda_module_or_skip(module):
    try:
        return module.cuda()
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _cuda_randn_or_skip(*shape, **kwargs):
    try:
        return torch.randn(*shape, **kwargs)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _make_config(**overrides):
    defaults = dict(
        input_dim=10,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        model_arch="mamba",
        max_len=48,
        num_outputs=4,
        dropout=0.0,
        logits_softcap=12.0,
    )
    defaults.update(overrides)
    return PolicyConfig(**defaults)


class TestPureTorchSSMBlock:
    def test_output_shape(self):
        block = PureTorchSSMBlock(d_model=64, d_state=16, expand=2)
        x = torch.randn(2, 32, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)

    def test_gradient_flows(self):
        block = PureTorchSSMBlock(d_model=32, d_state=8, expand=2)
        x = torch.randn(2, 16, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_seq_lengths(self):
        block = PureTorchSSMBlock(d_model=32, d_state=8)
        for L in [1, 8, 48, 128]:
            out = block(torch.randn(1, L, 32))
            assert out.shape == (1, L, 32)


class TestBinanceHourlyPolicyMamba:
    def test_build_policy_mamba(self):
        cfg = _make_config()
        model = build_policy(cfg)
        assert isinstance(model, BinanceHourlyPolicyMamba)

    def test_build_policy_ssm(self):
        cfg = _make_config(model_arch="ssm")
        model = build_policy(cfg)
        assert isinstance(model, BinanceHourlyPolicyMamba)

    def test_forward_shape(self):
        cfg = _make_config()
        model = build_policy(cfg)
        x = torch.randn(2, 48, 10)
        out = model(x)
        assert "buy_price_logits" in out
        assert "sell_price_logits" in out
        assert "buy_amount_logits" in out
        assert "sell_amount_logits" in out
        assert out["buy_price_logits"].shape == (2, 48, 1)

    def test_forward_with_hold_hours(self):
        cfg = _make_config(num_outputs=5)
        model = build_policy(cfg)
        x = torch.randn(2, 48, 10)
        out = model(x)
        assert "hold_hours_logits" in out
        assert out["hold_hours_logits"].shape == (2, 48, 1)

    def test_forward_with_allocation(self):
        cfg = _make_config(num_outputs=6)
        model = build_policy(cfg)
        x = torch.randn(2, 48, 10)
        out = model(x)
        assert "allocation_logits" in out

    def test_decode_actions(self):
        cfg = _make_config()
        model = build_policy(cfg)
        x = torch.randn(2, 48, 10)
        out = model(x)
        ref = torch.ones(2, 48) * 100.0
        ch = torch.ones(2, 48) * 105.0
        cl = torch.ones(2, 48) * 95.0
        actions = model.decode_actions(out, reference_close=ref, chronos_high=ch, chronos_low=cl)
        assert "buy_price" in actions
        assert "sell_price" in actions
        assert "trade_amount" in actions
        assert actions["buy_price"].shape == (2, 48)
        assert (actions["sell_price"] > actions["buy_price"]).all()

    def test_gradient_flow(self):
        cfg = _make_config(hidden_dim=32, num_layers=2)
        model = build_policy(cfg)
        x = torch.randn(2, 16, 10, requires_grad=True)
        out = model(x)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert x.grad is not None
        grads = [p for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "no parameters received gradients"

    def test_param_count_reasonable(self):
        cfg = _make_config(hidden_dim=384, num_layers=4)
        model = build_policy(cfg)
        n = sum(p.numel() for p in model.parameters())
        assert n > 10_000
        assert n < 50_000_000

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_forward(self):
        cfg = _make_config(hidden_dim=128, num_layers=2)
        model = _cuda_module_or_skip(build_policy(cfg))
        x = _cuda_randn_or_skip(2, 48, 10, device="cuda")
        out = model(x)
        assert out["buy_price_logits"].device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_backward(self):
        cfg = _make_config(hidden_dim=128, num_layers=2)
        model = _cuda_module_or_skip(build_policy(cfg))
        x = _cuda_randn_or_skip(2, 48, 10, device="cuda", requires_grad=True)
        out = model(x)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert x.grad is not None

    def test_softcap(self):
        cfg = _make_config(logits_softcap=5.0, hidden_dim=32, num_layers=1)
        model = build_policy(cfg)
        x = torch.randn(2, 16, 10) * 100
        out = model(x)
        for v in out.values():
            assert v.abs().max() <= 5.0 + 1e-5

    def test_backend_attribute(self):
        cfg = _make_config()
        model = build_policy(cfg)
        assert hasattr(model, '_backend')
        assert model._backend in ("mamba3", "mamba2", "pure_torch")
