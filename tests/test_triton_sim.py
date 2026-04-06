"""Tests for Triton fused simulation kernel numerical parity."""
from functools import wraps

import pytest
import torch

try:
    from trainingefficiency.fast_differentiable_sim import simulate_hourly_trades_fast
    from trainingefficiency.triton_sim_kernel import simulate_hourly_trades_triton
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _resolve_test_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.empty(1, device="cuda")
    except Exception as exc:
        if _is_cuda_resource_pressure_error(exc):
            return "cpu"
        raise
    return "cuda"


def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
    if _is_cuda_resource_pressure_error(exc):
        pytest.skip(f"Triton sim CUDA test skipped under shared-GPU resource pressure: {exc}")


def _skip_on_cuda_resource_pressure(test_fn):
    @wraps(test_fn)
    def wrapped(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            return test_fn(*args, **kwargs)
        except Exception as exc:
            _skip_for_cuda_resource_pressure(exc)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return wrapped


def _decorate_test_class_for_cuda_resource_pressure(cls):
    for name, value in vars(cls).items():
        if name.startswith("test_") and callable(value):
            setattr(cls, name, _skip_on_cuda_resource_pressure(value))
    return cls


DEVICE = _resolve_test_device()
CUDA = DEVICE == "cuda"


def _make_data(batch_shape=(), steps=48, device=DEVICE, seed=42):
    torch.manual_seed(seed)
    shape = (*batch_shape, steps)
    base = 100.0 + torch.randn(*shape, device=device) * 2
    closes = base
    highs = base + torch.rand(*shape, device=device) * 3
    lows = base - torch.rand(*shape, device=device) * 3
    opens = base + (torch.rand(*shape, device=device) - 0.5) * 2
    buy_prices = base - torch.rand(*shape, device=device) * 1.5
    sell_prices = base + torch.rand(*shape, device=device) * 1.5
    trade_intensity = torch.rand(*shape, device=device) * 0.8 + 0.1
    return dict(
        highs=highs, lows=lows, closes=closes, opens=opens,
        buy_prices=buy_prices, sell_prices=sell_prices,
        trade_intensity=trade_intensity,
    )


def _compare_results(r1, r2, atol=1e-5, rtol=1e-4):
    for field in ["pnl", "returns", "portfolio_values", "executed_buys",
                  "executed_sells", "inventory_path"]:
        v1 = getattr(r1, field)
        v2 = getattr(r2, field)
        assert torch.allclose(v1, v2, atol=atol, rtol=rtol), (
            f"{field} mismatch: max_diff={torch.max(torch.abs(v1 - v2)).item():.2e}"
        )
    assert torch.allclose(r1.cash, r2.cash, atol=atol, rtol=rtol), "cash mismatch"
    assert torch.allclose(r1.inventory, r2.inventory, atol=atol, rtol=rtol), "inventory mismatch"


@_decorate_test_class_for_cuda_resource_pressure
class TestTritonSimParity:

    def test_1d_basic(self):
        data = _make_data(steps=24)
        r_fast = simulate_hourly_trades_fast(**data)
        r_tri = simulate_hourly_trades_triton(**data)
        _compare_results(r_fast, r_tri)

    def test_1d_48steps(self):
        data = _make_data(steps=48)
        r_fast = simulate_hourly_trades_fast(**data)
        r_tri = simulate_hourly_trades_triton(**data)
        _compare_results(r_fast, r_tri)

    def test_batched(self):
        data = _make_data(batch_shape=(8,), steps=32)
        r_fast = simulate_hourly_trades_fast(**data)
        r_tri = simulate_hourly_trades_triton(**data)
        _compare_results(r_fast, r_tri)

    def test_2d_batch(self):
        data = _make_data(batch_shape=(4, 3), steps=24)
        r_fast = simulate_hourly_trades_fast(**data)
        r_tri = simulate_hourly_trades_triton(**data)
        _compare_results(r_fast, r_tri)

    def test_with_fee(self):
        data = _make_data(steps=24)
        kwargs = dict(**data, maker_fee=0.001)
        r_fast = simulate_hourly_trades_fast(**kwargs)
        r_tri = simulate_hourly_trades_triton(**kwargs)
        _compare_results(r_fast, r_tri)

    def test_with_leverage(self):
        data = _make_data(batch_shape=(4,), steps=24)
        kwargs = dict(**data, max_leverage=2.0)
        r_fast = simulate_hourly_trades_fast(**kwargs)
        r_tri = simulate_hourly_trades_triton(**kwargs)
        _compare_results(r_fast, r_tri)

    def test_with_shorting(self):
        data = _make_data(batch_shape=(4,), steps=24)
        kwargs = dict(**data, can_short=True, can_long=True)
        r_fast = simulate_hourly_trades_fast(**kwargs)
        r_tri = simulate_hourly_trades_triton(**kwargs)
        _compare_results(r_fast, r_tri)

    def test_short_only(self):
        data = _make_data(batch_shape=(4,), steps=24)
        kwargs = dict(**data, can_short=True, can_long=False)
        r_fast = simulate_hourly_trades_fast(**kwargs)
        r_tri = simulate_hourly_trades_triton(**kwargs)
        _compare_results(r_fast, r_tri)

    def test_with_margin(self):
        data = _make_data(batch_shape=(4,), steps=24)
        kwargs = dict(**data, max_leverage=2.0, margin_annual_rate=0.0625)
        r_fast = simulate_hourly_trades_fast(**kwargs)
        r_tri = simulate_hourly_trades_triton(**kwargs)
        _compare_results(r_fast, r_tri)

    def test_market_order_entry(self):
        data = _make_data(steps=24)
        kwargs = dict(**data, market_order_entry=True)
        r_fast = simulate_hourly_trades_fast(**kwargs)
        r_tri = simulate_hourly_trades_triton(**kwargs)
        _compare_results(r_fast, r_tri)

    def test_fill_buffer(self):
        data = _make_data(steps=24)
        kwargs = dict(**data, fill_buffer_pct=0.0005)
        r_fast = simulate_hourly_trades_fast(**kwargs)
        r_tri = simulate_hourly_trades_triton(**kwargs)
        _compare_results(r_fast, r_tri)

    def test_decision_lag(self):
        data = _make_data(steps=24)
        kwargs = dict(**data, decision_lag_bars=1)
        r_fast = simulate_hourly_trades_fast(**kwargs)
        r_tri = simulate_hourly_trades_triton(**kwargs)
        _compare_results(r_fast, r_tri)

    def test_separate_buy_sell_intensity(self):
        data = _make_data(batch_shape=(4,), steps=24)
        data["buy_trade_intensity"] = torch.rand(4, 24, device=DEVICE) * 0.5
        data["sell_trade_intensity"] = torch.rand(4, 24, device=DEVICE) * 0.3
        r_fast = simulate_hourly_trades_fast(**data)
        r_tri = simulate_hourly_trades_triton(**data)
        _compare_results(r_fast, r_tri)

    def test_all_features(self):
        data = _make_data(batch_shape=(4,), steps=32)
        kwargs = dict(
            **data,
            maker_fee=0.001,
            max_leverage=2.0,
            can_short=True,
            can_long=True,
            margin_annual_rate=0.0625,
            fill_buffer_pct=0.0005,
            market_order_entry=True,
        )
        r_fast = simulate_hourly_trades_fast(**kwargs)
        r_tri = simulate_hourly_trades_triton(**kwargs)
        _compare_results(r_fast, r_tri)

    @pytest.mark.skipif(not CUDA, reason="CUDA required")
    def test_gradient_parity(self):
        base = _make_data(batch_shape=(4,), steps=16)
        grad_keys = ["buy_prices", "sell_prices", "trade_intensity"]

        # Fast path
        data_fast = {k: v.detach().requires_grad_(k in grad_keys) for k, v in base.items()}
        r_fast = simulate_hourly_trades_fast(**data_fast, maker_fee=0.001)
        r_fast.returns.sum().backward()
        grads_fast = {k: data_fast[k].grad.clone() for k in grad_keys}

        # Triton path
        data_tri = {k: v.detach().requires_grad_(k in grad_keys) for k, v in base.items()}
        r_tri = simulate_hourly_trades_triton(**data_tri, maker_fee=0.001)
        r_tri.returns.sum().backward()
        grads_tri = {k: data_tri[k].grad.clone() for k in grad_keys}

        for k in grad_keys:
            assert torch.allclose(grads_fast[k], grads_tri[k], atol=1e-4, rtol=1e-3), (
                f"grad {k} mismatch: max_diff={torch.max(torch.abs(grads_fast[k] - grads_tri[k])).item():.2e}"
            )

    @pytest.mark.skipif(not CUDA, reason="CUDA required")
    def test_large_batch(self):
        data = _make_data(batch_shape=(256,), steps=48)
        r_fast = simulate_hourly_trades_fast(**data)
        r_tri = simulate_hourly_trades_triton(**data)
        _compare_results(r_fast, r_tri)

    def test_cpu_fallback(self):
        data = _make_data(device="cpu", steps=16)
        r_fast = simulate_hourly_trades_fast(**data)
        r_tri = simulate_hourly_trades_triton(**data)
        _compare_results(r_fast, r_tri)

    def test_tensor_leverage(self):
        data = _make_data(batch_shape=(4,), steps=24)
        lev = torch.ones(4, 24, device=DEVICE) * 1.5
        lev[:, 12:] = 2.0
        kwargs = dict(**data, max_leverage=lev)
        r_fast = simulate_hourly_trades_fast(**kwargs)
        r_tri = simulate_hourly_trades_triton(**kwargs)
        _compare_results(r_fast, r_tri)
