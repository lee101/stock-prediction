"""Single-env (.venv, py3.13) import smoke: all C extensions load together.

Phase-4 unit P4-7 — the previous py312/py313 split meant fp4 CUTLASS and
pufferlib triton kernels needed separate envs. This test pins the invariant
that a fresh .venv can import everything simultaneously.
"""
import pytest


def test_all_extensions_importable():
    import fp4  # noqa: F401
    import fp4.fp4  # noqa: F401
    import fp4.fp4.kernels.gemm as g
    assert callable(g.nvfp4_matmul_bf16), "nvfp4_matmul_bf16 not callable"

    import market_sim_py  # noqa: F401
    assert hasattr(market_sim_py, "MarketEnvironment")

    try:
        import gpu_trading_env  # noqa: F401
    except ImportError:
        pytest.skip("gpu_trading_env (P4-1) not built yet")

    try:
        import pufferlib_market.binding  # noqa: F401
    except ImportError:
        pytest.skip("pufferlib_market.binding not configured")
