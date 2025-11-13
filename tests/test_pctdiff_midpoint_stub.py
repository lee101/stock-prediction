import numpy as np

from src.pctdiff_helpers import pctdiff_midpoint_stub_returns


def test_pctdiff_midpoint_stub_returns_zero():
    returns, metadata = pctdiff_midpoint_stub_returns()
    assert isinstance(returns, np.ndarray)
    assert returns.size == 0
    assert metadata["pctdiff_midpoint_reason"] == "not_implemented"
