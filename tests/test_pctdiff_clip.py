import numpy as np

from src.pctdiff_helpers import clip_pctdiff_returns, reset_pctdiff_clip_flag


def test_clip_pctdiff_returns_limits_values():
    reset_pctdiff_clip_flag()
    values = np.array([0.2, -0.15, 0.05], dtype=float)
    clipped = clip_pctdiff_returns(values, max_abs_return=0.1)
    assert np.allclose(clipped, np.array([0.1, -0.1, 0.05], dtype=float))


def test_clip_pctdiff_returns_no_change():
    reset_pctdiff_clip_flag()
    values = np.array([0.02, -0.03], dtype=float)
    clipped = clip_pctdiff_returns(values, max_abs_return=0.1)
    assert np.allclose(clipped, values)
