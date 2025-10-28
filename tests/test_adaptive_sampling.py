import pytest

from src.adaptive_sampling import AdaptiveSamplerBounds


def test_adaptive_clamp_respects_minima_and_maxima():
    bounds = AdaptiveSamplerBounds(base_min_num=128, base_min_batch=32, num_floor=64, batch_floor=16)
    # Below minima clamps up
    num, batch = bounds.clamp(32, 8, max_num=512, max_batch=128)
    assert (num, batch) == (128, 32)
    # Above maxima clamps down
    num, batch = bounds.clamp(8192, 2048, max_num=4096, max_batch=512)
    assert num == 4096
    assert batch == 512
    # Maintain divisibility
    num, batch = bounds.clamp(1000, 125, max_num=2000, max_batch=200)
    assert num % batch == 0


def test_adaptive_reduce_and_reset():
    bounds = AdaptiveSamplerBounds(base_min_num=128, base_min_batch=32, num_floor=64, batch_floor=16)
    assert bounds.minima() == (128, 32)
    changed = bounds.reduce()
    assert changed is True
    assert bounds.minima() == (64, 16)
    # Further reduction hits the floor
    assert bounds.reduce() is False
    assert bounds.minima() == (64, 16)
    bounds.reset()
    assert bounds.minima() == (128, 32)


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        AdaptiveSamplerBounds(base_min_num=0, base_min_batch=1, num_floor=1, batch_floor=1)
    with pytest.raises(ValueError):
        AdaptiveSamplerBounds(base_min_num=32, base_min_batch=64, num_floor=16, batch_floor=8)
