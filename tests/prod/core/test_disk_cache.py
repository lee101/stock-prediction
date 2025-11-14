import os

import numpy as np
import pytest
import torch

from disk_cache import disk_cache

# Set the environment variable for testing
os.environ['TESTING'] = 'False'


@disk_cache
def cached_function(tensor):
    return tensor * 2


_noise_counter = {"calls": 0}


@disk_cache(ignore_kwargs={"noise"})
def cached_with_noise(value, *, noise: float = 0.0):
    _noise_counter["calls"] += 1
    return value + noise


def test_disk_cache_with_torch_tensor():
    # Create a random tensor
    tensor = torch.rand(5, 5)

    # Call the function for the first time
    result1 = cached_function(tensor)

    # Call the function again with the same tensor
    result2 = cached_function(tensor)

    # Check if the results are the same
    assert torch.all(result1.eq(result2)), "Cached result doesn't match the original result"


def test_disk_cache_with_different_tensors():
    # Create two different random tensors
    tensor1 = torch.rand(5, 5)
    tensor2 = torch.rand(5, 5)

    # Call the function with both tensors
    result1 = cached_function(tensor1)
    result2 = cached_function(tensor2)

    # Check if the results are different
    assert not torch.all(result1.eq(result2)), "Results for different tensors should not be the same"


def test_disk_cache_persistence():
    # Create a random tensor
    tensor = torch.rand(5, 5)

    # Call the function and get the result
    result1 = cached_function(tensor)

    # Clear the cache
    cached_function.cache_clear()

    tensor2 = torch.rand(5, 5)

    # Call the function again with the same tensor
    result2 = cached_function(tensor2)

    # Check if the results are different (since cache was cleared)
    assert not torch.all(result1.eq(result2)), "Results should be different after clearing cache"

    # Call the function once more
    result3 = cached_function(tensor)

    # Check if the last two results are the same (cached)
    assert torch.all(result1.eq(result3)), "Cached result doesn't match after re-caching"

    # Ensure that result2 and result3 are actually equal to tensor * 2
    assert torch.all(result2.eq(tensor2 * 2)), "Result2 is not correct"
    assert torch.all(result3.eq(tensor * 2)), "Result3 is not correct"


def test_disk_cache_with_numpy_array():
    # Create a random numpy array
    array = np.random.rand(5, 5)

    # Convert to torch tensor
    tensor = torch.from_numpy(array)

    # Call the function
    result = cached_function(tensor)

    # Check if the result is correct
    assert torch.all(result.eq(tensor * 2)), "Result is not correct for numpy array converted to tensor"


def test_disk_cache_ignores_specified_kwargs():
    cached_with_noise.cache_clear()
    _noise_counter["calls"] = 0
    result1 = cached_with_noise(5, noise=0.1)
    result2 = cached_with_noise(5, noise=0.9)
    assert result1 == result2 == 5.1
    assert _noise_counter["calls"] == 1

    # Different positional argument should produce miss
    result3 = cached_with_noise(6, noise=0.0)
    assert result3 == 6.0
    assert _noise_counter["calls"] == 2


if __name__ == "__main__":
    pytest.main([__file__])
