import torch

from src.conversion_utils import unwrap_tensor
def test_unwrap_tensor():
    assert unwrap_tensor(torch.tensor(1)) == 1
    assert unwrap_tensor(torch.tensor([1, 2])) == [1, 2]
    assert unwrap_tensor(1) == 1
    assert unwrap_tensor([1, 2]) == [1, 2]