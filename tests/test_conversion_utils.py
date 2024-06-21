import torch

from src.conversion_utils import convert_string_to_datetime, unwrap_tensor
def test_unwrap_tensor():
    assert unwrap_tensor(torch.tensor(1)) == 1
    assert unwrap_tensor(torch.tensor([1, 2])) == [1, 2]
    assert unwrap_tensor(1) == 1
    assert unwrap_tensor([1, 2]) == [1, 2]

def test_convert_string_to_datetime():
    from datetime import datetime
    assert convert_string_to_datetime("2024-04-16T19:53:01.577838") == datetime(2024, 4, 16, 19, 53, 1, 577838)
    assert convert_string_to_datetime(datetime(2024, 4, 16, 19, 53, 1, 577838)) == datetime(2024, 4, 16, 19, 53, 1, 577838)