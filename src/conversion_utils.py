from datetime import datetime

from .dependency_injection import register_observer, resolve_torch

torch = resolve_torch()


def _refresh_torch(module):
    global torch
    torch = module


register_observer("torch", _refresh_torch)


def unwrap_tensor(data):
    if isinstance(data, torch.Tensor):
        if data.dim() == 0:
            return float(data)
        elif data.dim() >= 1:
            return data.tolist()
    else:
        return data


def convert_string_to_datetime(data):
    """
    convert string to datetime
    2024-04-16T19:53:01.577838 -> 2024-04-16 19:53:01.577838

    """
    if isinstance(data, str):
        return datetime.strptime(data, "%Y-%m-%dT%H:%M:%S.%f")
    else:
        return data
