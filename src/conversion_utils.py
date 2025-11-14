from datetime import datetime
from importlib import import_module
from types import ModuleType
from typing import Any


def _optional_import(module_name: str) -> ModuleType | None:
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        return None


torch: ModuleType | None = _optional_import("torch")


def setup_conversion_utils_imports(
    *,
    torch_module: ModuleType | None = None,
    **_: Any,
) -> None:
    global torch
    if torch_module is not None:
        torch = torch_module


def _torch_module() -> ModuleType | None:
    global torch
    if torch is not None:
        return torch
    try:
        module = import_module("torch")
    except ModuleNotFoundError:
        return None
    torch = module
    return module


def unwrap_tensor(data: Any):
    torch_mod = _torch_module()
    if torch_mod is not None and isinstance(data, torch_mod.Tensor):
        if data.dim() == 0:
            return float(data)
        if data.dim() >= 1:
            return data.tolist()
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
