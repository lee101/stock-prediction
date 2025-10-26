from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Iterable, Optional, Tuple

_SETUP_TARGETS: Tuple[Tuple[str, str], ...] = (
    ("src.conversion_utils", "setup_conversion_utils_imports"),
    ("src.forecasting_bolt_wrapper", "setup_forecasting_bolt_imports"),
    ("src.models.toto_wrapper", "setup_toto_wrapper_imports"),
    ("src.models.kronos_wrapper", "setup_kronos_wrapper_imports"),
    ("src.models.toto_aggregation", "setup_toto_aggregation_imports"),
)


def _iter_setup_functions() -> Iterable:
    for module_path, attr_name in _SETUP_TARGETS:
        try:
            module = import_module(module_path)
        except Exception:
            continue
        setup_fn = getattr(module, attr_name, None)
        if callable(setup_fn):
            yield setup_fn


def setup_src_imports(
    torch_module: Optional[ModuleType],
    numpy_module: Optional[ModuleType],
    pandas_module: Optional[ModuleType] = None,
    **extra_modules: Optional[ModuleType],
) -> None:
    """
    Inject heavy numerical dependencies into src.* modules that require them.
    """

    for setup_fn in _iter_setup_functions():
        try:
            setup_fn(
                torch_module=torch_module,
                numpy_module=numpy_module,
                pandas_module=pandas_module,
                **extra_modules,
            )
        except TypeError:
            kwargs = {
                "torch_module": torch_module,
                "numpy_module": numpy_module,
                "pandas_module": pandas_module,
            }
            setup_fn(**kwargs)


# Allow legacy import paths during the transition away from dependency_injection.
setup_imports = setup_src_imports


def _reset_for_tests() -> None:
    """
    Test helper preserved for backward compatibility.
    """
