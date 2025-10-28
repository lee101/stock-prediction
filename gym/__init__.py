"""
Compatibility shim that exposes the legacy `gym` import path while delegating to
Gymnasium. PufferLib 3.0 still imports `gym`, and older packages can coexist
with Gymnasium so long as the core spaces and base classes are available.
"""

from gymnasium import *  # type: ignore
from gymnasium import __all__ as _gymnasium_all  # type: ignore
from gymnasium import __version__ as __version__  # type: ignore
from gymnasium import spaces  # type: ignore

__all__ = list(_gymnasium_all)

