"""Path utilities for backtesting."""

from pathlib import Path
from typing import Union


def canonicalize_path(path_like: Union[str, Path]) -> Path:
    """Return an absolute path for cache directories regardless of environment input.

    Args:
        path_like: Path as string or Path object

    Returns:
        Absolute, resolved Path
    """
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)
