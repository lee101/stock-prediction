import functools
import hashlib
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Set

import torch


def _normalise_tensor(arg: torch.Tensor):
    tensor = arg.detach().cpu().numpy() if hasattr(arg, "detach") else arg.cpu().numpy()
    return hashlib.md5(tensor.tobytes()).hexdigest()


def _normalise_value(value):
    if isinstance(value, torch.Tensor):
        return _normalise_tensor(value)
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_value(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _normalise_value(val) for key, val in sorted(value.items())}
    return str(value)


def disk_cache(func: Optional[Callable] = None, *, ignore_kwargs: Optional[Iterable[str]] = None):
    """Cache decorator that stores results on disk keyed by tensor-friendly arguments.

    Args:
        func: Target function when used as ``@disk_cache``.
        ignore_kwargs: Optional iterable of keyword names that should be excluded from
            the cache key. This is useful for parameters like ``samples_per_batch``
            that influence performance but not the deterministic result.
    """

    def _decorate(target: Callable) -> Callable:
        cache_dir = Path(__file__).resolve().parent / '.cache' / target.__name__
        ignored: Set[str] = set(ignore_kwargs or ())

        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            if os.environ.get('TESTING') == 'True':
                return target(*args, **kwargs)

            key_parts = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    key_parts.append(_normalise_tensor(arg))
                else:
                    key_parts.append(str(arg))

            for k, v in sorted(kwargs.items()):
                if k in ignored:
                    continue
                key_parts.append(f"{k}:{_normalise_value(v)}")

            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f'{key}.pkl'

            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

            result = target(*args, **kwargs)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

            return result

        def cache_clear():
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            time.sleep(0.1)
            cache_dir.mkdir(parents=True, exist_ok=True)

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        return wrapper

    if func is not None and callable(func):
        return _decorate(func)
    return _decorate
