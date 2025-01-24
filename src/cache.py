from pathlib import Path

from diskcache import Cache

cache_dir = Path(".cache")
cache_dir.mkdir(exist_ok=True, parents=True)
cache = Cache(str(cache_dir))

import asyncio
import functools
from typing import Any, Callable, Optional

def async_cache_decorator(
    name: Optional[str] = None,
    typed: bool = False,
    expire: Optional[int] = None,
    tag: Optional[str] = None,
    ignore: tuple = ()
):
    """Cache decorator for async functions that works with running event loops"""
    def decorator(func: Callable) -> Callable:
        # Create sync function for cache key generation
        @functools.wraps(func)
        def sync_key_func(*args: Any, **kwargs: Any) -> Any:
            return args, kwargs

        # Apply cache to key function
        cached_key_func = cache.memoize(
            name=name,
            typed=typed,
            expire=expire,
            tag=tag,
            ignore=ignore
        )(sync_key_func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = cached_key_func.__cache_key__(*args, **kwargs)
            result = cache.get(cache_key)
            
            if result is None:
                result = await func(*args, **kwargs)
                cache.set(cache_key, result)
            
            return result

        # Preserve cache key generation
        wrapper.__cache_key__ = cached_key_func.__cache_key__
        return wrapper

    return decorator
