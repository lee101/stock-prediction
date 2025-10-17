import functools
import hashlib
import pickle
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Tuple, TypeVar, cast

from diskcache import Cache

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])

cache_dir = Path(".cache")
cache_dir.mkdir(exist_ok=True, parents=True)
cache = Cache(str(cache_dir))


def async_cache_decorator(
    name: Optional[str] = None,
    typed: bool = False,
    expire: Optional[int] = None,
    tag: Optional[str] = None,
    ignore: Tuple[Any, ...] = (),
) -> Callable[[F], F]:
    """Cache decorator for async functions that works with running event loops"""
    def decorator(func: F) -> F:
        # Create sync function for cache key generation
        @functools.wraps(func)
        def sync_key_func(*args: Any, **kwargs: Any) -> Any:
            return args, kwargs

        # Apply cache to key function
        cached_key_func: Any = cache.memoize(
            name=name,
            typed=typed,
            expire=expire,
            tag=tag,
            ignore=ignore
        )(sync_key_func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate a hash of the cache key to avoid "string or blob too big" error
            cache_key_fn = getattr(cached_key_func, "__cache_key__", None)
            if cache_key_fn is None:
                raise AttributeError("DiskCache memoize wrapper missing __cache_key__ attribute.")

            cache_key = cache_key_fn(*args, **kwargs)
            key_hash = hashlib.md5(pickle.dumps(cache_key)).hexdigest()

            result = cache.get(key_hash)

            if result is None:
                result = await func(*args, **kwargs)
                cache.set(key_hash, result)

            return result

        # Preserve cache key generation
        cache_key_fn = getattr(cached_key_func, "__cache_key__", None)
        if cache_key_fn is None:
            raise AttributeError("DiskCache memoize wrapper missing __cache_key__ attribute.")
        setattr(wrapper, "__cache_key__", cache_key_fn)
        return cast(F, wrapper)

    return decorator
