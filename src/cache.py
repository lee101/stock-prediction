import functools
import hashlib
import logging
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Tuple, TypeVar, cast

from diskcache import Cache as DiskCache

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])
SyncF = TypeVar("SyncF", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


class _MemoryCache:
    def __init__(self) -> None:
        self._entries: dict[Any, tuple[Any, Optional[float]]] = {}

    def get(self, key: Any, default: Any = None) -> Any:
        entry = self._entries.get(key)
        if entry is None:
            return default
        value, expires_at = entry
        if expires_at is not None and expires_at <= time.monotonic():
            self._entries.pop(key, None)
            return default
        return value

    def set(self, key: Any, value: Any, expire: Optional[float] = None, **_kwargs: Any) -> bool:
        expires_at = None if expire is None else time.monotonic() + max(0.0, float(expire))
        self._entries[key] = (value, expires_at)
        return True

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count

    def memoize(
        self,
        name: Optional[str] = None,
        typed: bool = False,
        expire: Optional[int] = None,
        tag: Optional[str] = None,
        ignore: Tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            def cache_key(*args: Any, **kwargs: Any) -> Any:
                ignored_positions = {item for item in ignore if isinstance(item, int)}
                ignored_names = {str(item) for item in ignore if isinstance(item, str)}
                filtered_args = tuple(arg for idx, arg in enumerate(args) if idx not in ignored_positions)
                filtered_kwargs = tuple(
                    sorted((key, value) for key, value in kwargs.items() if key not in ignored_names)
                )
                type_payload = None
                if typed:
                    type_payload = (
                        tuple(type(arg).__qualname__ for arg in filtered_args),
                        tuple((key, type(value).__qualname__) for key, value in filtered_kwargs),
                    )
                return (
                    name or f"{func.__module__}.{func.__qualname__}",
                    tag,
                    filtered_args,
                    filtered_kwargs,
                    type_payload,
                )

            setattr(wrapper, "__cache_key__", cache_key)
            return wrapper

        return decorator

    def close(self) -> None:
        return None


class _ResilientCache:
    def __init__(self, cache_path: Path) -> None:
        self._cache_path = cache_path
        self._backend: Any = self._build_backend()

    @property
    def backend_name(self) -> str:
        return "memory" if isinstance(self._backend, _MemoryCache) else "disk"

    def _build_backend(self) -> Any:
        try:
            self._cache_path.mkdir(exist_ok=True, parents=True)
            return DiskCache(str(self._cache_path))
        except Exception as exc:
            if not self._is_storage_error(exc):
                raise
            logger.warning(
                "Persistent cache unavailable at %s (%s). Falling back to in-memory cache.",
                self._cache_path,
                exc,
            )
            return _MemoryCache()

    @staticmethod
    def _is_storage_error(exc: Exception) -> bool:
        return isinstance(exc, (sqlite3.OperationalError, OSError))

    def _switch_to_memory(self, exc: Exception) -> None:
        if isinstance(self._backend, _MemoryCache):
            return
        logger.warning("Persistent cache disabled after storage error: %s", exc)
        close = getattr(self._backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        self._backend = _MemoryCache()

    def _call_backend(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._backend, method_name)
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            if not self._is_storage_error(exc):
                raise
            self._switch_to_memory(exc)
            return getattr(self._backend, method_name)(*args, **kwargs)

    def get(self, key: Any, default: Any = None) -> Any:
        return self._call_backend("get", key, default)

    def set(self, key: Any, value: Any, expire: Optional[float] = None, **kwargs: Any) -> Any:
        return self._call_backend("set", key, value, expire=expire, **kwargs)

    def clear(self) -> Any:
        return self._call_backend("clear")

    def memoize(
        self,
        name: Optional[str] = None,
        typed: bool = False,
        expire: Optional[int] = None,
        tag: Optional[str] = None,
        ignore: Tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._call_backend(
            "memoize",
            name=name,
            typed=typed,
            expire=expire,
            tag=tag,
            ignore=ignore,
        )

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()


cache_dir = Path(".cache")
cache = _ResilientCache(cache_dir)


def async_cache_decorator(
    name: Optional[str] = None,
    typed: bool = False,
    expire: Optional[int] = None,
    tag: Optional[str] = None,
    ignore: Tuple[Any, ...] = (),
) -> Callable[[F], F]:
    """Cache decorator for async functions that works with running event loops"""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_key_func(*args: Any, **kwargs: Any) -> Any:
            return args, kwargs

        cached_key_func: Any = cache.memoize(
            name=name,
            typed=typed,
            expire=expire,
            tag=tag,
            ignore=ignore,
        )(sync_key_func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key_fn = getattr(cached_key_func, "__cache_key__", None)
            if cache_key_fn is None:
                raise AttributeError("Cache memoize wrapper missing __cache_key__ attribute.")

            cache_key = cache_key_fn(*args, **kwargs)
            key_hash = hashlib.md5(pickle.dumps(cache_key)).hexdigest()

            result = cache.get(key_hash)
            if result is None:
                result = await func(*args, **kwargs)
                cache.set(key_hash, result, expire=expire)

            return result

        cache_key_fn = getattr(cached_key_func, "__cache_key__", None)
        if cache_key_fn is None:
            raise AttributeError("Cache memoize wrapper missing __cache_key__ attribute.")
        setattr(wrapper, "__cache_key__", cache_key_fn)
        return cast(F, wrapper)

    return decorator


def sync_cache_decorator(
    name: Optional[str] = None,
    typed: bool = False,
    expire: Optional[int] = None,
    tag: Optional[str] = None,
    ignore: Tuple[Any, ...] = (),
) -> Callable[[SyncF], SyncF]:
    """Cache decorator for synchronous functions"""

    def decorator(func: SyncF) -> SyncF:
        @functools.wraps(func)
        def sync_key_func(*args: Any, **kwargs: Any) -> Any:
            return args, kwargs

        cached_key_func: Any = cache.memoize(
            name=name,
            typed=typed,
            expire=expire,
            tag=tag,
            ignore=ignore,
        )(sync_key_func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key_fn = getattr(cached_key_func, "__cache_key__", None)
            if cache_key_fn is None:
                raise AttributeError("Cache memoize wrapper missing __cache_key__ attribute.")

            cache_key = cache_key_fn(*args, **kwargs)
            key_hash = hashlib.md5(pickle.dumps(cache_key)).hexdigest()

            result = cache.get(key_hash)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key_hash, result, expire=expire)

            return result

        cache_key_fn = getattr(cached_key_func, "__cache_key__", None)
        if cache_key_fn is None:
            raise AttributeError("Cache memoize wrapper missing __cache_key__ attribute.")
        setattr(wrapper, "__cache_key__", cache_key_fn)
        return cast(SyncF, wrapper)

    return decorator
