"""Generic background-refresh registry for periodic tasks.

Used by ``trading_server`` and ``binance_trading_server`` to keep
price quotes fresh via a background thread that periodically calls a
user-supplied refresh function.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass
class BackgroundRefreshHandle(Generic[_T]):  # noqa: UP046
    """Tracks a single running background-refresh thread."""

    engine: _T
    thread: threading.Thread
    stop_event: threading.Event = field(default_factory=threading.Event)
    poll_seconds: int = 60


class BackgroundRefreshRegistry(Generic[_T]):  # noqa: UP046
    """Manages background-refresh threads keyed by engine instance.

    Usage::

        registry = BackgroundRefreshRegistry[MyEngine](
            thread_name_prefix="my-refresh",
            min_poll_seconds=1,
        )
        thread = registry.ensure(
            engine,
            refresh_fn=lambda e: e.refresh(),
            default_poll_seconds=60,
        )
        # ... later ...
        registry.stop(engine, timeout=2.0)
    """

    def __init__(
        self,
        *,
        thread_name_prefix: str = "bg-refresh",
        min_poll_seconds: int = 1,
    ) -> None:
        self._prefix = thread_name_prefix
        self._min_poll = min_poll_seconds
        self.lock = threading.Lock()
        self.refreshers: dict[int, BackgroundRefreshHandle[_T]] = {}

    # -- public API ----------------------------------------------------------

    def ensure(
        self,
        engine: _T,
        *,
        refresh_fn: Callable[[_T], object],
        default_poll_seconds: int = 60,
        poll_seconds: int | None = None,
    ) -> threading.Thread:
        """Start (or return existing) background-refresh thread for *engine*."""
        key = id(engine)
        with self.lock:
            handle = self.refreshers.get(key)
            if handle is not None and handle.thread.is_alive():
                return handle.thread

            interval = max(self._min_poll, poll_seconds or default_poll_seconds)
            stop_event = threading.Event()
            handle = BackgroundRefreshHandle(
                engine=engine,
                thread=threading.Thread(target=lambda: None, daemon=True),
                stop_event=stop_event,
                poll_seconds=interval,
            )

            def _loop() -> None:
                while not stop_event.is_set():
                    try:
                        refresh_fn(engine)
                    except Exception:
                        logger.exception("%s: refresh failed", self._prefix)
                    stop_event.wait(timeout=interval)

            t = threading.Thread(
                target=_loop,
                name=f"{self._prefix}-{key}",
                daemon=True,
            )
            handle.thread = t
            self.refreshers[key] = handle
            t.start()
            return t

    def stop(self, engine: _T | None = None, *, timeout: float = 1.0) -> None:
        """Stop background refresh for *engine* (or all if *None*)."""
        with self.lock:
            if engine is not None:
                key = id(engine)
                handle = self.refreshers.pop(key, None)
                targets = [handle] if handle else []
            else:
                targets = list(self.refreshers.values())
                self.refreshers.clear()

        for handle in targets:
            handle.stop_event.set()
            handle.thread.join(timeout=timeout)
