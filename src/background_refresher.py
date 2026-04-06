from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable, Generic, TypeVar
import weakref

EngineT = TypeVar("EngineT")


@dataclass
class BackgroundRefreshHandle:
    thread: threading.Thread
    stop_event: threading.Event
    stopped_event: threading.Event
    owners: int = 1
    restart_requested: bool = False


class BackgroundRefreshRegistry(Generic[EngineT]):
    def __init__(self, *, thread_name_prefix: str, min_poll_seconds: int) -> None:
        self._thread_name_prefix = thread_name_prefix
        self._min_poll_seconds = max(int(min_poll_seconds), 1)
        self._lock = threading.Lock()
        self._refreshers: weakref.WeakKeyDictionary[EngineT, BackgroundRefreshHandle] = weakref.WeakKeyDictionary()

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    @property
    def refreshers(self) -> weakref.WeakKeyDictionary[EngineT, BackgroundRefreshHandle]:
        return self._refreshers

    def _run(
        self,
        engine: EngineT,
        handle: BackgroundRefreshHandle,
        refresh_fn: Callable[[EngineT], object],
        poll_seconds: int,
    ) -> None:
        try:
            while True:
                while not handle.stop_event.is_set():
                    try:
                        refresh_fn(engine)
                    except Exception:
                        pass
                    if handle.stop_event.wait(poll_seconds):
                        break
                with self._lock:
                    if handle.restart_requested:
                        handle.restart_requested = False
                        handle.stop_event.clear()
                        continue
                    handle.stopped_event.set()
                    return
        finally:
            if not handle.stopped_event.is_set():
                handle.stopped_event.set()

    def ensure(
        self,
        engine: EngineT,
        *,
        refresh_fn: Callable[[EngineT], object],
        default_poll_seconds: int,
        poll_seconds: int | None = None,
    ) -> threading.Thread:
        resolved_poll_seconds = default_poll_seconds if poll_seconds is None else poll_seconds
        with self._lock:
            handle = self._refreshers.get(engine)
            if handle is not None and handle.thread.is_alive():
                if not handle.stopped_event.is_set():
                    if handle.stop_event.is_set():
                        handle.restart_requested = True
                    handle.owners += 1
                    return handle.thread
            stop_event = threading.Event()
            stopped_event = threading.Event()
            handle: BackgroundRefreshHandle

            def _runner() -> None:
                self._run(
                    engine,
                    handle,
                    refresh_fn,
                    max(int(resolved_poll_seconds), self._min_poll_seconds),
                )

            thread = threading.Thread(
                target=_runner,
                name=f"{self._thread_name_prefix}-{id(engine):x}",
                daemon=True,
            )
            handle = BackgroundRefreshHandle(
                thread=thread,
                stop_event=stop_event,
                stopped_event=stopped_event,
            )
            self._refreshers[engine] = handle
            thread.start()
            return thread

    def stop(self, engine: EngineT | None = None, *, timeout: float = 1.0) -> None:
        with self._lock:
            if engine is None:
                entries = list(self._refreshers.items())
            else:
                handle = self._refreshers.get(engine)
                entries = [(engine, handle)] if handle is not None else []
            to_join: list[tuple[EngineT, BackgroundRefreshHandle]] = []
            for current_engine, handle in entries:
                if engine is None:
                    handle.owners = 0
                else:
                    handle.owners = max(handle.owners - 1, 0)
                if handle.owners == 0:
                    handle.restart_requested = False
                    handle.stop_event.set()
                    to_join.append((current_engine, handle))
        for _, handle in to_join:
            handle.thread.join(timeout=timeout)
        with self._lock:
            for current_engine, handle in to_join:
                current = self._refreshers.get(current_engine)
                if current is handle and handle.owners == 0 and not handle.thread.is_alive():
                    self._refreshers.pop(current_engine, None)
