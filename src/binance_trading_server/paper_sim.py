from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from .quote_provider import fetch_1m_klines_batch, kline_to_quote

if TYPE_CHECKING:
    from .server import BinanceTradingServerEngine


@dataclass
class PaperSimHandle:
    thread: threading.Thread
    stop_event: threading.Event
    poll_seconds: int


def _run_paper_sim_loop(
    engine: BinanceTradingServerEngine,
    stop_event: threading.Event,
    poll_seconds: int,
) -> None:
    while not stop_event.is_set():
        try:
            _tick(engine)
        except Exception as e:
            logger.opt(exception=False).warning(f"paper_sim tick error: {e}")
        stop_event.wait(poll_seconds)


def _tick(engine: BinanceTradingServerEngine) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for account_name, config in engine._registry.items():
        if config["mode"] != "paper":
            continue
        with engine._account_state_guard(account_name, write=False):
            with engine._lock:
                state = engine._load_state_unlocked(account_name, config)
                open_orders = state.get("open_orders", [])
                if not open_orders:
                    continue
                symbols = sorted({o["symbol"] for o in open_orders})

        klines = fetch_1m_klines_batch(symbols)
        if not klines:
            continue

        with engine._account_state_guard(account_name):
            with engine._lock:
                state = engine._load_state_unlocked(account_name, config)
                for sym, kline in klines.items():
                    engine._store_quote_unlocked(state, sym, kline_to_quote(sym, kline))

        filled = engine.attempt_open_order_fills(account_name, klines=klines)
        if filled:
            filled_ids = [o["id"] for o in filled]
            logger.info(f"paper_sim filled {len(filled)} orders for {account_name}: {filled_ids}")
            results[account_name] = filled_ids
    return results


def start_paper_sim(engine: BinanceTradingServerEngine, *, poll_seconds: int | None = None) -> PaperSimHandle:
    resolved = poll_seconds or engine.settings.sim_poll_seconds
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_run_paper_sim_loop,
        args=(engine, stop_event, resolved),
        name=f"binance-paper-sim-{id(engine):x}",
        daemon=True,
    )
    thread.start()
    return PaperSimHandle(thread=thread, stop_event=stop_event, poll_seconds=resolved)


def stop_paper_sim(handle: PaperSimHandle, timeout: float = 2.0) -> None:
    handle.stop_event.set()
    handle.thread.join(timeout=timeout)
