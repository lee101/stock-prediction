from __future__ import annotations

import importlib
import logging
import os
import time
from contextlib import nullcontext
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional

from falmarket.shared_logger import get_logger, log_timing
from src.runtime_imports import setup_src_imports

LOG = get_logger("falmarket.runner", logging.INFO)
# fal sdk NEEDS this as setup() is only way to get these imported in fal
try:
    import numpy as np
    import pandas as pd
    import torch
except ImportError:
    # These will be injected via setup_imports() in FAL environment
    torch = None
    np = None
    pd = None


def setup_training_imports(
    torch_module: Optional[ModuleType],
    numpy_module: Optional[ModuleType],
    pandas_module: Optional[ModuleType] = None,
) -> None:
    """Register shared heavy dependencies for the simulation runtime."""

    global torch, np, pd
    if torch_module is not None:
        torch = torch_module
    if numpy_module is not None:
        np = numpy_module
    if pandas_module is not None:
        pd = pandas_module
    setup_src_imports(torch, np, pd)


def _configure_logging(compact: bool) -> Optional[int]:
    if not compact:
        return None

    from loguru import logger as loguru_logger  # type: ignore

    loguru_logger.remove()
    handler_id = loguru_logger.add(
        lambda message: print(message, flush=True),
        level="WARNING",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    return handler_id


def _restore_logging(handler_id: Optional[int]) -> None:
    if handler_id is None:
        return
    from loguru import logger as loguru_logger  # type: ignore

    loguru_logger.remove(handler_id)


def _summarise_pick(data: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "avg_return",
        "confidence",
        "predicted_return",
        "expected_return",
        "expected_profit",
    ]
    summary = {}
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            summary[key] = float(value)
    return summary


def simulate_trading(
    *,
    symbols: Iterable[str],
    steps: int,
    step_size: int,
    initial_cash: float,
    top_k: int,
    kronos_only: bool,
    compact_logs: bool,
) -> Dict[str, Any]:
    """Run the trade_stock_e2e loop inside the fal worker and return results."""

    if torch is None or np is None:
        raise RuntimeError("Torch and NumPy must be configured via setup_training_imports before simulate_trading.")

    symbols_list = list(symbols)
    symbols_unique = list(dict.fromkeys(symbols_list))

    LOG.info(
        "simulate_trading symbols=%s steps=%s step_size=%s top_k=%s kronos_only=%s compact_logs=%s",
        symbols_unique,
        steps,
        step_size,
        top_k,
        kronos_only,
        compact_logs,
    )

    start = time.time()
    handler_id = _configure_logging(compact_logs)
    timeline: List[Dict[str, Any]] = []

    from marketsimulator.environment import activate_simulation

    def _analysis_context():
        inference_ctor = getattr(torch, "inference_mode", None)
        if callable(inference_ctor):
            return inference_ctor()
        no_grad_ctor = getattr(torch, "no_grad", None)
        if callable(no_grad_ctor):
            return no_grad_ctor()
        return nullcontext()

    step_chunk = max(1, int(os.getenv("MARKETSIM_SIM_ANALYSIS_CHUNK", "0") or 0))

    summary: Dict[str, Any] = {}

    with log_timing(LOG, "activate_simulation context"):
        activate_kwargs = {
            "symbols": symbols_list,
            "initial_cash": initial_cash,
            "use_mock_analytics": False,
            "force_kronos": kronos_only,
        }
        with activate_simulation(**activate_kwargs) as controller:
            trade_module = importlib.import_module("trade_stock_e2e")
            analyze_symbols = getattr(trade_module, "analyze_symbols")
            log_trading_plan = getattr(trade_module, "log_trading_plan", lambda *args, **kwargs: None)
            manage_positions = getattr(trade_module, "manage_positions")
            release_model_resources = getattr(trade_module, "release_model_resources", lambda: None)

            previous_picks: Dict[str, Dict[str, Any]] = {}
            for step in range(max(1, steps)):
                timestamp = controller.current_time()
                analyzed: Dict[str, Any] = {}
                with _analysis_context():
                    if step_chunk > 0:
                        for idx in range(0, len(symbols_unique), step_chunk):
                            batch = symbols_unique[idx : idx + step_chunk]
                            if not batch:
                                continue
                            batch_result = analyze_symbols(batch)
                            analyzed.update(batch_result)
                    else:
                        analyzed = analyze_symbols(symbols_unique)

                ordered_symbols = list(analyzed.items())
                current = {
                    symbol: info
                    for symbol, info in ordered_symbols[: max(1, top_k)]
                    if isinstance(info, dict) and info.get("avg_return", 0) > 0
                }
                if current:
                    log_trading_plan(current, f"SIM-STEP-{step + 1}")
                manage_positions(current, previous_picks, analyzed)

                timeline.append(
                    {
                        "step": step + 1,
                        "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                        "picked": {symbol: _summarise_pick(data) for symbol, data in current.items()},
                        "analyzed_count": len(analyzed),
                    }
                )
                previous_picks = current
                controller.advance_steps(max(1, step_size))
                # if torch.cuda.is_available():
                #     try:
                #         torch.cuda.synchronize()
                #     except Exception:
                #         pass

            summary = controller.summary()
            release_model_resources()

            if steps >= 1:
                LOG.info(
                    "simulate_trading progressed steps=%s last_pick_count=%s analyzed_last=%s",
                    steps,
                    len(timeline[-1]["picked"]) if timeline else 0,
                    timeline[-1]["analyzed_count"] if timeline else 0,
                )

    _restore_logging(handler_id)
    duration = time.time() - start

    LOG.info(
        "simulate_trading completed run_seconds=%.3f final_cash=%.2f final_equity=%.2f",
        duration,
        float(summary.get("cash", 0.0)),
        float(summary.get("equity", 0.0)),
    )

    return {
        "timeline": timeline,
        "summary": {
            "cash": float(summary.get("cash", 0.0)),
            "equity": float(summary.get("equity", 0.0)),
            "positions": summary.get("positions", {}),
            "initial_cash": float(initial_cash),
        },
        "run_seconds": duration,
    }
