from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    import pandas as pd

from marketsimulator.logging_utils import logger
from marketsimulator.state import get_state

from .forecasting_utils import export_price_history

ALLOW_MOCK_ANALYTICS = os.getenv("MARKETSIM_ALLOW_MOCK_ANALYTICS", "0").lower() in {"1", "true", "yes"}

if ALLOW_MOCK_ANALYTICS:
    from . import predict_stock_forecasting_mock as fallback_module  # pragma: no cover
else:
    fallback_module = None  # type: ignore[assignment]

try:
    _real_module = importlib.import_module("predict_stock_forecasting")
    _REAL_MODULE_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised when Kronos deps missing
    _real_module = None
    _REAL_MODULE_ERROR = exc
    if not ALLOW_MOCK_ANALYTICS:
        raise RuntimeError(
            "Failed to import the real predict_stock_forecasting module. "
            "Install its dependencies or set MARKETSIM_ALLOW_MOCK_ANALYTICS=1 to fall back "
            "to the lightweight simulator forecasts."
        ) from exc
    logger.warning(
        "[sim] Unable to import real predict_stock_forecasting module (%s); "
        "falling back to deterministic simulator forecasts.",
        exc,
    )


def _run_real_predictions(
    pred_name: str = "",
    retrain: bool = False,
    alpaca_wrapper=None,
) -> Optional[pd.DataFrame]:
    if _real_module is None:
        return None

    state = get_state()
    base_dir: Path = _real_module.base_dir  # type: ignore[attr-defined]
    horizon = getattr(_real_module, "FORECAST_HORIZON", 7)
    simulator_dir = base_dir / "data" / "_simulator"

    export_price_history(state, simulator_dir, padding=horizon * 2)

    try:
        return _real_module.make_predictions(
            input_data_path="_simulator",
            pred_name=pred_name,
            retrain=retrain,
            alpaca_wrapper=alpaca_wrapper,
        )
    except Exception as exc:  # pragma: no cover - depends on external deps
        logger.warning(
            "[sim] Real forecasting pipeline failed (%s); reverting to fallback predictions.",
            exc,
        )
        return None


def make_predictions(
    input_data_path: Optional[str] = None,
    pred_name: str = "",
    retrain: bool = False,
    alpaca_wrapper=None,
) -> pd.DataFrame:
    """
    Proxy to the real ``predict_stock_forecasting`` module when available,
    otherwise fall back to a lightweight deterministic implementation.
    """
    if alpaca_wrapper is None:
        try:
            alpaca_wrapper = importlib.import_module("alpaca_wrapper")
        except Exception:  # pragma: no cover - safeguard if import fails
            alpaca_wrapper = None

    real_results = _run_real_predictions(
        pred_name=pred_name,
        retrain=retrain,
        alpaca_wrapper=alpaca_wrapper,
    )
    if real_results is not None:
        return real_results

    if not ALLOW_MOCK_ANALYTICS or fallback_module is None:
        if _REAL_MODULE_ERROR is not None:
            raise RuntimeError(
                "Real forecasting pipeline failed and mock analytics are disabled. "
                "Set MARKETSIM_ALLOW_MOCK_ANALYTICS=1 if you want to fall back to the simulator."
            ) from _REAL_MODULE_ERROR
        raise RuntimeError(
            "Real forecasting pipeline returned no results and mock analytics are disabled."
        )

    return fallback_module.make_predictions(  # pragma: no cover - requires explicit opt-in
        input_data_path=input_data_path,
        pred_name=pred_name,
        retrain=retrain,
        alpaca_wrapper=alpaca_wrapper,
    )
