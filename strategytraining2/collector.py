from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

DATASET_DIR = Path("strategytraining2/datasets")
DATASET_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PATH = DATASET_DIR / "trade_strategy_daily.jsonl"
_LOCK = threading.Lock()


def _coerce_value(value: Any) -> Any:
    if isinstance(value, (int, float, str)) or value is None:
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, np.generic):
        return value.astype(float)
    if isinstance(value, (list, tuple)):
        return [_coerce_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_value(v) for k, v in value.items()}
    return str(value)


def _sanitize_row(symbol: str, data: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
    fields = {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "strategy": data.get("strategy"),
        "side": data.get("side"),
        "mode": data.get("mode"),
        "trade_mode": data.get("trade_mode"),
        "gate_config": data.get("gate_config"),
        "trade_blocked": data.get("trade_blocked"),
        "avg_return": data.get("avg_return"),
        "simple_return": data.get("simple_return"),
        "takeprofit_return": data.get("takeprofit_return"),
        "highlow_return": data.get("highlow_return"),
        "maxdiff_return": data.get("maxdiff_return"),
        "maxdiffalwayson_return": data.get("maxdiffalwayson_return"),
        "pctdiff_return": data.get("pctdiff_return"),
        "rolling_sharpe": data.get("rolling_sharpe"),
        "rolling_sortino": data.get("rolling_sortino"),
        "rolling_ann_return": data.get("rolling_ann_return"),
        "daily_pnl": data.get("daily_pnl"),
        "capital": data.get("capital"),
        "predicted_movement": data.get("predicted_movement"),
        "predicted_close": data.get("predicted_close"),
        "predicted_high": data.get("predicted_high"),
        "predicted_low": data.get("predicted_low"),
        "forecast_move_pct": data.get("forecast_move_pct"),
        "forecast_volatility_pct": data.get("forecast_volatility_pct"),
        "day_class": data.get("day_class"),
        "edge_strength": data.get("edge_strength"),
        "chronos_quantiles": data.get("chronos2_quantile_levels"),
    }
    return {key: _coerce_value(value) for key, value in fields.items() if value is not None}


def log_strategy_snapshot(symbol: str, data: Dict[str, Any], timestamp: datetime) -> None:
    row = _sanitize_row(symbol, data, timestamp)
    if not row:
        return
    payload = json.dumps(row, separators=(",", ":"))
    with _LOCK:
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DATASET_PATH.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
