"""Prod trade log — per-session JSONL with sim-realism residuals.

Goal: measure the gap between what the backtest sim thinks the trader
will realise and what Alpaca actually fills at. Two residuals we care
about:

    slippage_vs_last_close_bps
        10000 * (fill_px - last_close) / last_close
        Our sim's `open_px = last_close * (1 + fill_buffer_bps / 10000)`
        assumption holds when this residual ≈ fill_buffer_bps.

    realized_pnl_vs_sim_pct
        Computed session-over-session from Alpaca equity deltas.

Every session emits one JSONL file under ``analysis/xgb_live_trade_log/``
named ``YYYY-MM-DD.jsonl``. Each line is one event. Schema is intentionally
loose — append fields freely; readers should tolerate missing keys.

Event types
-----------
    session_start    date, mode, paper, equity_pre, args
    scored           n_candidates, top20 [{symbol,score,per_seed_scores,
                       last_close,spread_bps}], filter_stats
    pick             symbol, score, last_close, spread_bps, rank
    hold             held, picks  [hold-through only, when unchanged]
    rotate           to_sell, to_buy, keep                 [hold-through]
    buy_submitted    symbol, qty, expected_price, order_id
    buy_filled       symbol, fill_price, slippage_bps_vs_last_close,
                       fill_source ("fill"|"last_close")
    sell_submitted   symbol, qty, expected_price, order_id
    sell_filled      symbol, fill_price [best-effort; may be absent]
    session_end      equity_post, session_pnl_abs, session_pnl_pct

The file is opened append-binary and fsync'd on each write — resilient
to process crash / supervisor restart.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO / "analysis" / "xgb_live_trade_log"


class TradeLogger:
    """Append-only JSONL session logger. Safe to use when disabled=True."""

    def __init__(self, log_dir: Path | str = DEFAULT_LOG_DIR,
                 disabled: bool = False,
                 session_date: date | None = None):
        self.disabled = disabled
        self.log_dir = Path(log_dir)
        self._session_date = session_date or date.today()
        if not disabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._path = self.log_dir / f"{self._session_date.isoformat()}.jsonl"
        else:
            self._path = None

    @property
    def path(self) -> Path | None:
        return self._path

    def log(self, event: str, **fields: Any) -> None:
        if self.disabled or self._path is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        try:
            line = json.dumps(record, default=_json_default) + "\n"
        except TypeError:
            line = json.dumps(_coerce(record), default=_json_default) + "\n"
        try:
            with open(self._path, "ab") as fh:
                fh.write(line.encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _coerce(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def slippage_bps(fill_price: float | None, reference: float | None) -> float | None:
    """Return 10_000 * (fill - ref) / ref; None on bad inputs."""
    if fill_price is None or reference is None:
        return None
    try:
        fp = float(fill_price)
        rp = float(reference)
    except (TypeError, ValueError):
        return None
    if not (rp > 0 and fp > 0):
        return None
    return 10_000.0 * (fp - rp) / rp
