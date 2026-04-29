#!/usr/bin/env python3
"""XGBoost daily open-to-close live trader.

Strategy: each trading morning, score all stocks → buy top-N at open → sell at close.

Session flow
------------
1. Pre-open  (~9:25 ET): load yesterday's OHLCV, compute features, score 846 stocks
2. At open   (9:30 ET): place priced BUY limits for top-N candidates
3. Near close (15:50 ET): place priced SELL limits for all held positions
4. Post-close: log results

Safety
------
- Imports src.alpaca_singleton for live-writer singleton guard
- Only one live process allowed (fcntl lock on alpaca_live_writer.lock)
- ALP_PAPER=1 runs in paper mode (bypasses singleton)
- Reads API keys from env_real.py

Usage
-----
  # Paper mode (safe for testing):
  ALP_PAPER=1 python -m xgbnew.live_trader --top-n 2

  # Live mode:
  python -m xgbnew.live_trader --top-n 2 --live --allocation 0.25

  # Dry run (score only, no orders):
  python -m xgbnew.live_trader --top-n 2 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import _load_symbol_csv
from xgbnew.features import (
    DAILY_DISPERSION_FEATURE_COLS,
    DAILY_FEATURE_COLS,
    DAILY_RANK_FEATURE_COLS,
    FM_LATENT_FEATURE_COLS,
    LIVE_SUPPORTED_FEATURE_COLS,
    add_cross_sectional_dispersion,
    add_cross_sectional_ranks,
    build_features_for_symbol,
    evaluate_cross_sectional_regime_gate,
)
from xgbnew.backtest import (
    _allocation_weights,
    _build_regime_flags,
    _build_vol_scale,
    _inv_vol_pick_scale,
)
from xgbnew.model import XGBStockModel
from xgbnew.trade_log import TradeLogger, slippage_bps

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

MARKET_OPEN  = (9, 30)   # HH, MM ET
MARKET_CLOSE = (15, 50)  # sell by 15:50 (10 min before 4pm close)
DEFAULT_MODEL_PATH = REPO / "analysis/xgbnew_daily/live_model.pkl"
DEFAULT_LIVE_BAR_BATCH_SIZE = 200
MIN_FRACTIONAL_QTY = 0.0001
CRYPTO_SYMBOL_COMPACT = {"BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "LTCUSD", "AVAXUSD"}


# ── Alpaca client ─────────────────────────────────────────────────────────────

def _build_trading_client(paper: bool):
    """Build Alpaca TradingClient with correct keys."""
    try:
        import importlib
        env_real = importlib.import_module("env_real")
    except ImportError as exc:
        raise RuntimeError("env_real.py not found — cannot get Alpaca keys") from exc

    if paper:
        key_id = getattr(env_real, "ALP_KEY_ID_PAPER", getattr(env_real, "ALP_KEY_ID", ""))
        secret = getattr(env_real, "ALP_SECRET_KEY_PAPER", getattr(env_real, "ALP_SECRET_KEY", ""))
    else:
        key_id = getattr(env_real, "ALP_KEY_ID_PROD", "")
        secret = getattr(env_real, "ALP_SECRET_KEY_PROD", "")

    from alpaca.trading.client import TradingClient
    return TradingClient(key_id, secret, paper=paper)


def _to_alpaca_symbol(sym: str) -> str:
    """Translate local dash-form tickers to Alpaca's dot form.

    Our symbol lists use ``BRK-B`` / ``BF-B`` but Alpaca's trading and data
    APIs require ``BRK.B`` / ``BF.B``. A single ``-`` in a ticker always
    represents a share class on the real exchange — the translation is
    unambiguous for US equities.
    """
    if not sym:
        return sym
    return sym.replace("-", ".") if "-" in sym else sym


def _normalize_alpaca_symbol(sym: str) -> str:
    return str(sym or "").upper().replace("/", "")


def _is_crypto_position_symbol(sym: str) -> bool:
    compact = _normalize_alpaca_symbol(sym)
    return compact in CRYPTO_SYMBOL_COMPACT


def _submit_market_order(client, *, symbol: str, qty: float, side: str):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    side_val = OrderSide.BUY if side == "buy" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=_to_alpaca_symbol(symbol),
        qty=round(float(qty), 4),
        side=side_val,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(req)


def _submit_limit_order(client, *, symbol: str, qty: float, side: str, limit_price: float):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest

    side_val = OrderSide.BUY if side == "buy" else OrderSide.SELL
    req = LimitOrderRequest(
        symbol=_to_alpaca_symbol(symbol),
        qty=round(float(qty), 4),
        side=side_val,
        time_in_force=TimeInForce.DAY,
        limit_price=round(float(limit_price), 2),
    )
    return client.submit_order(req)


def _latest_stock_bid_ask(symbol: str) -> tuple[float, float]:
    """Best-effort latest stock quote for explicit-price order placement."""
    try:
        import alpaca_wrapper as aw

        quote = aw.latest_data(_to_alpaca_symbol(symbol))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Quote lookup failed for %s: %s", symbol, exc)
        return 0.0, 0.0

    try:
        bid_price = float(getattr(quote, "bid_price", 0.0) or 0.0)
    except (TypeError, ValueError):
        bid_price = 0.0
    try:
        ask_price = float(getattr(quote, "ask_price", 0.0) or 0.0)
    except (TypeError, ValueError):
        ask_price = 0.0
    return bid_price, ask_price


def _stock_limit_price_near_market(
    symbol: str,
    *,
    side: str,
    reference_price: float,
    aggressiveness_bps: float,
) -> float:
    """Return an explicit stock limit price bounded near the touch.

    Buys use the current ask plus a small bounded buffer so the order is
    marketable but never unbounded like a true market order. Sells use the bid
    minus the same style of bounded buffer.
    """
    try:
        reference_price = float(reference_price or 0.0)
    except (TypeError, ValueError):
        reference_price = 0.0
    bps = max(float(aggressiveness_bps or 0.0), 0.0) / 10_000.0
    bid_price, ask_price = _latest_stock_bid_ask(symbol)
    side_norm = str(side or "").strip().lower()

    if side_norm == "buy":
        if ask_price > 0:
            return ask_price * (1.0 + bps)
        if bid_price > 0:
            return bid_price * (1.0 + bps)
        return reference_price

    if side_norm == "sell":
        if bid_price > 0:
            return max(bid_price * (1.0 - bps), 0.01)
        if ask_price > 0:
            return max(ask_price * (1.0 - bps), 0.01)
        return reference_price

    return reference_price


def _poll_filled_avg_price(client, order_id: str, *, timeout_s: float = 30.0) -> float | None:
    """Poll an order until it reports a filled_avg_price or timeout.

    Returns the float fill price (None on timeout/no fill). Used to feed the
    death-spiral guard's buy-price memory so future sells are vetted against
    the actual cost basis rather than a pre-fill estimate.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            o = client.get_order_by_id(order_id)
        except Exception:
            time.sleep(1.0)
            continue
        price = getattr(o, "filled_avg_price", None)
        if price is not None:
            try:
                p = float(price)
                if p > 0:
                    return p
            except (TypeError, ValueError):
                pass
        status = str(getattr(o, "status", "")).lower()
        if status in ("canceled", "rejected", "expired"):
            return None
        time.sleep(1.0)
    return None


def _get_account(client):
    return client.get_account()


def _is_today_trading_day(client, now: datetime | None = None) -> tuple[bool, str]:
    """Query Alpaca's market clock and return (is_trading_day, reason).

    Uses the broker's own calendar — covers weekends AND holidays without
    needing pandas_market_calendars. Reason string is for logging.

    The check is: if `is_open` is True, obviously a trading day. Otherwise,
    if `next_open.date()` equals today (ET), the market will open today →
    trading day. Anything else (weekend, holiday) is not a trading day.
    """
    now_et_date = (now or datetime.now(timezone.utc)).astimezone(ET).date()
    try:
        clock = client.get_clock()
    except Exception as exc:
        return True, f"clock_query_failed: {exc} (assuming trading day)"
    is_open = bool(getattr(clock, "is_open", False))
    next_open = getattr(clock, "next_open", None)
    next_open_date = None
    if next_open is not None:
        try:
            next_open_date = next_open.astimezone(ET).date()
        except Exception:
            try:
                next_open_date = next_open.date()
            except Exception:
                next_open_date = None
    if is_open:
        return True, f"market_open (is_open=true)"
    if next_open_date == now_et_date:
        return True, f"pre-open (next_open={next_open_date})"
    return False, f"closed (next_open={next_open_date}, today={now_et_date})"


def _get_positions(client) -> dict[str, float]:
    """Return {symbol: qty} for all open positions."""
    positions = client.get_all_positions()
    return {str(p.symbol): float(p.qty) for p in positions}


def _get_position_details(client) -> dict[str, dict]:
    """Return {symbol: {qty, current_price, avg_entry_price}} for open positions.

    current_price is the price the guard consults before submitting a sell.
    """
    positions = client.get_all_positions()
    out: dict[str, dict] = {}
    for p in positions:
        sym = str(p.symbol)
        if _is_crypto_position_symbol(sym):
            continue
        try:
            cur = float(getattr(p, "current_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            cur = 0.0
        try:
            entry = float(getattr(p, "avg_entry_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            entry = 0.0
        out[sym] = {
            "qty": float(p.qty),
            "current_price": cur,
            "avg_entry_price": entry,
        }
    return out


def _previous_trading_day(day: date) -> date:
    return (pd.Timestamp(day) - BDay(1)).date()


def _session_date_et(now: datetime | None = None) -> date:
    return (now or datetime.now(timezone.utc)).astimezone(ET).date()


def _expected_latest_daily_bar_date(now: datetime | None = None) -> date:
    now_et = (now or datetime.now(timezone.utc)).astimezone(ET)
    market_open = now_et.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    if now_et >= market_open:
        return now_et.date()
    return _previous_trading_day(now_et.date())


def _expected_latest_completed_daily_bar_date(now: datetime | None = None) -> date:
    """Return the latest completed daily close expected to be available for live gates."""
    now_et = (now or datetime.now(timezone.utc)).astimezone(ET)
    today = now_et.date()
    if today.weekday() >= 5:
        return _previous_trading_day(today)
    close_ready = now_et.replace(hour=17, minute=0, second=0, microsecond=0)
    if now_et >= close_ready:
        return today
    return _previous_trading_day(today)


def _latest_daily_bar_is_fresh(df: pd.DataFrame, *, now: datetime | None = None) -> bool:
    if df is None or len(df) == 0 or "timestamp" not in df.columns:
        return False
    latest = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if latest.empty:
        return False
    latest_date = latest.max().tz_convert(ET).date()
    expected_latest = _expected_latest_daily_bar_date(now=now)
    return latest_date >= expected_latest


def _iter_symbol_batches(symbols: list[str], batch_size: int = DEFAULT_LIVE_BAR_BATCH_SIZE):
    size = max(int(batch_size), 1)
    for i in range(0, len(symbols), size):
        yield symbols[i : i + size]


def _get_latest_bars(
    symbols: list[str],
    n_days: int = 5,
    *,
    batch_size: int = DEFAULT_LIVE_BAR_BATCH_SIZE,
) -> dict[str, pd.DataFrame]:
    """Fetch last N calendar days of daily bars from Alpaca data API.

    Data API is read-only; falls through paper → prod keys so it still
    works when paper credentials are revoked (which blocks trading but
    shouldn't block scoring/dry-run).
    """
    try:
        import importlib
        env_real = importlib.import_module("env_real")
        from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame
        from alpaca.data.enums import DataFeed
        candidate_keys = [
            (getattr(env_real, "ALP_KEY_ID_PAPER", ""),
             getattr(env_real, "ALP_SECRET_KEY_PAPER", "")),
            (getattr(env_real, "ALP_KEY_ID_PROD", ""),
             getattr(env_real, "ALP_SECRET_KEY_PROD", "")),
            (getattr(env_real, "ALP_KEY_ID", ""),
             getattr(env_real, "ALP_SECRET_KEY", "")),
        ]
        data_client = None
        for key_id, secret in candidate_keys:
            if not key_id or not secret:
                continue
            try:
                test_client = StockHistoricalDataClient(key_id, secret)
                test_req = StockBarsRequest(
                    symbol_or_symbols="AAPL",
                    timeframe=TimeFrame.Day,
                    start=datetime.now(timezone.utc) - timedelta(days=7),
                    end=datetime.now(timezone.utc),
                    feed=DataFeed.IEX,
                )
                test_client.get_stock_bars(test_req)
                data_client = test_client
                break
            except Exception:
                continue
        if data_client is None:
            raise RuntimeError("no working Alpaca data credentials in env_real.py")
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(n_days * 2, 14))  # extra buffer for holidays
        # Alpaca uses BRK.B / BF.B; symbol list uses BRK-B / BF-B. Translate
        # on the wire then map back when stitching results.
        api_to_local = {_to_alpaca_symbol(s): s for s in symbols}

        result = {}
        for batch in _iter_symbol_batches(symbols, batch_size=batch_size):
            api_batch = [_to_alpaca_symbol(s) for s in batch]
            req = StockBarsRequest(
                symbol_or_symbols=api_batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(req)
            for api_sym in api_batch:
                if api_sym in bars.data:
                    sym = api_to_local.get(api_sym, api_sym)
                    rows = [
                        {"timestamp": b.timestamp, "open": b.open, "high": b.high,
                         "low": b.low, "close": b.close, "volume": b.volume}
                        for b in bars.data[api_sym]
                    ]
                    if rows:
                        df = pd.DataFrame(rows)
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                        result[sym] = df
        return result
    except Exception as exc:
        logger.warning("Alpaca data API fetch failed: %s — will use local CSVs only", exc)
        return {}


# ── Feature computation for today ─────────────────────────────────────────────

def _extend_with_live_bars(
    local_df: pd.DataFrame,
    live_bars: dict[str, pd.DataFrame],
    symbol: str,
) -> pd.DataFrame:
    """Append any live bars newer than what's in local_df."""
    if symbol not in live_bars:
        return local_df
    new_df = live_bars[symbol]
    if local_df is None or len(local_df) == 0:
        return new_df
    latest_local = local_df["timestamp"].max()
    new_rows = new_df[new_df["timestamp"] > latest_local]
    if len(new_rows) == 0:
        return local_df
    combined = pd.concat([local_df, new_rows], ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)


def score_all_symbols(
    symbols: list[str],
    data_root: Path,
    model: XGBStockModel | list[XGBStockModel],
    live_bars: dict[str, pd.DataFrame] | None = None,
    min_dollar_vol: float = 5e6,
    max_spread_bps: float = 30.0,
    min_vol_20d: float = 0.0,
    max_vol_20d: float = 0.0,
    max_ret_20d_rank_pct: float = 1.0,
    min_ret_5d_rank_pct: float = 0.0,
    score_uncertainty_penalty: float = 0.0,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Score all symbols for today's open-to-close trade.

    ``model`` may be a single ``XGBStockModel`` or a list of models — in the
    latter case per-model ``predict_scores`` are averaged (prob-mean blend,
    matches ``xgbnew/eval_pretrained.py`` default).

    Returns DataFrame with columns [symbol, score, spread_bps, last_close]
    sorted by score descending.
    """
    if live_bars is None:
        live_bars = {}

    models: list[XGBStockModel] = model if isinstance(model, list) else [model]
    if not models:
        raise ValueError("score_all_symbols: no models provided")

    feature_rows = []

    for sym in symbols:
        local_df = _load_symbol_csv(sym, data_root)
        if local_df is None:
            continue
        df = _extend_with_live_bars(local_df, live_bars, sym)
        if len(df) < 60:
            continue
        if not _latest_daily_bar_is_fresh(df, now=now):
            continue

        feat = build_features_for_symbol(df, symbol=sym)
        # Get the LAST row — that's features for trading today
        last = feat.dropna(subset=DAILY_FEATURE_COLS[:5])
        if len(last) == 0:
            continue
        last_row = last.iloc[[-1]].copy()
        if "symbol" not in last_row.columns:
            last_row["symbol"] = sym

        # Add Chronos2 zeros
        for col in ["chronos_oc_return", "chronos_cc_return",
                    "chronos_pred_range", "chronos_available"]:
            if col not in last_row.columns:
                last_row[col] = 0.0
        feature_rows.append(last_row)

    if not feature_rows:
        return pd.DataFrame()

    # Build model-required panel features before applying inference-only
    # filters. This matches sweeps: models score the full feature panel, then
    # BacktestConfig filters the pick pool.
    score_frame = pd.concat(feature_rows, ignore_index=True)
    score_frame = _attach_live_model_required_features(score_frame, models)

    meta_rows = []
    keep_abs: list[bool] = []
    for _, row in score_frame.iterrows():
        dolvol = float(row["dolvol_20d_log"])
        spread = (
            float(row["spread_bps"])
            if "spread_bps" in row.index and pd.notna(row["spread_bps"])
            else 25.0
        )
        spread_cap = float(max_spread_bps or 0.0)
        keep = dolvol >= np.log1p(min_dollar_vol)
        if keep and spread_cap > 0.0:
            keep = spread <= spread_cap
        # Realised-vol floor/ceiling — matches BacktestConfig inference filters.
        if keep and min_vol_20d > 0.0 and "vol_20d" in row.index:
            v20 = float(row["vol_20d"])
            keep = bool(np.isfinite(v20) and v20 >= min_vol_20d)
        if keep and max_vol_20d > 0.0 and "vol_20d" in row.index:
            v20 = float(row["vol_20d"])
            keep = bool(np.isfinite(v20) and v20 <= max_vol_20d)
        keep_abs.append(bool(keep))
        if not keep:
            continue
        meta = {
            "symbol": str(row["symbol"]),
            "spread_bps": spread,
            "last_close": float(row["actual_close"]),
            "dolvol_20d_log": dolvol,
            "ret_5d": float(row["ret_5d"]),
        }
        if "ret_20d" in row.index:
            meta["ret_20d"] = float(row["ret_20d"])
        if "vol_20d" in row.index:
            meta["vol_20d"] = float(row["vol_20d"])
        meta_rows.append(meta)

    if not meta_rows:
        return pd.DataFrame()
    score_frame = score_frame.loc[keep_abs].reset_index(drop=True)

    meta_df = pd.DataFrame(meta_rows)
    keep = pd.Series(True, index=meta_df.index)
    if float(max_ret_20d_rank_pct) < 1.0 and "ret_20d" in meta_df.columns:
        r20 = pd.to_numeric(meta_df["ret_20d"], errors="coerce").rank(
            pct=True, method="average"
        )
        keep &= r20 <= float(max_ret_20d_rank_pct)
    if float(min_ret_5d_rank_pct) > 0.0 and "ret_5d" in meta_df.columns:
        r5 = pd.to_numeric(meta_df["ret_5d"], errors="coerce").rank(
            pct=True, method="average"
        )
        keep &= r5 >= float(min_ret_5d_rank_pct)
    if not bool(keep.any()):
        return pd.DataFrame()
    if not bool(keep.all()):
        keep_mask = keep.to_numpy(dtype=bool)
        meta_rows = [
            row for row, keep_row in zip(meta_rows, keep_mask.tolist(), strict=False)
            if keep_row
        ]
        score_frame = score_frame.loc[keep_mask].reset_index(drop=True)

    per_model_scores = [
        pd.Series(m.predict_scores(score_frame), dtype="float64").to_numpy()
        for m in models
    ]
    score_matrix = np.vstack(per_model_scores)
    blended = np.mean(score_matrix, axis=0)
    score_std = np.std(score_matrix, axis=0)
    penalty = max(float(score_uncertainty_penalty or 0.0), 0.0)
    adjusted = blended - penalty * score_std

    rows = []
    for idx, meta in enumerate(meta_rows):
        row = dict(meta)
        row["score"] = float(adjusted[idx])
        row["raw_score_mean"] = float(blended[idx])
        row["score_std"] = float(score_std[idx])
        row["per_seed_scores"] = [float(score_matrix[j, idx]) for j in range(score_matrix.shape[0])]
        rows.append(row)

    df_scores = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df_scores


def _attach_live_model_required_features(
    score_frame: pd.DataFrame,
    models: list[XGBStockModel],
) -> pd.DataFrame:
    """Attach optional panel features needed by rank/dispersion-trained models."""
    required_cols = {
        col
        for model in models
        for col in (getattr(model, "feature_cols", None) or [])
        if isinstance(col, str) and col
    }
    if not required_cols:
        return score_frame
    if required_cols.intersection(DAILY_RANK_FEATURE_COLS):
        score_frame = add_cross_sectional_ranks(score_frame)
    if required_cols.intersection(DAILY_DISPERSION_FEATURE_COLS):
        score_frame = add_cross_sectional_dispersion(score_frame)
    missing = sorted(col for col in required_cols if col not in score_frame.columns)
    if missing:
        raise ValueError(
            "score_all_symbols: live feature frame missing model feature columns: "
            f"{missing}"
        )
    return score_frame


def _apply_cross_sectional_regime_gate(
    scores_df: pd.DataFrame,
    *,
    regime_cs_iqr_max: float = 0.0,
    regime_cs_skew_min: float = -1e9,
    trade_logger: TradeLogger | None = None,
) -> pd.DataFrame:
    """Apply simulator-matched day-level ret_5d IQR/skew gates.

    The backtest computes these stats after liquidity/spread/vol filters and
    before min_score/top_n. ``score_all_symbols`` returns that same post-filter
    live universe, so the gate here intentionally sits before conviction
    filtering.
    """
    iqr_active = float(regime_cs_iqr_max) > 0.0
    skew_active = float(regime_cs_skew_min) > -1e8
    if not (iqr_active or skew_active) or len(scores_df) == 0:
        return scores_df
    if "ret_5d" not in scores_df.columns:
        logger.warning("Cross-sectional regime gate requested but ret_5d is missing")
        return scores_df

    keep, cs_iqr, cs_skew = evaluate_cross_sectional_regime_gate(
        scores_df["ret_5d"],
        regime_cs_iqr_max=float(regime_cs_iqr_max),
        regime_cs_skew_min=float(regime_cs_skew_min),
    )

    if trade_logger is not None:
        trade_logger.log(
            "regime_cs_gate",
            cs_iqr_ret5=cs_iqr,
            cs_skew_ret5=cs_skew,
            regime_cs_iqr_max=float(regime_cs_iqr_max),
            regime_cs_skew_min=float(regime_cs_skew_min),
            kept=bool(keep),
            n_total=int(len(scores_df)),
        )

    if keep:
        return scores_df

    print(
        "[xgb-live] Cross-sectional regime gate closed: "
        f"cs_iqr_ret5={cs_iqr:.4f} max={float(regime_cs_iqr_max):.4f}, "
        f"cs_skew_ret5={cs_skew:.4f} min={float(regime_cs_skew_min):.4f}",
        flush=True,
    )
    return scores_df.iloc[0:0].copy()


def _load_spy_close_by_date(spy_csv: Path) -> pd.Series:
    """Load SPY closes indexed by ET date for live vol targeting."""
    if not spy_csv.exists():
        return pd.Series(dtype="float64")
    try:
        df = pd.read_csv(spy_csv)
    except Exception as exc:
        logger.warning("Failed to read SPY CSV %s: %s", spy_csv, exc)
        return pd.Series(dtype="float64")
    if "close" not in df.columns:
        logger.warning("SPY CSV %s is missing close column", spy_csv)
        return pd.Series(dtype="float64")
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        dates = ts.dt.tz_convert(ET).dt.date
    elif "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dt.date
    else:
        logger.warning("SPY CSV %s is missing timestamp/date column", spy_csv)
        return pd.Series(dtype="float64")
    out = (
        pd.DataFrame({"date": dates, "close": pd.to_numeric(df["close"], errors="coerce")})
        .dropna(subset=["date", "close"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
    )
    if len(out) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(out["close"].astype(float).to_numpy(), index=pd.Index(out["date"]))


def _resolved_spy_csv_for_args(args: argparse.Namespace) -> Path | None:
    if (
        float(getattr(args, "vol_target_ann", 0.0) or 0.0) <= 0.0
        and int(getattr(args, "regime_gate_window", 0) or 0) <= 0
    ):
        return None
    raw_spy_csv = getattr(args, "spy_csv", None)
    if raw_spy_csv:
        return Path(raw_spy_csv).resolve(strict=False)
    raw_data_root = getattr(args, "data_root", None)
    if raw_data_root is None:
        return None
    return (Path(raw_data_root) / "SPY.csv").resolve(strict=False)


def _optional_file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _live_spy_vol_target_scale(
    data_root: Path,
    args: argparse.Namespace,
    *,
    trade_logger: TradeLogger | None = None,
) -> float:
    """Return today's SPY vol-target scale in [0, 1], fail-closed on missing SPY data."""
    target_ann = float(getattr(args, "vol_target_ann", 0.0) or 0.0)
    if target_ann <= 0.0:
        return 1.0

    spy_csv = Path(getattr(args, "spy_csv", "") or (Path(data_root) / "SPY.csv")).resolve(strict=False)
    spy_csv_sha256 = _optional_file_sha256(spy_csv)
    spy_close = _load_spy_close_by_date(spy_csv)
    expected_latest = _expected_latest_completed_daily_bar_date()
    eligible = spy_close[spy_close.index <= expected_latest]
    if len(eligible) == 0:
        print(
            "[xgb-live] SPY vol-target requested but no SPY close history is "
            "available; holding cash.",
            file=sys.stderr,
            flush=True,
        )
        if trade_logger is not None:
            trade_logger.log(
                "spy_vol_target",
                target_ann=target_ann,
                scale=0.0,
                reason="missing_spy_history",
                spy_csv=str(spy_csv),
                spy_csv_sha256=spy_csv_sha256,
            )
        return 0.0

    latest_date = max(eligible.index)
    if latest_date < expected_latest:
        print(
            "[xgb-live] SPY vol-target requested but SPY close history is stale "
            f"(latest={latest_date}, expected={expected_latest}); holding cash.",
            file=sys.stderr,
            flush=True,
        )
        if trade_logger is not None:
            trade_logger.log(
                "spy_vol_target",
                target_ann=target_ann,
                scale=0.0,
                reason="stale_spy_history",
                spy_date=str(latest_date),
                expected_spy_date=str(expected_latest),
                spy_csv=str(spy_csv),
                spy_csv_sha256=spy_csv_sha256,
            )
        return 0.0
    scale_series = _build_vol_scale(
        eligible,
        pd.Index([latest_date]),
        target_ann=target_ann,
    )
    scale = float(scale_series.iloc[0]) if len(scale_series) else 1.0
    if not np.isfinite(scale):
        scale = 0.0
    scale = float(np.clip(scale, 0.0, 1.0))
    if trade_logger is not None:
        trade_logger.log(
            "spy_vol_target",
            target_ann=target_ann,
            scale=scale,
            spy_date=str(latest_date),
            spy_csv=str(spy_csv),
            spy_csv_sha256=spy_csv_sha256,
        )
    if scale < 1.0:
        print(
            f"[xgb-live] SPY vol-target scale={scale:.3f} "
            f"(target_ann={target_ann:.3f}, spy_date={latest_date})",
            flush=True,
        )
    return scale


def _live_spy_regime_gate_closed(
    data_root: Path,
    args: argparse.Namespace,
    *,
    trade_logger: TradeLogger | None = None,
) -> bool:
    """Return True when the SPY MA regime gate says today's session should stay cash."""
    window = int(getattr(args, "regime_gate_window", 0) or 0)
    if window <= 0:
        return False

    spy_csv = Path(getattr(args, "spy_csv", "") or (Path(data_root) / "SPY.csv")).resolve(strict=False)
    spy_csv_sha256 = _optional_file_sha256(spy_csv)
    spy_close = _load_spy_close_by_date(spy_csv)
    expected_latest = _expected_latest_completed_daily_bar_date()
    eligible = spy_close[spy_close.index <= expected_latest]
    if len(eligible) < window:
        print(
            "[xgb-live] SPY regime gate requested but insufficient SPY close "
            f"history is available ({len(eligible)}/{window}); holding cash.",
            file=sys.stderr,
            flush=True,
        )
        if trade_logger is not None:
            trade_logger.log(
                "spy_regime_gate",
                window=window,
                closed=True,
                reason="insufficient_spy_history",
                n_spy_days=int(len(eligible)),
                spy_csv=str(spy_csv),
                spy_csv_sha256=spy_csv_sha256,
            )
        return True

    latest_date = max(eligible.index)
    if latest_date < expected_latest:
        print(
            "[xgb-live] SPY regime gate requested but SPY close history is stale "
            f"(latest={latest_date}, expected={expected_latest}); holding cash.",
            file=sys.stderr,
            flush=True,
        )
        if trade_logger is not None:
            trade_logger.log(
                "spy_regime_gate",
                window=window,
                closed=True,
                reason="stale_spy_history",
                spy_date=str(latest_date),
                expected_spy_date=str(expected_latest),
                spy_csv=str(spy_csv),
                spy_csv_sha256=spy_csv_sha256,
            )
        return True
    closed_series = _build_regime_flags(
        eligible,
        pd.Index([latest_date]),
        window=window,
        available_at_open=False,
    )
    closed = bool(closed_series.iloc[0]) if len(closed_series) else True
    if trade_logger is not None:
        trade_logger.log(
            "spy_regime_gate",
            window=window,
            closed=closed,
            spy_date=str(latest_date),
            spy_csv=str(spy_csv),
            spy_csv_sha256=spy_csv_sha256,
        )
    if closed:
        print(
            f"[xgb-live] SPY regime gate closed: window={window} "
            f"spy_date={latest_date}; holding cash.",
            flush=True,
        )
    return closed


# ── Main trading loop ─────────────────────────────────────────────────────────

def _wait_until(hour: int, minute: int, tz: ZoneInfo, poll_secs: float = 10.0) -> None:
    """Block until local clock reaches HH:MM in tz."""
    while True:
        now = datetime.now(tz)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        secs = (target - now).total_seconds()
        if secs <= 0:
            return
        wait = min(secs, poll_secs)
        logger.debug("Waiting %.0fs until %02d:%02d %s", secs, hour, minute, tz)
        time.sleep(wait)


def _target_buy_qty(*, buy_notional: float, price: float) -> float:
    if not np.isfinite(buy_notional) or not np.isfinite(price) or buy_notional <= 0 or price <= 0:
        return 0.0
    qty = round(float(buy_notional) / float(price), 4)
    if qty <= 0:
        return 0.0
    return max(qty, MIN_FRACTIONAL_QTY)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    def finite(name: str) -> float:
        value = float(getattr(args, name))
        if not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
        return value

    if int(args.top_n) < 1:
        parser.error("--top-n must be >= 1")
    if int(args.min_picks) < 0:
        parser.error("--min-picks must be >= 0")
    if int(args.min_picks) > int(args.top_n):
        parser.error("--min-picks must be <= --top-n")
    if int(args.regime_gate_window) < 0:
        parser.error("--regime-gate-window must be >= 0")
    if finite("allocation") <= 0.0:
        parser.error("--allocation must be > 0")
    if finite("allocation_temp") <= 0.0:
        parser.error("--allocation-temp must be > 0")
    min_score = finite("min_score")
    if not 0.0 <= min_score <= 1.0:
        parser.error("--min-score must be between 0 and 1")
    for name in (
        "commission_bps",
        "min_dollar_vol",
        "max_spread_bps",
        "min_vol_20d",
        "max_vol_20d",
        "regime_cs_iqr_max",
        "vol_target_ann",
        "crypto_max_gross",
        "score_uncertainty_penalty",
        "inv_vol_target_ann",
    ):
        if finite(name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if finite("inv_vol_floor") <= 0.0:
        parser.error("--inv-vol-floor must be > 0")
    if finite("inv_vol_cap") < 1.0:
        parser.error("--inv-vol-cap must be >= 1")
    for name in ("max_ret_20d_rank_pct", "min_ret_5d_rank_pct"):
        value = finite(name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    finite("regime_cs_skew_min")
    if finite("no_picks_fallback_alloc") < 0.0:
        parser.error("--no-picks-fallback-alloc must be >= 0")
    lo = finite("conviction_alloc_low")
    hi = finite("conviction_alloc_high")
    if hi <= lo:
        parser.error("--conviction-alloc-high must be > --conviction-alloc-low")
    if int(args.crypto_poll_seconds) < 1:
        parser.error("--crypto-poll-seconds must be >= 1")
    if finite("eod_max_gross_leverage") <= 0.0:
        parser.error("--eod-max-gross-leverage must be > 0")
    if int(args.eod_deleverage_window_minutes) < 1:
        parser.error("--eod-deleverage-window-minutes must be >= 1")
    if int(args.eod_force_market_minutes) < 0:
        parser.error("--eod-force-market-minutes must be >= 0")
    return args


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols-file", type=Path,
                   default=REPO / "symbol_lists/stocks_wide_1000_v1.txt")
    p.add_argument("--data-root",    type=Path, default=REPO / "trainingdata")
    p.add_argument("--spy-csv",      type=Path, default=None,
                   help="Optional explicit SPY daily CSV for --vol-target-ann "
                        "and --regime-gate-window. Defaults to data-root/SPY.csv.")
    p.add_argument("--model-path",   type=Path, default=DEFAULT_MODEL_PATH,
                   help="Pre-trained XGBStockModel pickle (ignored when --model-paths is set)")
    p.add_argument("--model-paths",  type=str, default="",
                   help="Comma-separated list of pickles — ensemble mean-blend (predict_proba avg). "
                        "Takes precedence over --model-path when set.")
    p.add_argument("--top-n",        type=int,   default=2,
                   help="Number of stocks to buy per day")
    p.add_argument("--min-picks",    type=int,   default=0,
                   help="Aggressive packing floor. When >0, fill at least "
                        "this many slots from the best-ranked candidates even "
                        "if their score is below --min-score. Bounded by "
                        "--top-n. 0 (default) preserves confidence-gated "
                        "behavior.")
    p.add_argument("--allocation",   type=float, default=0.25,
                   help="Fraction of portfolio to deploy (0.25 = 25%% per pick, shared)")
    p.add_argument("--allocation-mode", type=str, default="equal",
                   choices=("equal", "score_norm", "softmax"),
                   help="How to split --allocation across selected picks. "
                        "Matches BacktestConfig.allocation_mode.")
    p.add_argument("--allocation-temp", type=float, default=1.0,
                   help="Softmax temperature when --allocation-mode=softmax. "
                        "Ignored by equal and score_norm.")
    p.add_argument("--min-score",    type=float, default=0.0,
                   help="Skip pick if blended predict_proba < min_score. "
                        "0.0 (default) = no filter. 0.55-0.70 gates on conviction. "
                        "If all top_n candidates fail, session holds cash.")
    p.add_argument("--score-uncertainty-penalty", type=float, default=0.0,
                   help="Rank and gate by mean_score - penalty * std(score "
                        "across ensemble seeds). 0 disables. This implements "
                        "uncertainty-adjusted sorting for the live ensemble.")
    p.add_argument("--inv-vol-target-ann", type=float, default=0.0,
                   help="Per-pick inverse-volatility exposure target. 0 "
                        "disables. Matches BacktestConfig.inv_vol_target_ann.")
    p.add_argument("--inv-vol-floor", type=float, default=0.05,
                   help="Minimum vol_20d denominator for --inv-vol-target-ann.")
    p.add_argument("--inv-vol-cap", type=float, default=3.0,
                   help="Symmetric cap for per-pick inverse-vol multiplier.")
    p.add_argument("--commission-bps", type=float, default=10.0)
    p.add_argument("--min-dollar-vol", type=float, default=5e6)
    p.add_argument("--max-spread-bps", type=float, default=30.0,
                   help="Maximum volume-estimated spread allowed in the "
                        "live pick pool. 30 matches BacktestConfig default; "
                        "0 disables the spread filter.")
    p.add_argument("--min-vol-20d", type=float, default=0.0,
                   help="Realised 20d annualised vol floor (e.g. 0.10). 0 "
                        "disables. Drops dead-zone symbols that LOBO flagged; "
                        "strict-dominance at 0.10 (deploy + stress36x).")
    p.add_argument("--max-vol-20d", type=float, default=0.0,
                   help="Realised 20d annualised vol cap. 0 disables. "
                        "Matches BacktestConfig.max_vol_20d.")
    p.add_argument("--max-ret-20d-rank-pct", type=float, default=1.0,
                   help="Drop names whose ret_20d percentile rank is above "
                        "this threshold. 1.0 disables.")
    p.add_argument("--min-ret-5d-rank-pct", type=float, default=0.0,
                   help="Drop names whose ret_5d percentile rank is below "
                        "this threshold. 0.0 disables.")
    p.add_argument("--regime-cs-iqr-max", type=float, default=0.0,
                   help="Cross-sectional ret_5d IQR day gate. 0 disables. "
                        "Matches BacktestConfig.regime_cs_iqr_max.")
    p.add_argument("--regime-cs-skew-min", type=float, default=-1e9,
                   help="Cross-sectional ret_5d skew day gate. Values above "
                        "-1e8 enable the gate. Matches "
                        "BacktestConfig.regime_cs_skew_min.")
    p.add_argument("--vol-target-ann", type=float, default=0.0,
                   help="SPY realised-vol target for day-level exposure "
                        "scaling. 0 disables. Uses data-root/SPY.csv and "
                        "matches BacktestConfig.vol_target_ann.")
    p.add_argument("--regime-gate-window", type=int, default=0,
                   help="SPY moving-average regime window. 0 disables. When "
                        "enabled, the live session holds cash if the latest "
                        "available SPY close is below its moving average.")
    p.add_argument("--hold-through", action="store_true",
                   help="If tomorrow's picks match today's held positions, skip the "
                        "sell-at-close + buy-at-open round-trip. Rotation now happens "
                        "at next-open (not close): sell only names that dropped out of "
                        "picks, buy only names that just entered. Saves 2×(fee+buffer) "
                        "per carried day and captures overnight drift. Backtest-validated "
                        "strict-dominance upgrade (docs/xgbnew_hold_through_20260419.md).")
    p.add_argument("--live",         action="store_true",
                   help="Use live Alpaca account (default: paper)")
    p.add_argument("--dry-run",      action="store_true",
                   help="Score only — do not place any orders")
    p.add_argument("--trade-log-dir", type=Path, default=None,
                   help="Per-session JSONL event log (picks, per-seed scores, "
                        "order ids, fill slippage vs last_close, equity). "
                        "Defaults to analysis/xgb_live_trade_log/ when unset. "
                        "Set to '' to disable.")
    p.add_argument("--no-trade-log", action="store_true",
                   help="Disable the trade log entirely.")
    p.add_argument("--no-picks-fallback", type=str, default="",
                   help="Symbol to buy on days where no candidate clears "
                        "--min-score. Typical: 'SPY' (broad market) or 'QQQ' "
                        "(higher drift). Empty = hold cash (legacy). The "
                        "fallback trade is sized at --allocation * "
                        "--no-picks-fallback-alloc (so the caller can run a "
                        "small defensive position on low-conviction days). "
                        "Skipped automatically on hold-through continuation "
                        "days — only fires on clean no-pick days.")
    p.add_argument("--no-picks-fallback-alloc", type=float, default=0.5,
                   help="Fraction of --allocation to use for the no-picks "
                        "fallback order. 0.5 = half the deployed leverage. "
                        "Ignored when --no-picks-fallback is empty.")
    p.add_argument("--conviction-scaled-alloc", action="store_true",
                   help="Scale total stock exposure by top score using the "
                        "same ramp as BacktestConfig.conviction_scaled_alloc.")
    p.add_argument("--conviction-alloc-low", type=float, default=0.55,
                   help="Top-score ramp value that maps to 0%% exposure when "
                        "--conviction-scaled-alloc is enabled.")
    p.add_argument("--conviction-alloc-high", type=float, default=0.85,
                   help="Top-score ramp value that maps to 100%% exposure when "
                        "--conviction-scaled-alloc is enabled.")
    p.add_argument("--loop",         action="store_true",
                   help="Keep running (wait for next market open after each session)")
    # Embedded crypto weekend trader (single-leader-process architecture —
    # shares the `alpaca_live_writer` fcntl lock with stock trading so only
    # ONE program ever talks to Alpaca).
    p.add_argument("--crypto-weekend", action="store_true",
                   help="Enable the embedded crypto weekend trader. Polls every "
                        "--crypto-poll-seconds during inter-session sleep. "
                        "Buys BTC/ETH/SOL passing trend filter on Saturday, "
                        "exits Monday AM before US stock open.")
    p.add_argument("--crypto-poll-seconds", type=int, default=300,
                   help="Poll cadence for the embedded crypto trader (default 300s).")
    p.add_argument("--crypto-max-gross", type=float, default=0.5,
                   help="Crypto total gross exposure as fraction of equity "
                        "(actual cap is min(cash - $50, equity * this)).")
    p.add_argument("--eod-deleverage", action="store_true",
                   help="Run the embedded end-of-day equity deleverage tick "
                        "inside this single Alpaca writer process.")
    p.add_argument("--eod-max-gross-leverage", type=float, default=2.0,
                   help="Target max gross equity leverage before close "
                        "(default 2.0x). Crypto positions are ignored.")
    p.add_argument("--eod-deleverage-window-minutes", type=int, default=60,
                   help="Start deleveraging this many minutes before the "
                        "Alpaca stock market close.")
    p.add_argument("--eod-force-market-minutes", type=int, default=5,
                   help="Within this many minutes of close, use more aggressive "
                        "near-market limit prices instead of resting passively.")
    p.add_argument("--verbose", "-v", action="store_true")
    return _validate_args(p, p.parse_args(argv))


def _load_symbols(path: Path) -> list[str]:
    syms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().split("#", 1)[0].strip().upper()
        if s:
            syms.append(s)
    return syms


def _score_and_pick(
    symbols, data_root, model, args,
    trade_logger: TradeLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch live bars, score all symbols, apply min_score filter, return (picks, all_scores)."""
    print("[xgb-live] Fetching live bars from Alpaca...", flush=True)
    try:
        live_bars = _get_latest_bars(symbols, n_days=10)
    except Exception as exc:
        logger.warning("Live bar fetch failed: %s", exc)
        live_bars = {}

    print(f"[xgb-live] Scoring {len(symbols)} symbols...", flush=True)
    scores_df = score_all_symbols(
        symbols, data_root, model, live_bars,
        min_dollar_vol=args.min_dollar_vol,
        max_spread_bps=float(getattr(args, "max_spread_bps", 30.0) or 0.0),
        min_vol_20d=float(getattr(args, "min_vol_20d", 0.0) or 0.0),
        max_vol_20d=float(getattr(args, "max_vol_20d", 0.0) or 0.0),
        max_ret_20d_rank_pct=float(getattr(args, "max_ret_20d_rank_pct", 1.0)),
        min_ret_5d_rank_pct=float(getattr(args, "min_ret_5d_rank_pct", 0.0) or 0.0),
        score_uncertainty_penalty=float(
            getattr(args, "score_uncertainty_penalty", 0.0) or 0.0
        ),
    )

    if trade_logger is not None:
        top20 = scores_df.head(20) if len(scores_df) else scores_df
        trade_logger.log(
            "scored",
            n_candidates=int(len(scores_df)),
            top20=top20.to_dict(orient="records"),
        )

    if len(scores_df) == 0:
        return scores_df, scores_df

    scores_df = _apply_cross_sectional_regime_gate(
        scores_df,
        regime_cs_iqr_max=float(getattr(args, "regime_cs_iqr_max", 0.0) or 0.0),
        regime_cs_skew_min=float(getattr(args, "regime_cs_skew_min", -1e9)),
        trade_logger=trade_logger,
    )
    if len(scores_df) == 0:
        return scores_df, scores_df

    min_score = float(getattr(args, "min_score", 0.0) or 0.0)
    if min_score > 0.0:
        filtered = scores_df[scores_df["score"] >= min_score]
        min_pick_floor = min(
            max(int(getattr(args, "min_picks", 0) or 0), 0),
            int(args.top_n),
        )
        forced_low_conf = 0
        if len(filtered) < min_pick_floor:
            need = min_pick_floor - len(filtered)
            supplement = scores_df.loc[~scores_df.index.isin(filtered.index)].head(need)
            forced_low_conf = int(len(supplement))
            if forced_low_conf:
                filtered = pd.concat([filtered, supplement], ignore_index=False)
        print(f"[xgb-live] Conviction filter min_score={min_score:.2f}: "
              f"{len(filtered)}/{len(scores_df)} candidates pass "
              f"(top score={scores_df['score'].iloc[0]:.4f}, "
              f"forced_low_conf={forced_low_conf})", flush=True)
        if trade_logger is not None:
            trade_logger.log(
                "filtered",
                min_score=min_score,
                min_picks=min_pick_floor,
                n_pass=int(len(filtered)),
                n_forced_low_confidence=forced_low_conf,
                n_total=int(len(scores_df)),
                top_score=float(scores_df["score"].iloc[0]),
            )
        if len(filtered) == 0:
            return filtered, scores_df
        return filtered.head(args.top_n), scores_df
    return scores_df.head(args.top_n), scores_df


def _emit_session_start(
    tlog: TradeLogger,
    args: argparse.Namespace,
    client,
    *,
    mode: str,
) -> float | None:
    """Emit the session_start event and return equity_pre (or None)."""
    paper = not args.live
    equity_pre = None
    if client is not None and not args.dry_run:
        try:
            equity_pre = float(getattr(_get_account(client), "equity", 0.0) or 0.0)
        except Exception:
            equity_pre = None
    spy_csv = _resolved_spy_csv_for_args(args)
    tlog.log(
        "session_start",
        mode=mode,
        paper=paper,
        dry_run=bool(args.dry_run),
        top_n=int(args.top_n),
        min_picks=int(getattr(args, "min_picks", 0) or 0),
        allocation=float(args.allocation),
        allocation_mode=str(getattr(args, "allocation_mode", "equal") or "equal"),
        allocation_temp=float(getattr(args, "allocation_temp", 1.0) or 1.0),
        inv_vol_target_ann=float(getattr(args, "inv_vol_target_ann", 0.0) or 0.0),
        inv_vol_floor=float(getattr(args, "inv_vol_floor", 0.05) or 0.05),
        inv_vol_cap=float(getattr(args, "inv_vol_cap", 3.0) or 3.0),
        score_uncertainty_penalty=float(
            getattr(args, "score_uncertainty_penalty", 0.0) or 0.0
        ),
        conviction_scaled_alloc=bool(getattr(args, "conviction_scaled_alloc", False)),
        conviction_alloc_low=float(
            0.55 if getattr(args, "conviction_alloc_low", None) is None
            else getattr(args, "conviction_alloc_low")
        ),
        conviction_alloc_high=float(
            0.85 if getattr(args, "conviction_alloc_high", None) is None
            else getattr(args, "conviction_alloc_high")
        ),
        min_score=float(getattr(args, "min_score", 0.0) or 0.0),
        min_dollar_vol=float(args.min_dollar_vol),
        min_vol_20d=float(getattr(args, "min_vol_20d", 0.0) or 0.0),
        max_vol_20d=float(getattr(args, "max_vol_20d", 0.0) or 0.0),
        max_ret_20d_rank_pct=float(getattr(args, "max_ret_20d_rank_pct", 1.0)),
        min_ret_5d_rank_pct=float(getattr(args, "min_ret_5d_rank_pct", 0.0) or 0.0),
        regime_gate_window=int(getattr(args, "regime_gate_window", 0) or 0),
        vol_target_ann=float(getattr(args, "vol_target_ann", 0.0) or 0.0),
        spy_csv=str(spy_csv) if spy_csv is not None else None,
        spy_csv_sha256=_optional_file_sha256(spy_csv),
        equity_pre=equity_pre,
    )
    return equity_pre


def _wait_for_market_open() -> None:
    """Block until MARKET_OPEN ET if the clock says we're before it."""
    now_et = datetime.now(ET)
    if now_et.hour < MARKET_OPEN[0] or (now_et.hour == MARKET_OPEN[0] and
                                         now_et.minute < MARKET_OPEN[1]):
        print(f"[xgb-live] Waiting for {MARKET_OPEN[0]:02d}:{MARKET_OPEN[1]:02d} ET...",
              flush=True)
        _wait_until(MARKET_OPEN[0], MARKET_OPEN[1], ET)


def _announce_picks(
    picks: pd.DataFrame,
    today_str: str,
    top_n: int,
    tlog: TradeLogger,
    *,
    tag: str,
) -> None:
    """Print + trade-log the per-rank pick lines (shared by both entry points)."""
    print(f"\n[{tag}] Top-{top_n} picks for {today_str}:")
    for rank, (_, row) in enumerate(picks.iterrows(), start=1):
        print(f"  {row['symbol']:<8}  score={row['score']:.4f}  "
              f"last_close=${row['last_close']:.2f}  spread={row['spread_bps']:.1f}bps")
        tlog.log("pick", rank=rank, symbol=str(row["symbol"]),
                 score=float(row["score"]),
                 per_seed_scores=list(row.get("per_seed_scores", []) or []),
                 last_close=float(row["last_close"]),
                 spread_bps=float(row["spread_bps"]))


def _conviction_allocation_scale(
    picks: pd.DataFrame,
    all_scores: pd.DataFrame,
    args: argparse.Namespace,
    *,
    trade_logger: TradeLogger | None = None,
) -> tuple[pd.DataFrame, float]:
    """Apply simulator-matched top-score allocation scaling."""
    if not bool(getattr(args, "conviction_scaled_alloc", False)):
        return picks, 1.0

    if len(picks) > 0:
        top_score = float(pd.to_numeric(picks["score"], errors="coerce").iloc[0])
    elif len(all_scores) > 0 and "score" in all_scores.columns:
        top_score = float(pd.to_numeric(all_scores["score"], errors="coerce").max())
    else:
        top_score = 0.0
    if not np.isfinite(top_score):
        top_score = 0.0

    lo_raw = getattr(args, "conviction_alloc_low", None)
    hi_raw = getattr(args, "conviction_alloc_high", None)
    lo = float(0.55 if lo_raw is None else lo_raw)
    hi = float(0.85 if hi_raw is None else hi_raw)
    span = max(hi - lo, 1e-9)
    scale = float(np.clip((top_score - lo) / span, 0.0, 1.0))

    if trade_logger is not None:
        trade_logger.log(
            "conviction_scaled_alloc",
            top_score=top_score,
            scale=scale,
            conviction_alloc_low=lo,
            conviction_alloc_high=hi,
            n_picks_before=int(len(picks)),
        )
    if scale <= 0.0 and len(picks) > 0:
        return picks.iloc[0:0].copy(), scale
    return picks, scale


def _apply_no_picks_fallback(
    picks: pd.DataFrame,
    all_scores: pd.DataFrame,
    args: argparse.Namespace,
    *,
    today_str: str,
    trade_logger: TradeLogger | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Return fallback pick when the scored model emits no selected names.

    This mirrors ``BacktestConfig.no_picks_fallback_*`` for both live session
    modes. The fallback is only used on clean no-pick days; callers still keep
    their existing hold/rotation semantics after this function returns.
    """
    if len(picks) > 0:
        return picks, False

    fb_sym = str(getattr(args, "no_picks_fallback", "") or "").strip().upper()
    fb_alloc_frac = float(getattr(args, "no_picks_fallback_alloc", 0.0) or 0.0)
    min_score = float(getattr(args, "min_score", 0.0) or 0.0)
    if not (fb_sym and fb_alloc_frac > 0.0):
        print(
            f"[xgb-live] No picks today — holding current positions (if any).",
            flush=True,
        )
        if trade_logger is not None:
            trade_logger.log("no_picks")
        return picks, False

    if len(all_scores) == 0 or "symbol" not in all_scores.columns:
        print(
            f"[xgb-live] NO pick meets min_score={min_score:.2f} AND fallback "
            f"{fb_sym} has no scored universe — holding cash/positions for {today_str}.",
            flush=True,
        )
        if trade_logger is not None:
            trade_logger.log(
                "no_picks",
                reason="min_score_fallback_missing",
                fallback_symbol=fb_sym,
            )
        return picks, False

    fb_row = all_scores[all_scores["symbol"].astype(str).str.upper() == fb_sym]
    if len(fb_row) == 0:
        print(
            f"[xgb-live] NO pick meets min_score={min_score:.2f} AND fallback "
            f"{fb_sym} missing from scored universe — holding cash/positions "
            f"for {today_str}.",
            flush=True,
        )
        if trade_logger is not None:
            trade_logger.log(
                "no_picks",
                reason="min_score_fallback_missing",
                fallback_symbol=fb_sym,
            )
        return picks, False

    out = fb_row.head(1).reset_index(drop=True).copy()
    print(
        f"[xgb-live] NO pick meets min_score={min_score:.2f} — falling back "
        f"to {fb_sym} @ {fb_alloc_frac:.0%} of --allocation.",
        flush=True,
    )
    if trade_logger is not None:
        trade_logger.log(
            "no_picks_fallback",
            reason="min_score",
            fallback_symbol=fb_sym,
            fallback_alloc_frac=fb_alloc_frac,
        )
    return out, True


def _buy_notional_by_symbol(
    picks: pd.DataFrame,
    *,
    total_notional: float,
    allocation_mode: str = "equal",
    allocation_temp: float = 1.0,
    inv_vol_target_ann: float = 0.0,
    inv_vol_floor: float = 0.05,
    inv_vol_cap: float = 3.0,
) -> dict[str, float]:
    """Allocate BUY notional across picks using simulator-matched weights."""
    if len(picks) == 0 or not np.isfinite(total_notional) or total_notional <= 0:
        return {}
    scores = pd.to_numeric(picks["score"], errors="coerce").fillna(0.0).to_numpy()
    weights = _allocation_weights(
        scores,
        mode=str(allocation_mode or "equal"),
        temperature=float(allocation_temp or 1.0),
    )
    out: dict[str, float] = {}
    for weight, (_, row) in zip(weights, picks.iterrows(), strict=False):
        sym = str(row["symbol"])
        try:
            pick_vol = float(row.get("vol_20d", 0.0))
        except (TypeError, ValueError):
            pick_vol = 0.0
        pick_scale = _inv_vol_pick_scale(
            pick_vol,
            target_ann=float(inv_vol_target_ann or 0.0),
            floor=float(inv_vol_floor or 0.05),
            cap=float(inv_vol_cap or 3.0),
        )
        out[sym] = (
            out.get(sym, 0.0)
            + float(total_notional) * float(weight) * float(pick_scale)
        )
    return out


def _execute_buys(
    client,
    picks: pd.DataFrame,
    buy_notional: float | None,
    tlog: TradeLogger,
    *,
    notional_by_symbol: dict[str, float] | None = None,
    target_syms: set[str] | None = None,
) -> None:
    """Submit BUY orders for each pick (optionally restricted to ``target_syms``).

    Records buy price for HARD RULE #3 via ``record_buy_price``. Import is
    done inside the function so tests monkeypatching
    ``src.alpaca_singleton.record_buy_price`` still intercept the call.
    """
    from src.alpaca_singleton import record_buy_price

    for _, row in picks.iterrows():
        sym = str(row["symbol"])
        if target_syms is not None and sym not in target_syms:
            continue
        row_notional = (
            float(notional_by_symbol.get(sym, 0.0))
            if notional_by_symbol is not None
            else float(buy_notional or 0.0)
        )
        price = float(row["last_close"])
        if price <= 0:
            logger.warning("Skipping BUY %s — invalid last_close %s", sym, price)
            continue
        qty = _target_buy_qty(buy_notional=row_notional, price=price)
        if qty <= 0:
            logger.warning("Skipping BUY %s — invalid target qty for price=%s "
                           "notional=%s", sym, price, row_notional)
            continue
        limit_price = _stock_limit_price_near_market(
            sym,
            side="buy",
            reference_price=price,
            aggressiveness_bps=15.0,
        )
        if limit_price <= 0:
            logger.warning("Skipping BUY %s — invalid limit price %s", sym, limit_price)
            continue
        try:
            order = _submit_limit_order(
                client,
                symbol=sym,
                qty=qty,
                side="buy",
                limit_price=limit_price,
            )
            order_id = str(getattr(order, "id", "") or "")
            print(
                f"  BUY  {sym:<8}  qty={qty:.2f}  limit=${limit_price:.2f}  "
                f"~${qty*price:,.0f}  order_id={order_id or '?'}",
                flush=True,
            )
            tlog.log("buy_submitted", symbol=sym, qty=float(qty),
                     target_notional=float(row_notional),
                     expected_price=float(limit_price),
                     reference_price=float(price),
                     order_id=order_id)
            fill_px = _poll_filled_avg_price(client, order_id) if order_id else None
            recorded = fill_px if (fill_px and fill_px > 0) else price
            try:
                record_buy_price(sym, float(recorded))
                src = "fill" if fill_px else "last_close"
                print(f"       recorded buy_price={recorded:.4f} ({src}) for guard",
                      flush=True)
            except Exception as rec_exc:
                logger.warning("record_buy_price failed for %s: %s", sym, rec_exc)
            tlog.log("buy_filled", symbol=sym,
                     fill_price=(float(fill_px) if fill_px else None),
                     fill_source=("fill" if fill_px else "last_close"),
                     slippage_bps_vs_last_close=slippage_bps(fill_px, price),
                     last_close=float(price))
        except Exception as exc:
            logger.error("BUY failed for %s: %s", sym, exc)
            tlog.log("buy_failed", symbol=sym, error=str(exc))


def _execute_sells(
    client,
    position_details: dict,
    sell_syms,
    tlog: TradeLogger,
) -> None:
    """Submit SELL orders; every sell passes through the death-spiral guard.

    ``sell_syms`` may be a set, list, or any iterable — iterated in sorted
    order for deterministic tests.
    """
    from src.alpaca_singleton import guard_sell_against_death_spiral

    for sym in sorted(sell_syms):
        det = position_details.get(sym)
        if not det:
            continue
        qty = det["qty"]
        if qty <= 0:
            continue
        current_price = det.get("current_price", 0) or det.get("avg_entry_price", 0)
        if current_price is None or current_price <= 0:
            logger.error("SELL skipped for %s — no usable price for death-spiral "
                         "guard", sym)
            continue
        # Guard raises RuntimeError on death-spiral sells → propagate (crash).
        guard_sell_against_death_spiral(sym, "sell", float(current_price))
        limit_price = _stock_limit_price_near_market(
            sym,
            side="sell",
            reference_price=float(current_price),
            aggressiveness_bps=15.0,
        )
        if limit_price <= 0:
            logger.error("SELL skipped for %s — invalid limit price %s", sym, limit_price)
            continue
        try:
            order = _submit_limit_order(
                client,
                symbol=sym,
                qty=abs(qty),
                side="sell",
                limit_price=limit_price,
            )
            order_id = str(getattr(order, "id", "") or "")
            print(
                f"  SELL {sym:<8}  qty={qty:.2f}  limit=${limit_price:.2f}  "
                f"px={current_price:.2f}  order_id={order_id or '?'}",
                flush=True,
            )
            tlog.log("sell_submitted", symbol=sym, qty=float(qty),
                     expected_price=float(limit_price),
                     reference_price=float(current_price),
                     order_id=order_id)
        except Exception as exc:
            logger.error("SELL failed for %s: %s", sym, exc)
            tlog.log("sell_failed", symbol=sym, error=str(exc))


def _position_market_value(position) -> float:
    try:
        return abs(float(getattr(position, "market_value", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _position_qty(position) -> float:
    try:
        return abs(float(getattr(position, "qty", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _position_current_price(position) -> float:
    for attr in ("current_price", "market_price", "avg_entry_price"):
        try:
            value = float(getattr(position, attr, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    qty = _position_qty(position)
    if qty > 0:
        return _position_market_value(position) / qty
    return 0.0


def _position_close_side(position) -> str:
    side = str(getattr(position, "side", "long") or "long").lower()
    return "buy" if "short" in side else "sell"


def _equity_positions_for_deleverage(client) -> list:
    positions = []
    for pos in client.get_all_positions():
        sym = str(getattr(pos, "symbol", "") or "")
        if not sym or _is_crypto_position_symbol(sym):
            continue
        if _position_qty(pos) <= 0 or _position_market_value(pos) <= 0:
            continue
        positions.append(pos)
    return positions


def _minutes_to_market_close(client) -> float | None:
    try:
        clock = client.get_clock()
    except Exception as exc:
        logger.warning("EOD deleverage clock query failed: %s", exc)
        return None
    if not bool(getattr(clock, "is_open", False)):
        return None
    next_close = getattr(clock, "next_close", None)
    if next_close is None:
        return None
    if getattr(next_close, "tzinfo", None) is None:
        next_close = next_close.replace(tzinfo=timezone.utc)
    else:
        next_close = next_close.astimezone(timezone.utc)
    return max((next_close - datetime.now(timezone.utc)).total_seconds() / 60.0, 0.0)


def _eod_deleverage_tick(client, args, *, now: datetime | None = None) -> dict:
    """Reduce equity gross exposure toward the configured EOD leverage cap.

    This runs inside the xgb live process, using the same TradingClient that
    owns the Alpaca singleton lock. It only sells/buys-to-cover excess equity
    exposure; crypto positions are skipped so the weekend sleeve is not
    accidentally flattened.
    """
    if not bool(getattr(args, "eod_deleverage", False)) or client is None:
        return {"action": "disabled"}

    mtc = _minutes_to_market_close(client)
    window = float(getattr(args, "eod_deleverage_window_minutes", 60.0) or 60.0)
    if mtc is None:
        return {"action": "closed"}
    if mtc > window:
        return {"action": "outside_window", "minutes_to_close": mtc}

    try:
        account = client.get_account()
        equity = float(getattr(account, "equity", 0.0) or 0.0)
    except Exception as exc:
        logger.warning("EOD deleverage account query failed: %s", exc)
        return {"action": "account_error", "error": str(exc)}
    if equity <= 0:
        return {"action": "bad_equity", "equity": equity}

    try:
        positions = _equity_positions_for_deleverage(client)
    except Exception as exc:
        logger.warning("EOD deleverage position query failed: %s", exc)
        return {"action": "positions_error", "error": str(exc)}

    exposure = sum(_position_market_value(p) for p in positions)
    target_lev = float(getattr(args, "eod_max_gross_leverage", 2.0) or 2.0)
    target_exposure = max(0.0, equity * target_lev)
    if exposure <= target_exposure:
        return {
            "action": "already_ok",
            "minutes_to_close": mtc,
            "equity": equity,
            "exposure": exposure,
            "leverage": exposure / equity,
            "target_leverage": target_lev,
        }

    excess = exposure - target_exposure
    force_minutes = float(getattr(args, "eod_force_market_minutes", 5.0) or 5.0)
    force_window = mtc <= force_minutes
    span = max(window - force_minutes, 1.0)
    progress = 1.0 if force_window else min(max((window - mtc) / span, 0.0), 1.0)
    progressive_excess = excess if force_window else max(excess * progress, min(excess, 500.0))

    remaining = progressive_excess
    submitted = 0
    errors: list[str] = []
    ordered = sorted(positions, key=_position_market_value, reverse=True)
    for pos in ordered:
        if remaining <= 25.0:
            break
        sym = str(getattr(pos, "symbol", "") or "")
        qty_total = _position_qty(pos)
        px = _position_current_price(pos)
        if not sym or qty_total <= 0 or px <= 0:
            continue
        value = _position_market_value(pos)
        reduce_value = min(value, remaining)
        qty = math.floor((reduce_value / px) * 10_000) / 10_000
        qty = min(qty, qty_total)
        if qty <= 0:
            continue
        side = _position_close_side(pos)
        try:
            aggressive_bps = 25.0 if force_window else (5.0 + 20.0 * progress)
            limit_price = _stock_limit_price_near_market(
                sym,
                side=side,
                reference_price=px,
                aggressiveness_bps=aggressive_bps,
            )
            order = _submit_limit_order(
                client, symbol=sym, qty=qty, side=side, limit_price=limit_price
            )
            submitted += 1
            remaining -= qty * px
            print(
                "[xgb-live/eod] deleverage "
                f"{side.upper()} {sym} qty={qty:.4f} px={px:.2f} "
                f"limit=${limit_price:.2f} mode=limit "
                f"order_id={getattr(order, 'id', '?')}",
                flush=True,
            )
        except Exception as exc:
            msg = f"{sym}: {exc}"
            logger.error("EOD deleverage order failed: %s", msg)
            errors.append(msg)

    status = {
        "action": "submitted" if submitted else "no_actionable_orders",
        "submitted": submitted,
        "errors": errors,
        "minutes_to_close": mtc,
        "equity": equity,
        "exposure": exposure,
        "leverage": exposure / equity,
        "target_leverage": target_lev,
        "target_exposure": target_exposure,
        "requested_reduction": progressive_excess,
        "use_market": False,
        "force_window": force_window,
    }
    print(f"[xgb-live/eod] {json_dumps_compact(status)}", flush=True)
    return status


def json_dumps_compact(obj: object) -> str:
    import json

    return json.dumps(obj, separators=(",", ":"), default=str)


def run_session_hold_through(
    symbols: list[str],
    data_root: Path,
    model: XGBStockModel | list[XGBStockModel],
    client,
    args: argparse.Namespace,
) -> None:
    """Hold-through variant: rotate at NEXT-OPEN instead of selling at close.

    Flow per session (fired each trading morning):
      1. Gate on Alpaca calendar (/v2/clock).
      2. Score + pick top-N.
      3. Wait for open (9:30 ET).
      4. Query currently-held positions.
      5. If held_syms == pick_syms AND both non-empty → HOLD (no trades).
         This is the win case: skip 2×(fee+buffer) and earn overnight drift.
      6. Otherwise rotate:
           SELL (old ∖ new) — each goes through guard_sell_against_death_spiral.
           BUY (new ∖ old) — each records buy price via record_buy_price.
      7. Skip the 15:50 sell entirely. Positions carry overnight; next session
         decides whether to hold or rotate.
    """
    today_str = _session_date_et().isoformat()
    paper = not args.live
    print(f"\n[xgb-live/hold-through] Session {today_str}  paper={paper}  "
          f"dry_run={args.dry_run}", flush=True)

    tlog: TradeLogger = getattr(args, "_trade_logger", None) or TradeLogger(disabled=True)
    equity_pre = _emit_session_start(tlog, args, client, mode="hold_through")
    if _live_spy_regime_gate_closed(data_root, args, trade_logger=tlog):
        tlog.log("session_skipped", reason="spy_regime_gate_closed")
        if client is None or args.dry_run:
            return
        is_trading, reason = _is_today_trading_day(client)
        if not is_trading:
            print(f"[xgb-live] {today_str} is NOT a trading day ({reason}) — "
                  f"skipping regime-gate liquidation.", flush=True)
            tlog.log("session_skipped", reason=reason)
            return
        _wait_for_market_open()
        position_details = _get_position_details(client)
        held_syms = {s for s, det in position_details.items() if det["qty"] > 0}
        if held_syms:
            print(
                "[xgb-live/hold-through] SPY regime gate closed — "
                f"selling held stock positions {sorted(held_syms)}.",
                flush=True,
            )
            tlog.log("regime_gate_liquidate", to_sell=sorted(held_syms))
            _execute_sells(client, position_details, held_syms, tlog)
        _log_session_end(tlog, client, equity_pre)
        return
    spy_vol_scale = _live_spy_vol_target_scale(data_root, args, trade_logger=tlog)
    if spy_vol_scale <= 0.0:
        tlog.log("session_skipped", reason="spy_vol_target_unavailable")
        return

    if client is not None and not args.dry_run:
        is_trading, reason = _is_today_trading_day(client)
        if not is_trading:
            print(f"[xgb-live] {today_str} is NOT a trading day ({reason}) — "
                  f"skipping session.", flush=True)
            tlog.log("session_skipped", reason=reason)
            return

    picks, _all_scores = _score_and_pick(symbols, data_root, model, args,
                                         trade_logger=tlog)
    picks, conviction_scale = _conviction_allocation_scale(
        picks, _all_scores, args, trade_logger=tlog
    )
    picks, used_fallback = _apply_no_picks_fallback(
        picks,
        _all_scores,
        args,
        today_str=today_str,
        trade_logger=tlog,
    )
    if len(picks) == 0:
        return

    _announce_picks(picks, today_str, args.top_n, tlog, tag="xgb-live/hold-through")

    if args.dry_run:
        print("[xgb-live/hold-through] DRY RUN — no orders placed.", flush=True)
        tlog.log("dry_run_end")
        return

    _wait_for_market_open()

    pick_syms = set(picks["symbol"].astype(str))
    position_details = _get_position_details(client)
    held_syms = {s for s, det in position_details.items() if det["qty"] > 0}

    if held_syms == pick_syms:
        print(f"[xgb-live/hold-through] HOLD — picks unchanged "
              f"({sorted(pick_syms)}). Skipping round-trip.", flush=True)
        tlog.log("hold", held=sorted(pick_syms))
        _log_session_end(tlog, client, equity_pre)
        return

    to_sell = held_syms - pick_syms
    to_buy = pick_syms - held_syms
    print(f"[xgb-live/hold-through] Rotation: sell={sorted(to_sell)}  "
          f"buy={sorted(to_buy)}  keep={sorted(held_syms & pick_syms)}", flush=True)
    tlog.log("rotate", to_sell=sorted(to_sell), to_buy=sorted(to_buy),
             keep=sorted(held_syms & pick_syms))

    # SELL dropped-out names FIRST (frees up buying power before buys).
    _execute_sells(client, position_details, to_sell, tlog)

    account = _get_account(client)
    portfolio_value = float(getattr(account, "portfolio_value", 0.0) or 0.0)
    fb_alloc_frac = float(getattr(args, "no_picks_fallback_alloc", 0.0) or 0.0)
    alloc = float(args.allocation) * (fb_alloc_frac if used_fallback else 1.0)
    total_notional = (
        portfolio_value * alloc * (1.0 if used_fallback else conviction_scale) * spy_vol_scale
    )
    notionals = _buy_notional_by_symbol(
        picks,
        total_notional=total_notional,
        allocation_mode=getattr(args, "allocation_mode", "equal"),
        allocation_temp=getattr(args, "allocation_temp", 1.0),
        inv_vol_target_ann=(
            0.0 if used_fallback else getattr(args, "inv_vol_target_ann", 0.0)
        ),
        inv_vol_floor=getattr(args, "inv_vol_floor", 0.05),
        inv_vol_cap=getattr(args, "inv_vol_cap", 3.0),
    )
    print(f"[xgb-live/hold-through] BUY total=${total_notional:,.0f} "
          f"alloc={alloc:.2%} "
          f"mode={getattr(args, 'allocation_mode', 'equal')} "
          f"(portfolio=${portfolio_value:,.0f})", flush=True)
    _execute_buys(
        client,
        picks,
        None,
        tlog,
        notional_by_symbol=notionals,
        target_syms=to_buy,
    )

    print(f"[xgb-live/hold-through] Rotation complete: "
          f"sold {len(to_sell)}, bought {len(to_buy)}, held across "
          f"{len(held_syms & pick_syms)}.", flush=True)
    _log_session_end(tlog, client, equity_pre)


def _log_session_end(tlog: TradeLogger, client, equity_pre: float | None) -> None:
    """Record post-session equity + computed session PnL."""
    equity_post = None
    if client is not None:
        try:
            equity_post = float(getattr(_get_account(client), "equity", 0.0) or 0.0)
        except Exception:
            equity_post = None
    pnl_abs = None
    pnl_pct = None
    if equity_pre is not None and equity_post is not None and equity_pre > 0:
        pnl_abs = equity_post - equity_pre
        pnl_pct = 100.0 * pnl_abs / equity_pre
    tlog.log("session_end",
             equity_pre=equity_pre, equity_post=equity_post,
             session_pnl_abs=pnl_abs, session_pnl_pct=pnl_pct)


def run_session(
    symbols: list[str],
    data_root: Path,
    model: XGBStockModel | list[XGBStockModel],
    client,
    args: argparse.Namespace,
) -> None:
    """Execute one full open-to-close trading session (score → buy → sell).

    ``--hold-through`` flips the rotation policy so positions are carried
    overnight when today's pick matches yesterday's — see
    :func:`run_session_hold_through`.
    """
    if getattr(args, "hold_through", False):
        return run_session_hold_through(symbols, data_root, model, client, args)

    today_str = _session_date_et().isoformat()
    paper = not args.live
    print(f"\n[xgb-live] Session {today_str}  paper={paper}  "
          f"dry_run={args.dry_run}", flush=True)

    tlog: TradeLogger = getattr(args, "_trade_logger", None) or TradeLogger(disabled=True)
    equity_pre = _emit_session_start(tlog, args, client, mode="open_to_close")
    if _live_spy_regime_gate_closed(data_root, args, trade_logger=tlog):
        tlog.log("session_skipped", reason="spy_regime_gate_closed")
        return
    spy_vol_scale = _live_spy_vol_target_scale(data_root, args, trade_logger=tlog)
    if spy_vol_scale <= 0.0:
        tlog.log("session_skipped", reason="spy_vol_target_unavailable")
        return

    # Trading-day gate: Alpaca queues DAY orders placed outside RTH to the
    # next open — fill would run on stale data. Always gate on /v2/clock.
    if client is not None and not args.dry_run:
        is_trading, reason = _is_today_trading_day(client)
        if not is_trading:
            print(f"[xgb-live] {today_str} is NOT a trading day ({reason}) — "
                  f"skipping session.", flush=True)
            tlog.log("session_skipped", reason=reason)
            return

    picks, all_scores = _score_and_pick(symbols, data_root, model, args,
                                        trade_logger=tlog)
    if len(all_scores) == 0:
        print("[xgb-live] ERROR: No scoreable symbols today.", file=sys.stderr)
        tlog.log("no_candidates")
        return
    picks, conviction_scale = _conviction_allocation_scale(
        picks, all_scores, args, trade_logger=tlog
    )
    picks, used_fallback = _apply_no_picks_fallback(
        picks,
        all_scores,
        args,
        today_str=today_str,
        trade_logger=tlog,
    )
    if len(picks) == 0:
        return

    _announce_picks(picks, today_str, args.top_n, tlog, tag="xgb-live")

    if args.dry_run:
        print("[xgb-live] DRY RUN — no orders placed.", flush=True)
        tlog.log("dry_run_end")
        return

    _wait_for_market_open()

    account = _get_account(client)
    portfolio_value = float(getattr(account, "portfolio_value", 0.0) or 0.0)
    # Fallback path scales allocation by fb_alloc_frac.
    fb_alloc_frac = float(getattr(args, "no_picks_fallback_alloc", 0.0) or 0.0)
    alloc = float(args.allocation) * (fb_alloc_frac if used_fallback else 1.0)
    total_notional = (
        portfolio_value * alloc * (1.0 if used_fallback else conviction_scale) * spy_vol_scale
    )
    notionals = _buy_notional_by_symbol(
        picks,
        total_notional=total_notional,
        allocation_mode=getattr(args, "allocation_mode", "equal"),
        allocation_temp=getattr(args, "allocation_temp", 1.0),
        inv_vol_target_ann=(
            0.0 if used_fallback else getattr(args, "inv_vol_target_ann", 0.0)
        ),
        inv_vol_floor=getattr(args, "inv_vol_floor", 0.05),
        inv_vol_cap=getattr(args, "inv_vol_cap", 3.0),
    )
    print(f"\n[xgb-live] Placing BUY orders  portfolio=${portfolio_value:,.0f}  "
          f"alloc={alloc:.2%}  total=${total_notional:,.0f}  "
          f"mode={getattr(args, 'allocation_mode', 'equal')}", flush=True)
    _execute_buys(client, picks, None, tlog, notional_by_symbol=notionals)

    print(f"\n[xgb-live] Waiting for {MARKET_CLOSE[0]:02d}:{MARKET_CLOSE[1]:02d} "
          f"ET to sell...", flush=True)
    _wait_until(MARKET_CLOSE[0], MARKET_CLOSE[1], ET)

    position_details = _get_position_details(client)
    xgb_sell_syms = {s for s in position_details if s in picks["symbol"].values}
    print(f"\n[xgb-live] Placing SELL orders for {len(xgb_sell_syms)} positions",
          flush=True)
    _execute_sells(client, position_details, xgb_sell_syms, tlog)

    print(f"[xgb-live] Session {today_str} complete.", flush=True)
    _log_session_end(tlog, client, equity_pre)


def _load_models(args: argparse.Namespace) -> XGBStockModel | list[XGBStockModel] | None:
    """Load either an ensemble (``--model-paths``) or a single model (``--model-path``).

    Returns ``None`` if any referenced path is missing — the caller converts
    that to exit code 1.
    """
    if args.model_paths.strip():
        paths = [Path(p.strip()) for p in args.model_paths.split(",") if p.strip()]
        if not paths:
            print("ERROR: --model-paths did not contain any model paths", file=sys.stderr)
            return None
        normalized_paths = [path.expanduser().resolve(strict=False) for path in paths]
        if len(set(normalized_paths)) != len(normalized_paths):
            print("ERROR: --model-paths contains duplicate model paths", file=sys.stderr)
            return None
        seeds: list[int] = []
        for path in paths:
            try:
                seeds.append(_ensemble_model_path_seed(path))
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return None
        if len(set(seeds)) != len(seeds):
            print(f"ERROR: --model-paths contains duplicate model seeds: {seeds}", file=sys.stderr)
            return None
        for mp in paths:
            if not mp.exists():
                print(f"ERROR: Ensemble model not found at {mp}", file=sys.stderr)
                return None
        print(f"[xgb-live] Loading ensemble of {len(paths)} models", flush=True)
        for mp in paths:
            print(f"[xgb-live]   - {mp}", flush=True)
        try:
            models = [XGBStockModel.load(mp) for mp in paths]
        except Exception as exc:
            print(f"ERROR: Failed to load ensemble model: {exc}", file=sys.stderr)
            return None
        if not _validate_ensemble_feature_contract(models, paths):
            return None
        return models
    if not args.model_path.exists():
        print(f"ERROR: Model not found at {args.model_path}", file=sys.stderr)
        print("Run train_alltrain.py to create it, or set --model-paths for "
              "an ensemble.", file=sys.stderr)
        return None
    print(f"[xgb-live] Loading model from {args.model_path}", flush=True)
    try:
        model = XGBStockModel.load(args.model_path)
    except Exception as exc:
        print(f"ERROR: Failed to load model: {exc}", file=sys.stderr)
        return None
    if _validated_model_features(model, args.model_path) is None:
        return None
    return model


def _ensemble_model_path_seed(path: Path) -> int:
    stem = path.stem
    prefix = "alltrain_seed"
    if not stem.startswith(prefix):
        raise ValueError(f"{path}: filename must match alltrain_seed<seed>.pkl")
    seed_text = stem[len(prefix):]
    try:
        return int(seed_text)
    except ValueError as exc:
        raise ValueError(f"{path}: filename must match alltrain_seed<seed>.pkl") from exc


def _validated_model_features(model: XGBStockModel, path: Path) -> tuple[str, ...] | None:
    raw_features = getattr(model, "feature_cols", None)
    if (
        not isinstance(raw_features, (list, tuple))
        or not raw_features
        or not all(isinstance(col, str) and col for col in raw_features)
    ):
        print(f"ERROR: Model has invalid feature_cols: {path}", file=sys.stderr)
        return None
    features = tuple(raw_features)
    unsupported = sorted(set(features) - LIVE_SUPPORTED_FEATURE_COLS)
    if unsupported:
        print(
            "ERROR: Model feature_cols contain unsupported live features: "
            f"{path}: {unsupported}",
            file=sys.stderr,
        )
        return None
    offline_only = sorted(set(features).intersection(FM_LATENT_FEATURE_COLS))
    if offline_only:
        print(
            "ERROR: Model feature_cols require offline FM latents not available "
            f"in live trading: {path}: {offline_only}",
            file=sys.stderr,
        )
        return None
    return features


def _validate_ensemble_feature_contract(
    models: list[XGBStockModel],
    paths: list[Path],
) -> bool:
    """Fail closed if ensemble members were trained on different features."""
    first_features: tuple[str, ...] | None = None
    first_path: Path | None = None
    for model, path in zip(models, paths, strict=True):
        features = _validated_model_features(model, path)
        if features is None:
            return False
        if first_features is None:
            first_features = features
            first_path = path
            continue
        if features != first_features:
            print(
                "ERROR: Ensemble feature_cols mismatch: "
                f"{path} has {len(features)} features but {first_path} has "
                f"{len(first_features)}",
                file=sys.stderr,
            )
            return False
    return True


def _next_session_open_et(now_et: datetime) -> datetime:
    """The 9:20 ET of the next non-weekend day after ``now_et``.

    Note: weekday skip is a coarse approximation — the real trading-day gate
    (``_is_today_trading_day``) handles holidays when the next session
    actually fires.
    """
    next_day = now_et.date() + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return datetime(next_day.year, next_day.month, next_day.day,
                    9, 20, 0, tzinfo=ET)


def _sleep_until_next_session(client=None, args=None) -> None:
    """Sleep until the next stock-session open (ET 9:20), optionally
    servicing the embedded crypto weekend trader on each poll.

    When `args.crypto_weekend` is True and we have a real `client`, this
    loops every `args.crypto_poll_seconds` instead of one big sleep; each
    poll calls `crypto_weekend.session.run_crypto_tick(client, ...)` so
    Saturday buys / Monday sells fire at the right UTC window.

    Stock trading remains untouched: the next session still fires at the
    same moment, and all Alpaca calls go through the single `client`
    which holds the one `alpaca_live_writer` singleton lock.
    """
    now_et = datetime.now(ET)
    next_open = _next_session_open_et(now_et)
    crypto_on = bool(args and getattr(args, "crypto_weekend", False) and client is not None)
    eod_on = bool(args and getattr(args, "eod_deleverage", False) and client is not None)

    if not crypto_on and not eod_on:
        wait_secs = (next_open - datetime.now(ET)).total_seconds()
        if wait_secs > 0:
            print(f"[xgb-live] Next session at {next_open.isoformat()}  "
                  f"(sleeping {wait_secs/3600:.1f}h)", flush=True)
            time.sleep(wait_secs)
        return

    cws = None
    if crypto_on:
        from crypto_weekend import session as cws
    poll = max(30, int(getattr(args, "crypto_poll_seconds", 300)))
    max_gross = float(getattr(args, "crypto_max_gross", 0.5))
    dry_run = bool(getattr(args, "dry_run", False))
    print(f"[xgb-live] Next session at {next_open.isoformat()}  "
          f"(service poll every {poll}s, crypto={crypto_on}, "
          f"eod_deleverage={eod_on}, crypto_max_gross={max_gross:.2f})",
          flush=True)
    while True:
        remaining = (next_open - datetime.now(ET)).total_seconds()
        if remaining <= 0:
            return
        if crypto_on and cws is not None:
            try:
                cws.run_crypto_tick(client, max_gross=max_gross, dry_run=dry_run)
            except Exception as exc:
                print(f"[xgb-live] crypto tick error: {exc}", flush=True)
        if eod_on:
            try:
                _eod_deleverage_tick(client, args)
            except Exception as exc:
                print(f"[xgb-live] eod deleverage tick error: {exc}", flush=True)
        time.sleep(min(remaining, poll))


def _make_trade_logger(args: argparse.Namespace, session_date: date) -> TradeLogger:
    trade_log_dir = args.trade_log_dir
    log_disabled = bool(args.no_trade_log) or (
        trade_log_dir is not None and str(trade_log_dir) == ""
    )
    tlog = TradeLogger(
        log_dir=trade_log_dir if trade_log_dir else REPO / "analysis/xgb_live_trade_log",
        disabled=log_disabled,
        session_date=session_date,
    )
    if not log_disabled:
        print(f"[xgb-live] trade_log -> {tlog.path}", flush=True)
    return tlog


def _enforce_live_startup_guards():
    """Fail closed before any live Alpaca client can place orders."""
    from src.alpaca_account_lock import require_explicit_live_trading_enable
    from src.alpaca_singleton import enforce_live_singleton

    require_explicit_live_trading_enable("xgb_live_trader")
    return enforce_live_singleton(
        service_name="xgb_live_trader",
        account_name="alpaca_live_writer",
        force_live=True,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    paper = not args.live

    if not paper and not args.dry_run:
        _lock = _enforce_live_startup_guards()

    model = _load_models(args)
    if model is None:
        return 1

    symbols = _load_symbols(args.symbols_file)
    print(f"[xgb-live] {len(symbols)} symbols  paper={paper}  top_n={args.top_n}  "
          f"allocation={args.allocation:.0%}", flush=True)

    if not args.dry_run:
        client = _build_trading_client(paper=paper)
        acct = _get_account(client)
        print(f"[xgb-live] Account equity=${float(getattr(acct, 'equity', 0)):,.0f}  "
              f"buying_power=${float(getattr(acct, 'buying_power', 0)):,.0f}",
              flush=True)
    else:
        client = None

    while True:
        args._trade_logger = _make_trade_logger(args, _session_date_et())
        run_session(symbols, args.data_root, model, client, args)
        if not args.loop:
            break
        _sleep_until_next_session(client=client, args=args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
