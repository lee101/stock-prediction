"""SPY-based market regime detection for the production trading system.

A simple trend-following regime filter: only allow opening new long positions
when SPY is above its N-day moving average.  If SPY is below the MA (bear
regime), we skip opening positions but do NOT force-close any existing ones.

Usage::

    from src.market_regime import is_bull_regime, regime_filter_reason

    ok, reason = regime_filter_reason(data_dir="trainingdata")
    if not ok:
        allow_open = False
        allow_open_reason = reason
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)


def is_bull_regime(
    data_dir: str | Path = "trainingdata",
    lookback: int = 50,
    symbol: str = "SPY",
) -> bool:
    """Return True if *symbol* close is above its *lookback*-day moving average.

    Defaults to SPY with a 50-day MA. Returns True (allow trading) if:
    - The CSV file does not exist (fail-open so we don't block on missing data)
    - The CSV has fewer than *lookback* rows
    - SPY close > SPY MA(lookback)

    MA50 chosen over MA20 after `scripts/regime_filter_realism_gate.py`
    showed MA50 lifts the prod ensemble realism gate's bull-cohort
    median monthly +1.16% (6.89→8.05%) and p10 +1.11% (2.34→3.45%) vs
    MA20, by catching slow-cycle tops that MA20 keeps flipping back to
    "bull" on. See docs/regime_filter_gate/regime_split_summary.md.
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not available — assuming bull regime")
        return True

    csv_path = Path(data_dir) / f"{symbol}.csv"
    if not csv_path.exists():
        log.warning("regime check: %s not found, assuming bull regime", csv_path)
        return True

    try:
        df = pd.read_csv(csv_path, usecols=["close"])
        if len(df) < lookback:
            log.warning(
                "regime check: %s has only %d rows (need %d), assuming bull",
                csv_path,
                len(df),
                lookback,
            )
            return True
        tail = df["close"].tail(lookback + 5)
        ma = tail.rolling(lookback).mean().iloc[-1]
        close = tail.iloc[-1]
        bull = bool(close > ma)
        log.info(
            "regime check: %s close=%.2f ma%d=%.2f → %s",
            symbol,
            close,
            lookback,
            ma,
            "BULL" if bull else "BEAR",
        )
        return bull
    except Exception as exc:
        log.warning("regime check failed (%s), assuming bull regime", exc)
        return True


def regime_filter_reason(
    data_dir: str | Path = "trainingdata",
    lookback: int = 50,
    symbol: str = "SPY",
) -> Tuple[bool, Optional[str]]:
    """Return (allow_open, reason_string_or_None).

    If in a bear regime, returns (False, "<reason>").
    If in a bull regime, returns (True, None).
    """
    try:
        import pandas as pd
    except ImportError:
        return True, None

    csv_path = Path(data_dir) / f"{symbol}.csv"
    if not csv_path.exists():
        return True, None

    try:
        df = pd.read_csv(csv_path, usecols=["close"])
        if len(df) < lookback:
            return True, None
        tail = df["close"].tail(lookback + 5)
        ma = tail.rolling(lookback).mean().iloc[-1]
        close = tail.iloc[-1]
        if close <= ma:
            reason = (
                f"bear regime: {symbol} close={close:.2f} < MA{lookback}={ma:.2f}"
            )
            return False, reason
        return True, None
    except Exception:
        return True, None
