"""Bid-ask spread estimation from OHLC data.

Two estimators:
  - corwin_schultz_spread_bps: Corwin & Schultz (2012) high-low estimator
  - volume_based_spread_bps: dollar-volume heuristic (fallback)

Both return estimated half-spread in basis points (bps).

References:
  Corwin, S.A. and Schultz, P. (2012).
  "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices."
  Journal of Finance 67(2), 719-760.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def corwin_schultz_spread_bps(df: pd.DataFrame, window: int = 20) -> float:
    """Estimate effective bid-ask spread in bps using Corwin-Schultz (2012).

    Uses the last ``window`` adjacent day pairs from the OHLC dataframe.
    Returns the mean spread in basis points over those pairs.

    Args:
        df: DataFrame with columns 'high', 'low'. Any extra columns are ignored.
        window: Rolling window size (number of day pairs to average over).

    Returns:
        Estimated spread in bps, or NaN if insufficient data or degenerate inputs.
    """
    if df is None or len(df) < 2:
        return float("nan")

    h = np.asarray(df["high"].values, dtype=float)
    lo = np.asarray(df["low"].values, dtype=float)

    if len(h) < 2:
        return float("nan")

    # Use the last (window + 1) rows so we get window pairs
    n = min(window + 1, len(h))
    h = h[-n:]
    lo = lo[-n:]

    # log(H/L) per day; guard against zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.where((h > 0) & (lo > 0), np.log(h / lo), 0.0)

    sqrt2 = np.sqrt(2.0)
    k = 3.0 - 2.0 * sqrt2  # ≈ 0.17157

    spreads: list[float] = []
    for i in range(1, n):
        beta_i = log_hl[i] ** 2 + log_hl[i - 1] ** 2

        h2 = max(h[i], h[i - 1])
        lo2 = min(lo[i], lo[i - 1])
        if lo2 <= 0 or h2 <= 0 or h2 < lo2:
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            gamma_i = np.log(h2 / lo2) ** 2

        if not np.isfinite(gamma_i) or not np.isfinite(beta_i):
            continue

        alpha_i = (np.sqrt(2.0 * beta_i) - np.sqrt(beta_i)) / k - np.sqrt(gamma_i / k)

        if not np.isfinite(alpha_i):
            continue

        exp_alpha = np.exp(alpha_i)
        # Fractional spread = 2(e^α - 1) / (1 + e^α)
        spread_frac = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
        # Clip: negative estimates are set to 0 (Corwin-Schultz recommendation)
        spreads.append(max(0.0, spread_frac))

    if not spreads:
        return float("nan")

    return float(np.mean(spreads)) * 10_000.0  # fractional → bps


def volume_based_spread_bps(df: pd.DataFrame, window: int = 20) -> float:
    """Estimate bid-ask spread in bps from average daily dollar volume.

    Tiers derived from empirical US equity literature:
      > $500M/day : ~3 bps  (mega-cap, ultra-liquid)
      > $100M/day : ~7 bps
      >  $50M/day : ~12 bps
      >  $10M/day : ~25 bps
      otherwise   : ~50 bps  (illiquid micro-cap)

    Args:
        df: DataFrame with columns 'close' and 'volume'.
        window: Number of recent days to average over.

    Returns:
        Estimated spread in bps.
    """
    if df is None or len(df) < 1:
        return 50.0

    n = min(window, len(df))
    close = pd.to_numeric(df["close"].iloc[-n:], errors="coerce")
    volume = pd.to_numeric(df["volume"].iloc[-n:], errors="coerce")

    if close.isna().all() or volume.isna().all():
        return 50.0

    dollar_vol = float((close * volume).mean(skipna=True))
    if not np.isfinite(dollar_vol) or dollar_vol <= 0:
        return 50.0

    if dollar_vol > 500e6:
        return 3.0
    if dollar_vol > 100e6:
        return 7.0
    if dollar_vol > 50e6:
        return 12.0
    if dollar_vol > 10e6:
        return 25.0
    return 50.0


def estimate_spread_bps(
    df: pd.DataFrame,
    window: int = 20,
    cs_max_bps: float = 200.0,
    fallback: str = "volume",
) -> float:
    """Estimate bid-ask spread in bps with fallback.

    First tries Corwin-Schultz (2012); if the result is NaN or implausibly
    large, falls back to the volume-based heuristic.

    Args:
        df: DataFrame with columns 'high', 'low', 'close', 'volume'.
        window: Rolling window size.
        cs_max_bps: Maximum plausible C-S spread; higher values trigger fallback.
        fallback: Fallback method if C-S fails: 'volume' (default) or 'fixed'.

    Returns:
        Estimated spread in bps (positive float).
    """
    cs = corwin_schultz_spread_bps(df, window=window)
    if np.isfinite(cs) and 0.0 < cs <= cs_max_bps:
        return cs

    if fallback == "volume":
        return volume_based_spread_bps(df, window=window)
    return 25.0  # generic mid-tier fallback


__all__ = [
    "corwin_schultz_spread_bps",
    "volume_based_spread_bps",
    "estimate_spread_bps",
]
