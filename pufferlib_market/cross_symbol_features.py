"""Cross-symbol correlation features for RL trading.

These features help the policy learn inter-symbol relationships:
- How correlated is each symbol with the "anchor" (BTC for crypto, SPY for stocks)
- Rolling beta (sensitivity to market moves)
- Relative strength (how much a symbol outperforms its basket)
- Breadth rank (rank of symbol among all symbols by recent return)

The features are computed over a rolling window and normalized to roughly [-1, 1].
They extend the per-symbol feature vector from 16 to 20 features.

NOTE on C environment compatibility:
  The C trading_env uses a compile-time constant FEATURES_PER_SYM=16.
  Binary files produced with cross_features=True write features_per_sym=20 in
  the header, which requires rebuilding the C extension with
      #define FEATURES_PER_SYM 20
  before training.  Without that rebuild the C env will misread the layout.
  Default exports (cross_features=False) remain byte-for-byte identical to
  previous output and are fully compatible with the current compiled .so.
"""

from __future__ import annotations

import numpy as np

CROSS_FEATURES = 4  # number of new features added per symbol


def _forward_fill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values along axis 0 (time axis), vectorised over columns."""
    out = arr.copy()
    # Build a mask of valid (non-NaN) positions and propagate the last valid index.
    mask = ~np.isnan(out)          # [T, S] bool
    # For each column, fill forward: at position t keep the last t' <= t where mask[t', s] is True.
    idx = np.where(mask, np.arange(arr.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = out[idx, np.arange(arr.shape[1])[None, :]]
    return out


def compute_cross_features(
    prices: np.ndarray,  # shape [T, S] of close prices
    symbols: list[str],
    window: int = 24,
    anchor_symbol: str = "BTC",
) -> np.ndarray:
    """Compute 4 cross-symbol features per symbol.

    Args:
        prices:        Float array [T, S] of close prices (one column per symbol).
        symbols:       List of S symbol names corresponding to columns in prices.
        window:        Rolling window in bars (default 24 — one day of hourly bars).
        anchor_symbol: Name of the anchor symbol for correlation/beta.  Case-
                       insensitive prefix match is tried first, then falls back
                       to the first symbol in the list.

    Returns:
        Array of shape [T, S, 4]:
            [..., 0]  rolling_corr_anchor  — Pearson correlation with anchor [window bars], clipped to [-1, 1]
            [..., 1]  rolling_beta         — beta vs anchor (cov/var), clipped to [-5, 5] then /5 → [-1, 1]
            [..., 2]  relative_return      — symbol_return_24h minus mean_return_24h, clipped [-1, 1]
            [..., 3]  breadth_rank         — rank of symbol by 24h return among all symbols, [0, 1]
    """
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 2:
        raise ValueError(f"prices must be 2-D [T, S], got shape {prices.shape}")
    T, S = prices.shape
    if S == 0 or T == 0:
        raise ValueError(f"prices must be non-empty, got shape {prices.shape}")
    if len(symbols) != S:
        raise ValueError(f"len(symbols)={len(symbols)} != prices.shape[1]={S}")

    # Forward-fill NaN values before computing returns.
    prices = _forward_fill_nan(prices)

    # Replace any remaining leading NaNs (before first valid price) with the
    # first valid price for that symbol so returns start at 0.
    for s in range(S):
        col = prices[:, s]
        first_valid = np.nanargmax(~np.isnan(col)) if np.any(~np.isnan(col)) else 0
        if np.isnan(col[first_valid]):
            prices[:, s] = 1.0  # all NaN — replace with constant
        else:
            prices[: first_valid + 1, s] = col[first_valid]

    # Determine anchor symbol index.
    anchor_upper = anchor_symbol.upper()
    anchor_idx = next(
        (i for i, sym in enumerate(symbols) if sym.upper().startswith(anchor_upper)),
        None,
    )
    if anchor_idx is None:
        anchor_idx = 0  # fallback to first symbol

    # Compute 1-bar log returns (safer than pct_change for ratio robustness).
    # Shape [T, S].  t=0 gets 0.0.
    safe_prices = np.where(prices > 0.0, prices, 1e-8)
    log_ret = np.zeros((T, S), dtype=np.float64)
    log_ret[1:] = np.log(safe_prices[1:] / safe_prices[:-1])

    anchor_ret = log_ret[:, anchor_idx]  # [T]

    out = np.zeros((T, S, CROSS_FEATURES), dtype=np.float32)

    # Cumulative sums used for O(1) rolling-window statistics.
    # ret_cumsum is shared by features 0/1/2/3 — compute once.
    ret_cumsum = np.cumsum(log_ret, axis=0)   # [T, S]

    # Helper: sliding window sum via prefix-sum difference.
    # roll_window_sum[t] = sum of log_ret[max(0,t-window+1)..t] for each symbol.
    pad_zeros_S = np.zeros((window, S))
    padded = np.vstack([pad_zeros_S, ret_cumsum])
    roll_ret = ret_cumsum - padded[:T]        # [T, S]  (window sum of log-returns)

    # Feature 2: relative_return — symbol window-return minus basket mean.
    basket_mean = roll_ret.mean(axis=1, keepdims=True)  # [T, 1]
    out[:, :, 2] = np.clip(roll_ret - basket_mean, -1.0, 1.0).astype(np.float32)

    # Feature 3: breadth_rank — rank each symbol by window-return, normalised to [0, 1].
    if S == 1:
        out[:, :, 3] = 0.5
    else:
        ranks = np.argsort(np.argsort(roll_ret, axis=1), axis=1).astype(np.float32)
        out[:, :, 3] = (ranks / (S - 1)).astype(np.float32)

    # Features 0 and 1: rolling correlation and beta with anchor (vectorised).
    anchor_ret_cs = np.cumsum(anchor_ret)          # [T]
    anchor_ret_sq_cs = np.cumsum(anchor_ret ** 2)  # [T]
    pad_zeros_1 = np.zeros(window)
    padded_a  = np.concatenate([pad_zeros_1, anchor_ret_cs])
    padded_a2 = np.concatenate([pad_zeros_1, anchor_ret_sq_cs])

    win_len = np.minimum(np.arange(1, T + 1), window).astype(np.float64)  # [T]
    anchor_win_sum = anchor_ret_cs    - padded_a[:T]    # [T]
    anchor_win_sq  = anchor_ret_sq_cs - padded_a2[:T]   # [T]
    anchor_mean = anchor_win_sum / win_len               # [T]
    anchor_var  = anchor_win_sq  / win_len - anchor_mean ** 2  # [T]
    anchor_std  = np.sqrt(np.maximum(anchor_var, 1e-12))       # [T]

    sym_ret_sq_cs = np.cumsum(log_ret ** 2, axis=0)
    sym_cross_cs  = np.cumsum(log_ret * anchor_ret[:, None], axis=0)

    padded_s  = np.vstack([pad_zeros_S, ret_cumsum])   # reuse ret_cumsum
    padded_s2 = np.vstack([pad_zeros_S, sym_ret_sq_cs])
    padded_sc = np.vstack([pad_zeros_S, sym_cross_cs])

    sym_win_sum = ret_cumsum    - padded_s[:T]   # [T, S]
    sym_win_sq  = sym_ret_sq_cs - padded_s2[:T]
    sym_cross   = sym_cross_cs  - padded_sc[:T]

    sym_mean = sym_win_sum / win_len[:, None]    # [T, S]
    sym_var = sym_win_sq / win_len[:, None] - sym_mean ** 2  # [T, S]
    sym_std = np.sqrt(np.maximum(sym_var, 1e-12))             # [T, S]

    cov = sym_cross / win_len[:, None] - sym_mean * anchor_mean[:, None]  # [T, S]

    # rolling_corr_anchor: cov / (sym_std * anchor_std)
    denom_corr = sym_std * anchor_std[:, None]
    corr = np.where(denom_corr > 1e-12, cov / denom_corr, 0.0)
    out[:, :, 0] = np.clip(corr, -1.0, 1.0).astype(np.float32)

    # rolling_beta: cov / anchor_var, normalised to [-1, 1] by dividing by 5
    denom_var = np.maximum(anchor_var[:, None], 1e-12)
    beta = cov / denom_var
    out[:, :, 1] = np.clip(beta / 5.0, -1.0, 1.0).astype(np.float32)

    return out
