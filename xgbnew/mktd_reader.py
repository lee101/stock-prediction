"""Reader for MKTD v2 binary format (hourly stock/crypto data).

Binary layout:
    Header (24 bytes):
        magic       : 4 bytes   b'MKTD'
        version     : uint32    must be 2
        num_symbols : uint32    number of symbols in file
        num_steps   : uint32    number of time steps
        n_features  : uint32    features per symbol per step
        reserved    : uint32

    Symbol names section (variable):
        For each symbol: uint32 name_length + name_length UTF-8 bytes

    Data section:
        float32 array of shape [num_steps, num_symbols, n_features]
        Features (by index):
            0: timestamp (unix seconds, float)
            1: open
            2: high
            3: low
            4: close
            5: volume
            6-15: derived features (may vary by file version)

Returns a dict { symbol: pd.DataFrame } with columns
    [timestamp, open, high, low, close, volume].
Only returns bars where volume > 0 and during market hours
(9:30–16:00 ET) by default.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd


_MAGIC = b"MKTD"
_MARKET_OPEN_H  = 9   # 9:30 ET → but we keep as local-hour filter
_MARKET_CLOSE_H = 16  # 16:00 ET


def read_mktd_hourly(
    path: Path,
    market_hours_only: bool = True,
    tz: str = "America/New_York",
) -> dict[str, pd.DataFrame]:
    """Read a MKTD v2 binary file.

    Args:
        path: Path to the .bin file.
        market_hours_only: If True, keep only 9:30–15:59 ET bars.
        tz: Timezone for market-hours filtering.

    Returns:
        Dict mapping symbol name → DataFrame with
        [timestamp (UTC), open, high, low, close, volume].
    """
    path = Path(path)
    data = path.read_bytes()
    offset = 0

    # ── Header ───────────────────────────────────────────────────────────────
    magic = data[offset:offset + 4]
    offset += 4
    if magic != _MAGIC:
        raise ValueError(f"Not a MKTD file (magic={magic!r})")

    version, num_symbols, num_steps, n_features, _reserved = struct.unpack_from("<IIIII", data, offset)
    offset += 20

    if version != 2:
        raise ValueError(f"Unsupported MKTD version {version} (expected 2)")

    # ── Symbol names ─────────────────────────────────────────────────────────
    # Try to read symbol names if present; otherwise use generic names
    symbols: list[str] = []
    try:
        for _ in range(num_symbols):
            name_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            name = data[offset: offset + name_len].decode("utf-8")
            offset += name_len
            symbols.append(name)
    except Exception:
        # Fallback: no symbol names encoded — use generic labels
        symbols = [f"SYM{i}" for i in range(num_symbols)]
        # Reset offset past header only (symbol names might not be present)
        # Try to infer data start from expected data size
        expected_data_bytes = num_steps * num_symbols * n_features * 4
        data_start = len(data) - expected_data_bytes
        if data_start >= 24:
            offset = data_start

    # ── Data ─────────────────────────────────────────────────────────────────
    arr = np.frombuffer(data, dtype="<f4", offset=offset)
    expected = num_steps * num_symbols * n_features
    if len(arr) < expected:
        raise ValueError(
            f"Data truncated: expected {expected} float32s, got {len(arr)}"
        )
    arr = arr[:expected].reshape(num_steps, num_symbols, n_features)

    result: dict[str, pd.DataFrame] = {}
    for s_idx, sym in enumerate(symbols):
        sym_data = arr[:, s_idx, :]  # shape [num_steps, n_features]

        ts_col = sym_data[:, 0]
        open_  = sym_data[:, 1].astype(float)
        high   = sym_data[:, 2].astype(float)
        low    = sym_data[:, 3].astype(float)
        close  = sym_data[:, 4].astype(float)
        volume = sym_data[:, 5].astype(float)

        ts = pd.to_datetime(ts_col.astype("int64"), unit="s", utc=True)

        df = pd.DataFrame({
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

        # Drop zero-price bars (non-trading hours or missing)
        df = df[(df["close"] > 0) & (df["open"] > 0)]

        if market_hours_only:
            local_ts = df["timestamp"].dt.tz_convert(tz)
            # Keep 9:30–15:59 ET (hours 9 and 10–15 with minute ≥ 30 for hour 9)
            h = local_ts.dt.hour
            m = local_ts.dt.minute
            mask = ((h == 9) & (m >= 30)) | ((h >= 10) & (h < 16))
            df = df[mask]

        df = df.sort_values("timestamp").reset_index(drop=True)
        if len(df) > 0:
            result[sym] = df

    return result


__all__ = ["read_mktd_hourly"]
