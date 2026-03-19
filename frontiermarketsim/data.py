from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas can be absent in lean runtimes
    pd = None

_CRYPTO_PREFIXES = (
    "BTC",
    "ETH",
    "SOL",
    "AVAX",
    "LINK",
    "UNI",
    "AAVE",
    "ALGO",
    "DOGE",
    "LTC",
    "XRP",
    "ADA",
    "DOT",
    "BNB",
)


@dataclass(frozen=True)
class SymbolDataset:
    symbol: str
    ohlcv: torch.Tensor  # [T, 5], float32, columns=[open, high, low, close, volume]
    is_crypto: bool


def is_crypto_symbol(symbol: str) -> bool:
    upper = symbol.strip().upper()
    if upper.endswith("-USD"):
        return True
    if upper.endswith("USD") and any(upper.startswith(prefix) for prefix in _CRYPTO_PREFIXES):
        return True
    return any(upper.startswith(prefix + "-") for prefix in _CRYPTO_PREFIXES)


def _required_columns(frame: "pd.DataFrame") -> list[str]:
    columns = [str(col).strip().lower() for col in frame.columns]
    frame.columns = columns
    missing = [col for col in ("open", "high", "low", "close") if col not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return columns


def _read_ohlcv_csv(
    path: Path,
    *,
    min_rows: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> torch.Tensor:
    if pd is None:
        raise RuntimeError("pandas is required to load CSV market data.")

    frame = pd.read_csv(path)
    _required_columns(frame)

    timestamp_col = None
    if "date" in frame.columns:
        timestamp_col = "date"
    elif "timestamp" in frame.columns:
        timestamp_col = "timestamp"

    if timestamp_col is not None:
        frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
        frame = frame.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
        if start_date is not None:
            frame = frame[frame[timestamp_col] >= pd.to_datetime(start_date)]
        if end_date is not None:
            frame = frame[frame[timestamp_col] <= pd.to_datetime(end_date)]

    if "volume" not in frame.columns:
        frame["volume"] = 1.0

    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    # Remove malformed or non-positive bars and enforce OHLC consistency.
    frame = frame.replace([float("inf"), -float("inf")], float("nan"))
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    frame = frame[(frame["open"] > 1e-6) & (frame["high"] > 1e-6) & (frame["low"] > 1e-6) & (frame["close"] > 1e-6)]

    # Filter extreme bars (typically bad ticks/splits) so simulations stay realistic.
    open_px = frame["open"]
    close_ratio = frame["close"] / open_px
    high_ratio = frame["high"] / open_px
    low_ratio = frame["low"] / open_px
    realistic_mask = (
        close_ratio.between(0.5, 1.5)
        & high_ratio.between(0.5, 1.8)
        & low_ratio.between(0.2, 1.5)
    )
    frame = frame[realistic_mask]

    frame["high"] = frame[["open", "high", "low", "close"]].max(axis=1)
    frame["low"] = frame[["open", "high", "low", "close"]].min(axis=1)
    if len(frame) < min_rows:
        raise ValueError(f"Not enough rows in {path.name}: need >= {min_rows}, got {len(frame)}")

    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype="float32", copy=True)
    tensor = torch.from_numpy(values).to(torch.float32).contiguous()
    return tensor


def load_symbol_datasets(
    data_root: str | Path,
    *,
    symbols: Sequence[str] | None = None,
    max_symbols: int | None = None,
    min_rows: int = 768,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[SymbolDataset]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    requested = None
    if symbols:
        requested = {sym.strip().upper() for sym in symbols if sym and sym.strip()}
        if not requested:
            raise ValueError("symbols was provided but no valid symbol names were found.")

    csv_paths = sorted(root.glob("**/*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {root}")

    datasets: list[SymbolDataset] = []
    for path in csv_paths:
        symbol = path.stem.strip().upper()
        if requested is not None and symbol not in requested:
            continue
        try:
            ohlcv = _read_ohlcv_csv(
                path,
                min_rows=min_rows,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception:
            continue
        datasets.append(
            SymbolDataset(
                symbol=symbol,
                ohlcv=ohlcv,
                is_crypto=is_crypto_symbol(symbol),
            )
        )
        if max_symbols is not None and len(datasets) >= max_symbols:
            break

    if not datasets:
        raise RuntimeError(
            "No valid symbol datasets were loaded. Check symbol filters, row counts, and CSV columns."
        )
    return datasets


def align_symbol_lengths(
    datasets: Iterable[SymbolDataset],
    *,
    target_length: int | None = None,
) -> list[SymbolDataset]:
    items = list(datasets)
    if not items:
        raise ValueError("Cannot align an empty dataset collection.")

    if target_length is None:
        target_length = min(int(item.ohlcv.shape[0]) for item in items)
    if target_length <= 0:
        raise ValueError(f"target_length must be > 0, got {target_length}")

    aligned: list[SymbolDataset] = []
    for item in items:
        if item.ohlcv.shape[0] < target_length:
            raise ValueError(
                f"Symbol {item.symbol} has only {item.ohlcv.shape[0]} rows; need >= {target_length}"
            )
        aligned.append(
            SymbolDataset(
                symbol=item.symbol,
                ohlcv=item.ohlcv[-target_length:].contiguous(),
                is_crypto=item.is_crypto,
            )
        )
    return aligned
