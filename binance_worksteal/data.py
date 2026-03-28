"""Dataset for daily crypto bars with technical indicators.

Loads OHLCV from trainingdata/train/{SYM}USDT.csv, with a compatibility
fallback for legacy {SYM}USD.csv files, computes features, and creates
sequences for the neural work-steal policy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


ORIGINAL_30_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "AAVEUSDT", "LTCUSDT", "XRPUSDT", "DOTUSDT", "UNIUSDT", "NEARUSDT",
    "APTUSDT", "ICPUSDT", "SHIBUSDT", "ADAUSDT", "FILUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "SUIUSDT", "TIAUSDT", "SEIUSDT", "ATOMUSDT",
    "ALGOUSDT", "BCHUSDT", "BNBUSDT", "TRXUSDT", "PEPEUSDT", "MATICUSDT",
]

EXPANDED_SYMBOLS = [
    "HBARUSDT", "VETUSDT", "RENDERUSDT", "FETUSDT", "GRTUSDT",
    "SANDUSDT", "MANAUSDT", "AXSUSDT", "CRVUSDT", "COMPUSDT",
    "MKRUSDT", "SNXUSDT", "ENJUSDT", "1INCHUSDT", "SUSHIUSDT",
    "YFIUSDT", "BATUSDT", "ZRXUSDT", "THETAUSDT", "FTMUSDT",
    "RUNEUSDT", "KAVAUSDT", "EGLDUSDT", "CHZUSDT", "GALAUSDT",
    "APEUSDT", "LDOUSDT", "GMXUSDT", "PENDLEUSDT", "WLDUSDT",
    "JUPUSDT", "WUSDT", "ENAUSDT", "STXUSDT", "FLOKIUSDT",
    "TONUSDT", "KASUSDT", "ONDOUSDT", "JASMYUSDT", "CFXUSDT",
]

DEFAULT_SYMBOLS = ORIGINAL_30_SYMBOLS + EXPANDED_SYMBOLS

FEATURE_NAMES = [
    "open_norm", "high_norm", "low_norm", "close_norm", "volume_norm",
    "sma_20_ratio", "atr_14_norm", "rsi_14_norm",
    "has_position", "unrealized_pnl_norm", "hold_days_norm",
]

N_MARKET_FEATURES = 8  # first 8 are market features, last 3 are position state


def compute_sma(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1).fillna(df["close"])
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.clip(lower=1e-10)
    return 100.0 - 100.0 / (1.0 + rs)


def _symbol_path_candidates(data_dir: str, symbol: str) -> List[Path]:
    data_path = Path(data_dir)
    candidates = [data_path / f"{symbol}.csv"]
    if symbol.endswith("USDT"):
        candidates.append(data_path / f"{symbol[:-1]}.csv")
    elif symbol.endswith("USD"):
        candidates.append(data_path / f"{symbol}T.csv")
    return candidates


def _normalize_symbol_name(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return symbol
    if symbol.endswith("USD"):
        return f"{symbol}T"
    return symbol


def load_symbol_data(data_dir: str, symbol: str) -> Optional[pd.DataFrame]:
    path = next((candidate for candidate in _symbol_path_candidates(data_dir, symbol) if candidate.exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or len(df) < 30:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df


def validate_symbol_data(
    df: pd.DataFrame,
    min_bars: int = 365,
    max_missing_pct: float = 0.05,
) -> Tuple[bool, str]:
    if df is None or df.empty:
        return False, "no data"
    if len(df) < min_bars:
        return False, f"only {len(df)} bars, need {min_bars}"
    required_cols = {"timestamp", "open", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"missing columns: {missing}"
    ts = pd.to_datetime(df["timestamp"], utc=True).sort_values()
    if len(ts) < 2:
        return False, "fewer than 2 timestamps"
    expected_days = (ts.iloc[-1] - ts.iloc[0]).days + 1
    if expected_days <= 0:
        return False, "invalid date range"
    missing_pct = 1.0 - len(ts) / expected_days
    if missing_pct > max_missing_pct:
        return False, f"{missing_pct:.1%} missing bars (max {max_missing_pct:.0%})"
    close = df["close"].astype(float)
    if (close <= 0).any():
        return False, "non-positive close prices"
    return True, "ok"


def _avg_daily_volume_usd(df: pd.DataFrame, lookback_days: int = 90) -> float:
    if "volume" not in df.columns or df.empty:
        return 0.0
    recent = df.tail(lookback_days)
    if recent.empty:
        return 0.0
    return float((recent["volume"] * recent["close"]).mean())


def filter_symbols_by_volume(
    data_dir: str,
    symbols: Optional[List[str]] = None,
    min_avg_daily_volume_usd: float = 1_000_000.0,
    lookback_days: int = 90,
) -> List[str]:
    symbols = symbols or DEFAULT_SYMBOLS
    passed = []
    for sym in symbols:
        df = load_symbol_data(data_dir, sym)
        if df is None:
            continue
        if _avg_daily_volume_usd(df, lookback_days) >= min_avg_daily_volume_usd:
            passed.append(sym)
    return passed


def discover_symbols(
    data_dir: str,
    min_bars: int = 365,
    max_missing_pct: float = 0.05,
    min_avg_daily_volume_usd: float = 1_000_000.0,
    lookback_days: int = 90,
) -> List[str]:
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    valid = []
    seen = set()
    csv_files = sorted(data_path.glob("*USDT.csv")) + sorted(data_path.glob("*USD.csv"))
    for csv_file in csv_files:
        sym = _normalize_symbol_name(csv_file.stem)
        if sym in seen:
            continue
        seen.add(sym)
        df = load_symbol_data(data_dir, sym)
        if df is None:
            continue
        ok, _reason = validate_symbol_data(df, min_bars=min_bars, max_missing_pct=max_missing_pct)
        if not ok:
            continue
        if _avg_daily_volume_usd(df, lookback_days) < min_avg_daily_volume_usd:
            continue
        valid.append(sym)
    return valid


def generate_symbols_arg(
    data_dir: str = "trainingdata/train",
    min_bars: int = 365,
    min_avg_daily_volume_usd: float = 1_000_000.0,
    use_usd_suffix: bool = True,
) -> str:
    symbols = discover_symbols(data_dir, min_bars=min_bars, min_avg_daily_volume_usd=min_avg_daily_volume_usd)
    if use_usd_suffix:
        symbols = [s.replace("USDT", "USD") for s in symbols]
    return " ".join(symbols)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    sma20 = compute_sma(close, 20)
    atr14 = compute_atr(df, 14)
    rsi14 = compute_rsi(close, 14)

    close_safe = close.clip(lower=1e-8)
    features = pd.DataFrame({
        "open_norm": df["open"] / close_safe - 1.0,
        "high_norm": df["high"] / close_safe - 1.0,
        "low_norm": df["low"] / close_safe - 1.0,
        "close_norm": close.pct_change().fillna(0.0),
        "volume_norm": np.log1p(df["volume"]) / 20.0 if "volume" in df.columns else 0.0,
        "sma_20_ratio": close / sma20.clip(lower=1e-8) - 1.0,
        "atr_14_norm": atr14 / close_safe,
        "rsi_14_norm": rsi14 / 100.0 - 0.5,
        "has_position": 0.0,
        "unrealized_pnl_norm": 0.0,
        "hold_days_norm": 0.0,
    }, index=df.index)

    features = features.fillna(0.0).clip(-5.0, 5.0)
    return features


def _build_aligned_arrays(
    symbol_data: Dict[str, pd.DataFrame],
    symbol_features: Dict[str, pd.DataFrame],
    date_index: pd.DatetimeIndex,
    symbols: List[str],
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_dates = len(date_index)
    n_symbols = len(symbols)
    feature_array = np.zeros((n_dates, n_symbols, n_features), dtype=np.float32)
    ohlcv_array = np.zeros((n_dates, n_symbols, 4), dtype=np.float32)

    date_map = {ts: i for i, ts in enumerate(date_index)}
    for si, sym in enumerate(symbols):
        if sym not in symbol_data:
            continue
        df = symbol_data[sym]
        feats = symbol_features[sym]
        for row_idx in range(len(df)):
            ts = df["timestamp"].iloc[row_idx]
            if ts not in date_map:
                continue
            di = date_map[ts]
            feature_array[di, si, :] = feats.iloc[row_idx].values[:n_features]
            ohlcv_array[di, si, 0] = df["open"].iloc[row_idx]
            ohlcv_array[di, si, 1] = df["high"].iloc[row_idx]
            ohlcv_array[di, si, 2] = df["low"].iloc[row_idx]
            ohlcv_array[di, si, 3] = df["close"].iloc[row_idx]

    return feature_array, ohlcv_array


class WorkStealDataset(Dataset):
    """Creates sliding windows of [seq_len, n_symbols, n_features] from daily bars."""

    def __init__(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        symbol_features: Dict[str, pd.DataFrame],
        date_index: pd.DatetimeIndex,
        seq_len: int = 30,
        symbols: Optional[List[str]] = None,
    ) -> None:
        self.seq_len = seq_len
        self.symbols = symbols or sorted(symbol_data.keys())
        self.n_symbols = len(self.symbols)
        self.n_features = len(FEATURE_NAMES)

        self.date_index = date_index
        n_dates = len(date_index)
        if n_dates < seq_len + 1:
            raise ValueError(f"Need at least {seq_len+1} dates, got {n_dates}")

        self.feature_array, self.ohlcv_array = _build_aligned_arrays(
            symbol_data, symbol_features, date_index, self.symbols, self.n_features,
        )
        self.n_samples = n_dates - seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx
        end = idx + self.seq_len
        features = torch.from_numpy(self.feature_array[start:end])
        ohlcv = torch.from_numpy(self.ohlcv_array[start:end])
        target_ohlcv = torch.from_numpy(self.ohlcv_array[end])
        return {
            "features": features,
            "ohlcv": ohlcv,
            "target_open": target_ohlcv[:, 0],
            "target_high": target_ohlcv[:, 1],
            "target_low": target_ohlcv[:, 2],
            "target_close": target_ohlcv[:, 3],
            "current_close": ohlcv[-1, :, 3],
        }


class WorkStealSequentialDataset(Dataset):
    """Produces overlapping sequential windows for multi-step rollout.

    Each item: rollout_len consecutive days of (features_window, ohlcv_target).
    Shape: features [rollout_len, seq_len, n_symbols, n_features]
           target ohlcv [rollout_len, n_symbols, 4]
    Position features (indices 8-10) are zeroed here; they're filled during rollout.
    """

    def __init__(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        symbol_features: Dict[str, pd.DataFrame],
        date_index: pd.DatetimeIndex,
        seq_len: int = 30,
        rollout_len: int = 10,
        symbols: Optional[List[str]] = None,
        stride: int = 1,
    ) -> None:
        self.seq_len = seq_len
        self.rollout_len = rollout_len
        self.symbols = symbols or sorted(symbol_data.keys())
        self.n_symbols = len(self.symbols)
        self.n_features = len(FEATURE_NAMES)

        self.date_index = date_index
        n_dates = len(date_index)
        min_dates = seq_len + rollout_len
        if n_dates < min_dates:
            raise ValueError(f"Need at least {min_dates} dates, got {n_dates}")

        self.feature_array, self.ohlcv_array = _build_aligned_arrays(
            symbol_data, symbol_features, date_index, self.symbols, self.n_features,
        )

        total_windows = n_dates - seq_len - rollout_len + 1
        self.start_indices = list(range(0, total_windows, stride))
        self.n_samples = len(self.start_indices)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base = self.start_indices[idx]
        features_windows = []
        target_ohlcv_list = []
        current_closes = []

        for step in range(self.rollout_len):
            start = base + step
            end = start + self.seq_len
            target_day = end  # day after the lookback window

            feat = self.feature_array[start:end].copy()
            # Zero out position features -- filled during rollout
            feat[:, :, N_MARKET_FEATURES:] = 0.0
            features_windows.append(feat)
            target_ohlcv_list.append(self.ohlcv_array[target_day])
            current_closes.append(self.ohlcv_array[end - 1, :, 3])

        features = torch.from_numpy(np.stack(features_windows))  # [R, T, S, F]
        target_ohlcv = torch.from_numpy(np.stack(target_ohlcv_list))  # [R, S, 4]
        curr_closes = torch.from_numpy(np.stack(current_closes))  # [R, S]

        return {
            "features": features,
            "target_open": target_ohlcv[:, :, 0],
            "target_high": target_ohlcv[:, :, 1],
            "target_low": target_ohlcv[:, :, 2],
            "target_close": target_ohlcv[:, :, 3],
            "current_close": curr_closes,
        }


def _load_and_split(
    data_dir: str,
    symbols: Optional[List[str]],
    seq_len: int,
    test_days: int,
    val_days: int,
    min_extra: int = 1,
):
    symbols = symbols or DEFAULT_SYMBOLS
    symbol_data: Dict[str, pd.DataFrame] = {}
    symbol_features: Dict[str, pd.DataFrame] = {}

    loaded = []
    for sym in symbols:
        df = load_symbol_data(data_dir, sym)
        if df is not None:
            feats = compute_features(df)
            symbol_data[sym] = df
            symbol_features[sym] = feats
            loaded.append(sym)

    if not loaded:
        raise ValueError(f"No data loaded from {data_dir} for {symbols}")

    all_dates = sorted(set().union(*[set(df["timestamp"].tolist()) for df in symbol_data.values()]))
    date_index = pd.DatetimeIndex(all_dates)

    n = len(date_index)
    test_start = n - test_days
    val_start = test_start - val_days

    min_needed = seq_len + min_extra + val_days + test_days
    if n < min_needed:
        raise ValueError(f"Not enough data: {n} dates, need {min_needed}")

    train_dates = date_index[:val_start]
    val_dates = date_index[max(0, val_start - seq_len):test_start]
    test_dates = date_index[max(0, test_start - seq_len):]

    return symbol_data, symbol_features, loaded, train_dates, val_dates, test_dates


def build_datasets(
    data_dir: str = "trainingdata/train",
    symbols: Optional[List[str]] = None,
    seq_len: int = 30,
    test_days: int = 60,
    val_days: int = 30,
) -> Tuple[WorkStealDataset, WorkStealDataset, WorkStealDataset, List[str]]:
    sym_data, sym_feats, loaded, train_d, val_d, test_d = _load_and_split(
        data_dir, symbols, seq_len, test_days, val_days, min_extra=1,
    )
    train_ds = WorkStealDataset(sym_data, sym_feats, train_d, seq_len, loaded)
    val_ds = WorkStealDataset(sym_data, sym_feats, val_d, seq_len, loaded)
    test_ds = WorkStealDataset(sym_data, sym_feats, test_d, seq_len, loaded)
    return train_ds, val_ds, test_ds, loaded


def build_sequential_datasets(
    data_dir: str = "trainingdata/train",
    symbols: Optional[List[str]] = None,
    seq_len: int = 30,
    rollout_len: int = 10,
    test_days: int = 60,
    val_days: int = 30,
    stride: int = 1,
) -> Tuple[WorkStealSequentialDataset, WorkStealSequentialDataset, WorkStealSequentialDataset, List[str]]:
    sym_data, sym_feats, loaded, train_d, val_d, test_d = _load_and_split(
        data_dir, symbols, seq_len, test_days, val_days, min_extra=rollout_len,
    )
    train_ds = WorkStealSequentialDataset(sym_data, sym_feats, train_d, seq_len, rollout_len, loaded, stride)
    val_ds = WorkStealSequentialDataset(sym_data, sym_feats, val_d, seq_len, rollout_len, loaded, stride)
    test_ds = WorkStealSequentialDataset(sym_data, sym_feats, test_d, seq_len, rollout_len, loaded, stride)
    return train_ds, val_ds, test_ds, loaded


def build_dataloader(dataset, batch_size: int = 32, shuffle: bool = True,
                     num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                     num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
