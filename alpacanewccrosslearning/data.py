from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.hourly_data_utils import resolve_hourly_symbol_path
from src.symbol_utils import is_crypto_symbol
from preaug_sweeps.augmentations import get_augmentation


DEFAULT_TARGET_COLS: Tuple[str, ...] = ("open", "high", "low", "close")


@dataclass(frozen=True)
class CovariateConfig:
    include_volume: bool = True
    include_log_volume: bool = True
    include_return_1h: bool = True
    include_range_pct: bool = True
    include_body_pct: bool = True


def _resolve_data_root(
    symbol: str,
    *,
    data_root: Optional[Path],
    crypto_root: Optional[Path],
    stock_root: Optional[Path],
) -> Optional[Path]:
    if data_root is not None:
        return Path(data_root)
    if is_crypto_symbol(symbol):
        return Path(crypto_root) if crypto_root else None
    return Path(stock_root) if stock_root else None


def _resolve_symbol_path(
    symbol: str,
    *,
    data_root: Optional[Path],
    crypto_root: Optional[Path],
    stock_root: Optional[Path],
) -> Path:
    root = _resolve_data_root(symbol, data_root=data_root, crypto_root=crypto_root, stock_root=stock_root)
    if root is not None:
        path = resolve_hourly_symbol_path(symbol, root)
        if path is not None:
            return path
        raise FileNotFoundError(f"Hourly data for {symbol} not found under {root}")
    default_root = Path("trainingdatahourly")
    path = resolve_hourly_symbol_path(symbol, default_root)
    if path is None:
        raise FileNotFoundError(f"Hourly data for {symbol} not found under {default_root}")
    return path


def load_symbol_frame(
    symbol: str,
    *,
    data_root: Optional[Path] = None,
    crypto_root: Optional[Path] = None,
    stock_root: Optional[Path] = None,
) -> pd.DataFrame:
    path = _resolve_symbol_path(
        symbol,
        data_root=data_root,
        crypto_root=crypto_root,
        stock_root=stock_root,
    )
    frame = pd.read_csv(path)
    frame.columns = [col.lower() for col in frame.columns]
    if "timestamp" not in frame.columns:
        raise KeyError(f"{path} missing timestamp column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    frame["symbol"] = symbol.upper()
    return frame.reset_index(drop=True)


def _compute_covariates(frame: pd.DataFrame, config: CovariateConfig) -> Dict[str, np.ndarray]:
    covariates: Dict[str, np.ndarray] = {}
    close = frame["close"].astype(float).to_numpy()
    open_ = frame["open"].astype(float).to_numpy()
    high = frame["high"].astype(float).to_numpy()
    low = frame["low"].astype(float).to_numpy()
    volume = frame["volume"].astype(float).to_numpy()

    if config.include_volume:
        covariates["volume"] = volume
    if config.include_log_volume:
        covariates["log_volume"] = np.log1p(np.maximum(volume, 0.0))
    if config.include_return_1h:
        ret = np.zeros_like(close)
        ret[1:] = np.divide(close[1:] - close[:-1], np.where(close[:-1] == 0, 1.0, close[:-1]))
        covariates["return_1h"] = ret
    if config.include_range_pct:
        denom = np.where(close == 0, 1.0, close)
        covariates["range_pct"] = (high - low) / denom
    if config.include_body_pct:
        denom = np.where(open_ == 0, 1.0, open_)
        covariates["body_pct"] = (close - open_) / denom

    # Ensure no NaNs
    for key, values in covariates.items():
        values = np.where(np.isfinite(values), values, 0.0)
        covariates[key] = values.astype(np.float32)

    return covariates


def build_chronos_input(
    frame: pd.DataFrame,
    *,
    target_cols: Sequence[str] = DEFAULT_TARGET_COLS,
    covariate_config: Optional[CovariateConfig] = None,
) -> Dict[str, np.ndarray | Dict[str, np.ndarray]]:
    cov_cfg = covariate_config or CovariateConfig()
    targets = frame[list(target_cols)].astype(float).to_numpy().T
    covariates = _compute_covariates(frame, cov_cfg)
    return {"target": targets, "past_covariates": covariates}


def _apply_preaug(
    frame: pd.DataFrame,
    strategy: Optional[str],
    params: Optional[Dict[str, object]],
) -> pd.DataFrame:
    if not strategy or strategy in {"baseline", "none"}:
        return frame
    augmentation = get_augmentation(strategy, **(params or {}))
    return augmentation.transform_dataframe(frame)


def split_frame(frame: pd.DataFrame, val_hours: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if val_hours <= 0 or len(frame) <= val_hours + 8:
        return frame, None
    split_idx = len(frame) - val_hours
    train = frame.iloc[:split_idx].reset_index(drop=True)
    val = frame.iloc[split_idx:].reset_index(drop=True)
    if len(val) < 4:
        return frame, None
    return train, val


def build_inputs_for_symbols(
    symbols: Iterable[str],
    *,
    data_root: Optional[Path] = None,
    crypto_root: Optional[Path] = None,
    stock_root: Optional[Path] = None,
    target_cols: Sequence[str] = DEFAULT_TARGET_COLS,
    covariate_config: Optional[CovariateConfig] = None,
    val_hours: int = 0,
    preaug_strategy: Optional[str] = None,
    preaug_params: Optional[Dict[str, object]] = None,
) -> Tuple[List[Dict[str, np.ndarray | Dict[str, np.ndarray]]], Optional[List[Dict[str, np.ndarray | Dict[str, np.ndarray]]]]]:
    train_inputs: List[Dict[str, np.ndarray | Dict[str, np.ndarray]]] = []
    val_inputs: List[Dict[str, np.ndarray | Dict[str, np.ndarray]]] = []

    for symbol in symbols:
        frame = load_symbol_frame(
            symbol,
            data_root=data_root,
            crypto_root=crypto_root,
            stock_root=stock_root,
        )
        train_frame, val_frame = split_frame(frame, val_hours)
        if preaug_strategy:
            train_frame = _apply_preaug(train_frame, preaug_strategy, preaug_params)
            if val_frame is not None:
                val_frame = _apply_preaug(val_frame, preaug_strategy, preaug_params)
        train_inputs.append(
            build_chronos_input(
                train_frame,
                target_cols=target_cols,
                covariate_config=covariate_config,
            )
        )
        if val_frame is not None:
            val_inputs.append(
                build_chronos_input(
                    val_frame,
                    target_cols=target_cols,
                    covariate_config=covariate_config,
                )
            )

    return train_inputs, (val_inputs if val_inputs else None)


__all__ = [
    "CovariateConfig",
    "DEFAULT_TARGET_COLS",
    "build_inputs_for_symbols",
    "build_chronos_input",
    "load_symbol_frame",
    "split_frame",
]
