"""Live RL signal generator for Binance spot trading.

Supports both legacy 4-symbol checkpoints and pufferlib portfolio-level
models (e.g. mixed23 with obs_dim=396, action_dim=47).

Architecture and symbol count are auto-detected from the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

FEATURES_PER_SYM = 16
INITIAL_CASH = 10000.0

# Legacy defaults (backward compat for importers)
SYMBOLS = ("BTCUSD", "ETHUSD", "DOGEUSD", "AAVEUSD")
NUM_SYMBOLS = 4
OBS_SIZE = NUM_SYMBOLS * FEATURES_PER_SYM + 5 + NUM_SYMBOLS  # 73
NUM_ACTIONS = 1 + 2 * NUM_SYMBOLS  # 9

ACTION_NAMES = [
    "FLAT",
    "LONG_BTC",
    "LONG_ETH",
    "LONG_DOGE",
    "LONG_AAVE",
    "SHORT_BTC",
    "SHORT_ETH",
    "SHORT_DOGE",
    "SHORT_AAVE",
]

# Mixed23 symbol order (must match pufferlib_market/data/mixed23_*.bin)
MIXED23_SYMBOLS = (
    "AAPL",
    "NFLX",
    "NVDA",
    "ADBE",
    "ADSK",
    "COIN",
    "GOOG",
    "MSFT",
    "PYPL",
    "SAP",
    "TSLA",
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "LTCUSD",
    "AVAXUSD",
    "DOGEUSD",
    "LINKUSD",
    "AAVEUSD",
    "UNIUSD",
    "DOTUSD",
    "SHIBUSD",
    "XRPUSD",
)

# Map from symbol name -> Binance trading pair for live kline fetch
SYMBOL_TO_BINANCE_PAIR = {
    "BTCUSD": "BTCFDUSD",
    "ETHUSD": "ETHFDUSD",
    "SOLUSD": "SOLUSDT",
    "DOGEUSD": "DOGEUSDT",
    "AAVEUSD": "AAVEUSDT",
    "LINKUSD": "LINKUSDT",
    "XRPUSD": "XRPUSDT",
    "LTCUSD": "LTCUSDT",
    "AVAXUSD": "AVAXUSDT",
    "UNIUSD": "UNIUSDT",
    "DOTUSD": "DOTUSDT",
    "SHIBUSD": "SHIBUSDT",
    "ADAUSD": "ADAUSDT",
    "ALGOUSD": "ALGOUSDT",
    "MATICUSD": "MATICUSDT",
}

CRYPTO15_SYMBOLS = (
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "LTCUSD",
    "AVAXUSD",
    "DOGEUSD",
    "LINKUSD",
    "ADAUSD",
    "UNIUSD",
    "AAVEUSD",
    "ALGOUSD",
    "DOTUSD",
    "SHIBUSD",
    "XRPUSD",
    "MATICUSD",
)

# Known obs_size -> symbol tuple mapping for auto-detection
_OBS_SIZE_TO_SYMBOLS = {
    73: ("BTCUSD", "ETHUSD", "DOGEUSD", "AAVEUSD"),
    260: CRYPTO15_SYMBOLS,
    396: MIXED23_SYMBOLS,
}


def _build_action_names(symbols: tuple[str, ...]) -> list[str]:
    names = ["FLAT"]
    for sym in symbols:
        short = sym.replace("USD", "")
        names.append(f"LONG_{short}")
    for sym in symbols:
        short = sym.replace("USD", "")
        names.append(f"SHORT_{short}")
    return names


# ---------------------------------------------------------------------------
# Policy architecture (must match training exactly)
# ---------------------------------------------------------------------------


class RunningObsNorm(nn.Module):
    def __init__(self, size, eps=1e-5, clip=10.0):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self.register_buffer("running_mean", torch.zeros(size))
        self.register_buffer("running_var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))

    def forward(self, x):
        return ((x - self.running_mean) / (self.running_var.sqrt() + self.eps)).clamp(-self.clip, self.clip)


class TradingPolicy(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 1024, use_obs_norm: bool = True):
        super().__init__()
        self._use_obs_norm = use_obs_norm
        if use_obs_norm:
            self.obs_norm = RunningObsNorm(obs_size)
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        if self._use_obs_norm:
            x = self.obs_norm(x)
        h = self.encoder(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature computation (mirrors pufferlib_market/export_data_hourly_forecast.py)
# ---------------------------------------------------------------------------


def _load_forecast_parquet(cache_root: Path, horizon: int, symbol: str) -> pd.DataFrame:
    path = cache_root / f"h{horizon}" / f"{symbol.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    if "issued_at" in frame.columns:
        frame["issued_at"] = pd.to_datetime(frame["issued_at"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["issued_at"]).drop_duplicates(subset="issued_at", keep="last")
        frame = frame.set_index("issued_at")
    elif "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).drop_duplicates(subset="timestamp", keep="last")
        frame = frame.set_index("timestamp")
    frame.index = frame.index.floor("h")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _forecast_delta(forecast: pd.DataFrame, index: pd.DatetimeIndex, col: str, close: pd.Series) -> pd.Series:
    if col not in forecast.columns or forecast.empty:
        return pd.Series(0.0, index=index, dtype=np.float32)
    aligned = forecast[col].reindex(index)
    aligned = pd.to_numeric(aligned, errors="coerce")
    delta = (aligned - close) / close.clip(lower=1e-8)
    return delta.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _forecast_confidence(forecast: pd.DataFrame, index: pd.DatetimeIndex, close: pd.Series) -> pd.Series:
    p90, p10 = "predicted_close_p90", "predicted_close_p10"
    if p90 not in forecast.columns or p10 not in forecast.columns or forecast.empty:
        return pd.Series(0.5, index=index, dtype=np.float32)
    spread = (forecast[p90] - forecast[p10]).reindex(index).abs()
    spread = pd.to_numeric(spread, errors="coerce")
    conf = 1.0 / (1.0 + spread / close.clip(lower=1e-8))
    return conf.replace([np.inf, -np.inf], np.nan).fillna(0.5).astype(np.float32)


def compute_symbol_features(
    price_df: pd.DataFrame,
    fc_h1: pd.DataFrame,
    fc_h24: pd.DataFrame,
) -> np.ndarray:
    """Compute 16 features for one symbol. Returns [T, 16] float32 array."""
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    idx = price_df.index

    feat = pd.DataFrame(index=idx)
    feat["chronos_close_delta_h1"] = _forecast_delta(fc_h1, idx, "predicted_close_p50", close)
    feat["chronos_high_delta_h1"] = _forecast_delta(fc_h1, idx, "predicted_high_p50", close)
    feat["chronos_low_delta_h1"] = _forecast_delta(fc_h1, idx, "predicted_low_p50", close)
    feat["chronos_close_delta_h24"] = _forecast_delta(fc_h24, idx, "predicted_close_p50", close)
    feat["chronos_high_delta_h24"] = _forecast_delta(fc_h24, idx, "predicted_high_p50", close)
    feat["chronos_low_delta_h24"] = _forecast_delta(fc_h24, idx, "predicted_low_p50", close)
    feat["forecast_confidence_h1"] = _forecast_confidence(fc_h1, idx, close)
    feat["forecast_confidence_h24"] = _forecast_confidence(fc_h24, idx, close)

    ret_1h = close.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat["return_1h"] = ret_1h.clip(-0.5, 0.5).astype(np.float32)
    feat["return_24h"] = (
        close.pct_change(24).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
    )
    feat["volatility_24h"] = ret_1h.rolling(24, min_periods=1).std(ddof=0).fillna(0.0).clip(0.0, 1.0).astype(np.float32)

    ma24 = close.rolling(24, min_periods=1).mean()
    ma72 = close.rolling(72, min_periods=1).mean()
    feat["ma_delta_24h"] = ((close - ma24) / ma24.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5).astype(np.float32)
    feat["ma_delta_72h"] = ((close - ma72) / ma72.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5).astype(np.float32)

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr24 = tr.rolling(24, min_periods=1).mean()
    feat["atr_pct_24h"] = (atr24 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5).astype(np.float32)
    feat["trend_72h"] = (
        close.pct_change(72).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
    )

    roll_max = close.rolling(72, min_periods=1).max()
    feat["drawdown_72h"] = (
        ((close - roll_max) / roll_max.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0).astype(np.float32)
    )

    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Portfolio state tracker
# ---------------------------------------------------------------------------


@dataclass
class PortfolioSnapshot:
    """Current portfolio state for obs construction."""

    cash_usd: float = 0.0
    total_value_usd: float = 0.0
    position_symbol: str | None = None
    position_value_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    hold_hours: int = 0
    is_short: bool = False


# ---------------------------------------------------------------------------
# Main signal generator
# ---------------------------------------------------------------------------


@dataclass
class RLSignal:
    action: int
    action_name: str
    target_symbol: str | None
    direction: str  # "long", "flat"
    logits: list[float] = field(default_factory=list)
    value: float = 0.0


def _infer_obs_size(state_dict: dict) -> int:
    if "encoder.0.weight" in state_dict:
        return int(state_dict["encoder.0.weight"].shape[1])
    if "input_proj.weight" in state_dict:
        return int(state_dict["input_proj.weight"].shape[1])
    raise ValueError("Cannot infer obs_size from checkpoint state_dict")


def _infer_num_actions(state_dict: dict) -> int:
    for key in ("actor.2.bias", "actor.2.weight"):
        if key in state_dict:
            t = state_dict[key]
            return int(t.shape[0])
    raise ValueError("Cannot infer num_actions from checkpoint state_dict")


def _infer_hidden_size(state_dict: dict) -> int:
    if "encoder.0.weight" in state_dict:
        return int(state_dict["encoder.0.weight"].shape[0])
    if "input_proj.weight" in state_dict:
        return int(state_dict["input_proj.weight"].shape[0])
    return 1024


def _has_obs_norm(state_dict: dict) -> bool:
    return "obs_norm.running_mean" in state_dict


def _infer_num_symbols(obs_size: int) -> int:
    # obs = S * 16 + 5 + S = S * 17 + 5
    return (obs_size - 5) // 17


def _infer_symbols(obs_size: int) -> tuple[str, ...]:
    if obs_size in _OBS_SIZE_TO_SYMBOLS:
        return _OBS_SIZE_TO_SYMBOLS[obs_size]
    n = _infer_num_symbols(obs_size)
    return tuple(f"SYM{i}" for i in range(n))


class RLSignalGenerator:
    def __init__(
        self,
        checkpoint_path: str | Path,
        forecast_cache_root: str | Path = "binanceneural/forecast_cache",
        device: str = "cpu",
        symbols: tuple[str, ...] | None = None,
    ):
        self.device = torch.device(device)
        self.forecast_cache_root = Path(forecast_cache_root)

        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        state_dict = ckpt["model"]

        obs_size = _infer_obs_size(state_dict)
        num_actions = _infer_num_actions(state_dict)
        hidden = _infer_hidden_size(state_dict)
        use_obs_norm = _has_obs_norm(state_dict)

        self.num_symbols = _infer_num_symbols(obs_size)
        self.obs_size = obs_size
        self.num_actions = num_actions

        if symbols is not None:
            self.symbols = symbols
        else:
            self.symbols = _infer_symbols(obs_size)

        if len(self.symbols) != self.num_symbols:
            raise ValueError(
                f"Symbol count mismatch: obs_size={obs_size} implies "
                f"{self.num_symbols} symbols but got {len(self.symbols)}"
            )

        alloc_bins = ckpt.get("action_allocation_bins", 1)
        level_bins = ckpt.get("action_level_bins", 1)
        self.per_symbol_actions = alloc_bins * level_bins
        self.disable_shorts = ckpt.get("disable_shorts", False)

        self.action_names = _build_action_names(self.symbols)

        self.policy = TradingPolicy(obs_size, num_actions, hidden=hidden, use_obs_norm=use_obs_norm)
        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device).eval()

        logger.info(
            f"RL policy loaded: obs_size={obs_size}, num_actions={num_actions}, "
            f"hidden={hidden}, symbols={self.num_symbols}, "
            f"obs_norm={use_obs_norm}, "
            f"update={ckpt.get('update')}, steps={ckpt.get('global_step')}"
        )

    def _fetch_klines(self, binance_pair: str, limit: int = 96) -> pd.DataFrame:
        from src.binan import binance_wrapper as bw

        # todo optimize to only get latest hour and have already got/cached this in trainingdata/ in daily case or trainingdatahourly/ in this case
        klines = bw.get_client().get_klines(symbol=binance_pair, interval="1h", limit=limit)
        rows = []
        for k in klines:
            rows.append(
                {
                    "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC").floor("h"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    def _build_market_features(self, klines_map: dict[str, pd.DataFrame]) -> np.ndarray:
        features = np.zeros((self.num_symbols, FEATURES_PER_SYM), dtype=np.float32)
        for i, sym in enumerate(self.symbols):
            klines = klines_map.get(sym)
            if klines is None or klines.empty:
                continue
            fc_h1 = _load_forecast_parquet(self.forecast_cache_root, 1, sym)
            fc_h24 = _load_forecast_parquet(self.forecast_cache_root, 24, sym)
            feat_arr = compute_symbol_features(klines, fc_h1, fc_h24)
            if len(feat_arr) > 0:
                features[i] = feat_arr[-1]
        return features

    def _build_obs(
        self,
        market_features: np.ndarray,
        portfolio: PortfolioSnapshot,
    ) -> np.ndarray:
        S = self.num_symbols
        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[: S * FEATURES_PER_SYM] = market_features.flatten()

        base = S * FEATURES_PER_SYM
        obs[base + 0] = portfolio.cash_usd / INITIAL_CASH
        pos_val = portfolio.position_value_usd
        if portfolio.is_short:
            pos_val = -pos_val
        obs[base + 1] = pos_val / INITIAL_CASH
        obs[base + 2] = portfolio.unrealized_pnl_usd / INITIAL_CASH
        obs[base + 3] = min(portfolio.hold_hours / 720.0, 1.0)
        obs[base + 4] = 0.25  # episode progress

        if portfolio.position_symbol and portfolio.position_symbol in self.symbols:
            sym_idx = self.symbols.index(portfolio.position_symbol)
            obs[base + 5 + sym_idx] = -1.0 if portfolio.is_short else 1.0

        return obs

    def _decode_action(self, action: int) -> tuple[str | None, str]:
        """Decode flat action index to (target_symbol, direction).

        Action layout: 0=flat, 1..S*psa=long, S*psa+1..2*S*psa=short
        where psa = per_symbol_actions (alloc_bins * level_bins).
        """
        if action == 0:
            return None, "flat"
        S = self.num_symbols
        psa = self.per_symbol_actions
        side_block = S * psa
        if action <= side_block:
            idx = action - 1
            sym_idx = idx // psa
            return self.symbols[sym_idx], "long"
        elif action <= 2 * side_block:
            idx = action - 1 - side_block
            sym_idx = idx // psa
            return self.symbols[sym_idx], "short"
        return None, "flat"

    def _action_to_symbol(self, action: int) -> str | None:
        """Return the symbol for a given action index, or None for FLAT."""
        sym, _ = self._decode_action(action)
        return sym

    def _mask_logits(self, logits: np.ndarray, tradable_symbols: list[str]) -> np.ndarray:
        """Mask logits for non-tradable symbols to -inf. Action 0 (FLAT) is never masked."""
        masked = logits.copy()
        tradable_set = set(tradable_symbols)
        for i in range(1, len(masked)):
            sym = self._action_to_symbol(i)
            if sym is not None and sym not in tradable_set:
                masked[i] = -np.inf
        return masked

    def _mask_shorts(self, logits: np.ndarray) -> np.ndarray:
        """Mask all SHORT actions to -inf for spot-only trading."""
        masked = logits.copy()
        S = self.num_symbols
        psa = self.per_symbol_actions
        short_start = 1 + S * psa
        masked[short_start:] = -np.inf
        return masked

    def get_signal(
        self,
        portfolio: PortfolioSnapshot,
        klines_map: dict[str, pd.DataFrame] | None = None,
        binance_pairs: dict[str, str] | None = None,
        tradable_symbols: list[str] | None = None,
        spot_only: bool = True,
    ) -> RLSignal:
        if klines_map is None:
            if binance_pairs is None:
                binance_pairs = {
                    sym: SYMBOL_TO_BINANCE_PAIR[sym] for sym in self.symbols if sym in SYMBOL_TO_BINANCE_PAIR
                }
            klines_map = {}
            for sym, pair in binance_pairs.items():
                try:
                    klines_map[sym] = self._fetch_klines(pair, limit=96)
                except Exception as e:
                    logger.warning(f"Failed to fetch klines for {sym} ({pair}): {e}")

        market_features = self._build_market_features(klines_map)
        obs = self._build_obs(market_features, portfolio)

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(obs_t)

        logits_np = logits[0].cpu().numpy()
        if tradable_symbols is not None:
            logits_np = self._mask_logits(logits_np, tradable_symbols)
        if spot_only:
            logits_np = self._mask_shorts(logits_np)
        action = int(logits_np.argmax())
        value_f = float(value[0].cpu())

        target_symbol, direction = self._decode_action(action)

        action_name = "FLAT"
        if target_symbol is not None:
            short_name = target_symbol.replace("USD", "")
            prefix = "SHORT" if direction == "short" else "LONG"
            action_name = f"{prefix}_{short_name}"

        sig = RLSignal(
            action=action,
            action_name=action_name,
            target_symbol=target_symbol,
            direction=direction,
            logits=logits_np.tolist(),
            value=value_f,
        )
        logger.info(
            f"RL signal: {sig.action_name} (value={sig.value:.3f}, "
            f"top_logits={sorted(enumerate(sig.logits), key=lambda x: -x[1])[:5]})"
        )
        return sig
