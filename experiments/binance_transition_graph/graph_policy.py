"""Sequential graph-policy optimizer for Binance hourly crypto rotation.

This module is intentionally independent from live trading. It evaluates a
stateful policy one hour at a time with decision lag, fees, slippage, leverage,
and borrow cost. The "graph" is represented as named transition edges between
portfolio states: cash, long(symbol), and short(symbol).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

HOURS_PER_MONTH = 30.0 * 24.0
HOURS_PER_YEAR = 365.0 * 24.0


DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "AAVEUSDT",
    "LINKUSDT",
)


@dataclass(frozen=True)
class TargetState:
    symbol: str | None = None
    side: int = 0
    leverage: float = 0.0
    edge: str = "hold"

    @property
    def label(self) -> str:
        if self.symbol is None or self.side == 0 or self.leverage <= 0:
            return "CASH"
        prefix = "LONG" if self.side > 0 else "SHORT"
        return f"{prefix}:{self.symbol}"


@dataclass(frozen=True)
class GraphPolicyConfig:
    name: str = "graph_default"
    fast_lookback_hours: int = 6
    slow_lookback_hours: int = 24
    regime_lookback_hours: int = 72
    rebalance_hours: int = 6
    target_leverage: float = 1.0
    max_leverage: float = 2.0
    allow_short: bool = False
    momentum_threshold: float = 0.015
    dip_buy_threshold: float = -0.04
    winner_sell_threshold: float = 0.035
    rotation_spread_threshold: float = 0.07
    short_threshold: float = -0.06
    market_regime_cash_threshold: float = -0.08
    market_regime_short_threshold: float = -0.04
    volatility_cash_threshold: float = 0.09
    drawdown_cash_threshold: float = -0.18
    stop_loss: float = -0.08
    take_profit: float = 0.18
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    borrow_apr: float = 0.0675
    decision_lag_hours: int = 2

    def clipped(self) -> "GraphPolicyConfig":
        return replace(
            self,
            fast_lookback_hours=max(1, int(self.fast_lookback_hours)),
            slow_lookback_hours=max(2, int(self.slow_lookback_hours)),
            regime_lookback_hours=max(4, int(self.regime_lookback_hours)),
            rebalance_hours=max(1, int(self.rebalance_hours)),
            target_leverage=float(min(max(self.target_leverage, 0.0), self.max_leverage)),
            max_leverage=float(max(0.0, self.max_leverage)),
            decision_lag_hours=max(0, int(self.decision_lag_hours)),
        )


@dataclass(frozen=True)
class SimulationMetrics:
    total_return: float
    monthly_return: float
    sortino: float
    max_drawdown: float
    ulcer_index: float
    pnl_smoothness: float
    num_trades: int
    win_rate: float
    hours: int
    final_equity: float


@dataclass(frozen=True)
class SimulationResult:
    config: GraphPolicyConfig
    metrics: SimulationMetrics
    equity_curve: pd.Series
    trades: pd.DataFrame
    state_counts: dict[str, int]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "metrics": asdict(self.metrics),
            "state_counts": self.state_counts,
            "trades": self.trades.to_dict(orient="records"),
        }


@dataclass(frozen=True)
class SlippageSweepResult:
    config: GraphPolicyConfig
    by_slippage: dict[float, SimulationMetrics]

    @property
    def worst_monthly_return(self) -> float:
        return min(metric.monthly_return for metric in self.by_slippage.values())

    @property
    def worst_max_drawdown(self) -> float:
        return max(metric.max_drawdown for metric in self.by_slippage.values())

    def to_json_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "worst_monthly_return": self.worst_monthly_return,
            "worst_max_drawdown": self.worst_max_drawdown,
            "by_slippage": {
                str(k): asdict(v) for k, v in sorted(self.by_slippage.items())
            },
        }


def load_hourly_panel(
    data_root: str | Path,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    *,
    tail_hours: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load aligned hourly OHLCV frames from CSV files."""
    root = Path(data_root)
    frames: dict[str, pd.DataFrame] = {}
    for raw_symbol in symbols:
        symbol = raw_symbol.upper()
        path = root / f"{symbol}.csv"
        if not path.exists() and symbol.endswith("USD"):
            path = root / f"{symbol}T.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing hourly CSV for {symbol}: {path}")

        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"{path} is missing timestamp column")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        df = df.set_index("timestamp")
        df = df[required].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[~df.index.duplicated(keep="last")]
        if tail_hours is not None and tail_hours > 0:
            df = df.tail(int(tail_hours))
        frames[symbol] = df

    common_index: pd.DatetimeIndex | None = None
    for df in frames.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    if common_index is None or common_index.empty:
        raise ValueError("No common hourly timestamps across requested symbols")
    common_index = common_index.sort_values()
    return {symbol: frame.loc[common_index].copy() for symbol, frame in frames.items()}


def close_matrix(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({symbol: frame["close"] for symbol, frame in frames.items()}).sort_index()


def _safe_pct_change(close: pd.DataFrame, periods: int) -> pd.DataFrame:
    return close.pct_change(int(periods)).replace([np.inf, -np.inf], np.nan)


def _rolling_vol(close: pd.DataFrame, periods: int) -> pd.DataFrame:
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    return returns.rolling(int(periods), min_periods=max(2, int(periods) // 3)).std()


def _rolling_drawdown(close: pd.DataFrame, periods: int) -> pd.DataFrame:
    high = close.rolling(int(periods), min_periods=max(2, int(periods) // 3)).max()
    return close / high - 1.0


def _build_feature_cache(close: pd.DataFrame, cfg: GraphPolicyConfig) -> dict[str, pd.DataFrame]:
    return {
        "slow": _safe_pct_change(close, cfg.slow_lookback_hours),
        "fast": _safe_pct_change(close, cfg.fast_lookback_hours),
        "regime": _safe_pct_change(close, cfg.regime_lookback_hours),
        "vol": _rolling_vol(close, cfg.slow_lookback_hours),
        "drawdown": _rolling_drawdown(close, cfg.regime_lookback_hours),
    }


def _score_sort(series: pd.Series, *, ascending: bool) -> tuple[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return "", 0.0
    ordered = clean.sort_values(ascending=ascending)
    symbol = str(ordered.index[0])
    return symbol, float(ordered.iloc[0])


def _target_for_symbol(
    symbol: str | None,
    *,
    side: int,
    leverage: float,
    edge: str,
) -> TargetState:
    if not symbol or side == 0 or leverage <= 0:
        return TargetState(edge=edge)
    return TargetState(symbol=symbol, side=side, leverage=float(leverage), edge=edge)


def decide_target(
    cfg: GraphPolicyConfig,
    *,
    close: pd.DataFrame,
    t: int,
    current: TargetState,
    entry_price: float | None,
    last_trade_t: int,
    feature_cache: Mapping[str, pd.DataFrame] | None = None,
) -> TargetState:
    """Choose the next graph node using only lagged data at row t."""
    cfg = cfg.clipped()
    if feature_cache is None:
        slow = _safe_pct_change(close.iloc[: t + 1], cfg.slow_lookback_hours).iloc[-1]
        fast = _safe_pct_change(close.iloc[: t + 1], cfg.fast_lookback_hours).iloc[-1]
        regime = _safe_pct_change(close.iloc[: t + 1], cfg.regime_lookback_hours).iloc[-1]
        vol = _rolling_vol(close.iloc[: t + 1], cfg.slow_lookback_hours).iloc[-1]
        drawdown = _rolling_drawdown(close.iloc[: t + 1], cfg.regime_lookback_hours).iloc[-1]
    else:
        slow = feature_cache["slow"].iloc[t]
        fast = feature_cache["fast"].iloc[t]
        regime = feature_cache["regime"].iloc[t]
        vol = feature_cache["vol"].iloc[t]
        drawdown = feature_cache["drawdown"].iloc[t]
    market_regime = float(regime.replace([np.inf, -np.inf], np.nan).median(skipna=True))
    market_vol = float(vol.replace([np.inf, -np.inf], np.nan).median(skipna=True))
    worst_drawdown = float(drawdown.replace([np.inf, -np.inf], np.nan).min(skipna=True))

    if not math.isfinite(market_regime):
        market_regime = 0.0
    if not math.isfinite(market_vol):
        market_vol = 0.0
    if not math.isfinite(worst_drawdown):
        worst_drawdown = 0.0

    leverage = min(float(cfg.target_leverage), float(cfg.max_leverage))
    current_price = None
    if current.symbol is not None and current.symbol in close.columns:
        current_price = float(close[current.symbol].iloc[t])

    if current.symbol and entry_price and entry_price > 0 and current_price and current_price > 0:
        pnl = current.side * (current_price / float(entry_price) - 1.0)
        if pnl <= cfg.stop_loss:
            return TargetState(edge="risk_stop_loss")
        if pnl >= cfg.take_profit:
            return TargetState(edge="risk_take_profit")

    if market_regime <= cfg.market_regime_cash_threshold:
        if not cfg.allow_short:
            return TargetState(edge="risk_market_regime_cash")
    if market_vol >= cfg.volatility_cash_threshold:
        return TargetState(edge="risk_high_vol_cash")
    if worst_drawdown <= cfg.drawdown_cash_threshold and market_regime < 0:
        return TargetState(edge="risk_drawdown_cash")

    hours_since_trade = t - int(last_trade_t)
    if current.symbol is not None and hours_since_trade < cfg.rebalance_hours:
        return TargetState(
            symbol=current.symbol,
            side=current.side,
            leverage=current.leverage,
            edge="hold_min_rebalance",
        )

    top_symbol, top_ret = _score_sort(slow, ascending=False)
    bottom_symbol, bottom_ret = _score_sort(slow, ascending=True)
    fast_top_symbol, fast_top_ret = _score_sort(fast, ascending=False)
    spread = top_ret - bottom_ret

    if (
        current.symbol == top_symbol
        and top_ret >= cfg.winner_sell_threshold
        and bottom_ret <= cfg.dip_buy_threshold
        and spread >= cfg.rotation_spread_threshold
    ):
        return _target_for_symbol(bottom_symbol, side=1, leverage=leverage, edge="rotate_winner_to_dip")

    if (
        bottom_ret <= cfg.dip_buy_threshold
        and spread >= cfg.rotation_spread_threshold
        and market_regime > cfg.market_regime_short_threshold
    ):
        return _target_for_symbol(bottom_symbol, side=1, leverage=leverage, edge="mean_revert_dip")

    if fast_top_ret >= cfg.momentum_threshold and top_ret >= 0.0:
        return _target_for_symbol(fast_top_symbol, side=1, leverage=leverage, edge="momentum_top")

    if cfg.allow_short and market_regime <= cfg.market_regime_short_threshold and bottom_ret <= cfg.short_threshold:
        return _target_for_symbol(bottom_symbol, side=-1, leverage=leverage, edge="bearish_short_bottom")

    if current.symbol is not None:
        return TargetState(
            symbol=current.symbol,
            side=current.side,
            leverage=current.leverage,
            edge="hold_no_better_edge",
        )
    return TargetState(edge="cash_no_edge")


def _turnover(old: TargetState, new: TargetState) -> float:
    if old.symbol == new.symbol and old.side == new.side:
        return abs(float(new.leverage) - float(old.leverage))
    return float(old.leverage) + float(new.leverage)


def _compute_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> SimulationMetrics:
    if equity_curve.empty:
        raise ValueError("equity_curve is empty")
    equity = equity_curve.astype(float)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    hours = max(1, int(len(equity) - 1))
    monthly_return = float((max(equity.iloc[-1] / equity.iloc[0], 1e-12) ** (HOURS_PER_MONTH / hours)) - 1.0)
    hourly = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    downside = hourly[hourly < 0.0]
    if downside.empty:
        sortino = float("inf") if hourly.mean() > 0 else 0.0
    else:
        sortino = float(hourly.mean() / max(downside.std(ddof=0), 1e-12) * math.sqrt(HOURS_PER_YEAR))
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_drawdown = float(abs(dd.min()))
    ulcer_index = float(np.sqrt(np.mean(np.square(np.minimum(dd.to_numpy(dtype=float), 0.0)))))
    x = np.arange(len(equity), dtype=float)
    if len(equity) > 2 and np.std(equity.to_numpy(dtype=float)) > 0:
        corr = float(np.corrcoef(x, equity.to_numpy(dtype=float))[0, 1])
        pnl_smoothness = corr * corr
    else:
        pnl_smoothness = 0.0
    if trades.empty or "pnl_after_cost" not in trades:
        win_rate = 0.0
    else:
        pnl = trades["pnl_after_cost"].astype(float)
        win_rate = float((pnl > 0.0).mean()) if len(pnl) else 0.0
    return SimulationMetrics(
        total_return=total_return,
        monthly_return=monthly_return,
        sortino=sortino,
        max_drawdown=max_drawdown,
        ulcer_index=ulcer_index,
        pnl_smoothness=pnl_smoothness,
        num_trades=int(len(trades)),
        win_rate=win_rate,
        hours=hours,
        final_equity=float(equity.iloc[-1]),
    )


def simulate_graph_policy(
    frames: Mapping[str, pd.DataFrame],
    config: GraphPolicyConfig,
    *,
    initial_cash: float = 10_000.0,
    warmup_rows: int = 0,
) -> SimulationResult:
    """Run the graph policy sequentially over aligned hourly frames."""
    cfg = config.clipped()
    close = close_matrix(frames)
    feature_cache = _build_feature_cache(close, cfg)
    max_lookback = max(cfg.fast_lookback_hours, cfg.slow_lookback_hours, cfg.regime_lookback_hours)
    start_i = max_lookback + cfg.decision_lag_hours + 1
    if len(close) <= start_i + 2:
        raise ValueError(f"Not enough data for {cfg}: got {len(close)} rows")

    equity = float(initial_cash)
    current = TargetState(edge="start_cash")
    pending = current
    entry_price: float | None = None
    last_trade_t = start_i
    last_mark_price: float | None = None
    fee_rate = (float(cfg.fee_bps) + float(cfg.slippage_bps)) / 10_000.0
    borrow_hourly = max(0.0, float(cfg.borrow_apr)) / HOURS_PER_YEAR

    equity_values: list[float] = []
    equity_index: list[pd.Timestamp] = []
    trades: list[dict[str, object]] = []
    state_counts: dict[str, int] = {}

    metric_start_i = max(start_i, int(warmup_rows))
    for i in range(start_i, len(close)):
        ts = close.index[i]
        if current.symbol is not None and last_mark_price and last_mark_price > 0:
            price = float(close[current.symbol].iloc[i])
            bar_return = price / last_mark_price - 1.0
            equity *= 1.0 + current.side * current.leverage * bar_return
            borrow_exposure = max(current.leverage - 1.0, 0.0)
            if current.side < 0:
                borrow_exposure += current.leverage
            if borrow_exposure > 0.0:
                equity -= equity * borrow_exposure * borrow_hourly
            last_mark_price = price
        elif current.symbol is not None:
            last_mark_price = float(close[current.symbol].iloc[i])

        decision_i = i - cfg.decision_lag_hours
        desired = decide_target(
            cfg,
            close=close,
            t=decision_i,
            current=current,
            entry_price=entry_price,
            last_trade_t=last_trade_t,
            feature_cache=feature_cache,
        )
        pending = desired
        if pending != current:
            before_cost = equity
            cost = before_cost * _turnover(current, pending) * fee_rate
            exit_equity = equity - cost
            old_label = current.label
            new_label = pending.label
            pnl_after_cost = exit_equity - before_cost
            if equity <= 0.0 or exit_equity <= 0.0:
                equity = max(exit_equity, 0.0)
            else:
                equity = exit_equity
            trades.append(
                {
                    "timestamp": ts.isoformat(),
                    "from_state": old_label,
                    "to_state": new_label,
                    "edge": pending.edge,
                    "turnover": _turnover(current, pending),
                    "cost": cost,
                    "equity_after": equity,
                    "pnl_after_cost": pnl_after_cost,
                }
            )
            current = pending
            last_trade_t = i
            if current.symbol is None:
                entry_price = None
                last_mark_price = None
            else:
                entry_price = float(close[current.symbol].iloc[i])
                last_mark_price = entry_price

        equity = max(0.0, float(equity))
        if i >= metric_start_i:
            equity_values.append(equity)
            equity_index.append(ts)
            state_counts[current.label] = state_counts.get(current.label, 0) + 1
        if equity <= 1e-9:
            break

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_index), name="equity")
    trades_df = pd.DataFrame(trades)
    metrics = _compute_metrics(equity_curve, trades_df)
    return SimulationResult(
        config=cfg,
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades_df,
        state_counts=state_counts,
    )


def train_eval_split(
    frames: Mapping[str, pd.DataFrame],
    *,
    train_hours: int,
    eval_hours: int,
    eval_warmup_hours: int = 0,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    total = int(train_hours) + int(eval_hours)
    if total <= 0:
        raise ValueError("train_hours + eval_hours must be positive")
    close = close_matrix(frames)
    if len(close) < total:
        raise ValueError(f"Need {total} common rows but only have {len(close)}")
    train_index = close.index[-total:-int(eval_hours)]
    warmup = max(0, min(int(eval_warmup_hours), int(train_hours)))
    eval_index = close.index[-int(eval_hours) - warmup:]
    train = {symbol: frame.loc[train_index].copy() for symbol, frame in frames.items()}
    eval_ = {symbol: frame.loc[eval_index].copy() for symbol, frame in frames.items()}
    return train, eval_


def sample_configs(
    *,
    n: int,
    seed: int,
    max_leverage: float,
    allow_short: bool,
) -> list[GraphPolicyConfig]:
    rng = np.random.default_rng(int(seed))
    configs: list[GraphPolicyConfig] = []
    fast_choices = np.array([3, 6, 12, 24])
    slow_choices = np.array([12, 24, 48, 72])
    regime_choices = np.array([48, 72, 120, 168])
    rebalance_choices = np.array([1, 3, 6, 12, 24])
    for i in range(int(n)):
        lev = float(rng.uniform(0.5, max(0.5, float(max_leverage))))
        configs.append(
            GraphPolicyConfig(
                name=f"rand_graph_{seed}_{i:04d}",
                fast_lookback_hours=int(rng.choice(fast_choices)),
                slow_lookback_hours=int(rng.choice(slow_choices)),
                regime_lookback_hours=int(rng.choice(regime_choices)),
                rebalance_hours=int(rng.choice(rebalance_choices)),
                target_leverage=lev,
                max_leverage=float(max_leverage),
                allow_short=bool(allow_short and rng.random() < 0.65),
                momentum_threshold=float(rng.uniform(0.003, 0.06)),
                dip_buy_threshold=float(-rng.uniform(0.015, 0.12)),
                winner_sell_threshold=float(rng.uniform(0.015, 0.10)),
                rotation_spread_threshold=float(rng.uniform(0.025, 0.18)),
                short_threshold=float(-rng.uniform(0.025, 0.14)),
                market_regime_cash_threshold=float(-rng.uniform(0.03, 0.18)),
                market_regime_short_threshold=float(-rng.uniform(0.015, 0.12)),
                volatility_cash_threshold=float(rng.uniform(0.025, 0.16)),
                drawdown_cash_threshold=float(-rng.uniform(0.08, 0.35)),
                stop_loss=float(-rng.uniform(0.03, 0.16)),
                take_profit=float(rng.uniform(0.04, 0.35)),
                decision_lag_hours=2,
            ).clipped()
        )
    return configs


def baseline_configs(*, max_leverage: float = 1.0, allow_short: bool = False) -> list[GraphPolicyConfig]:
    return [
        GraphPolicyConfig(name="graph_cash", target_leverage=0.0, max_leverage=max_leverage),
        GraphPolicyConfig(
            name="graph_momentum_conservative",
            fast_lookback_hours=12,
            slow_lookback_hours=48,
            regime_lookback_hours=168,
            rebalance_hours=12,
            target_leverage=min(1.0, max_leverage),
            max_leverage=max_leverage,
            allow_short=False,
            momentum_threshold=0.02,
            dip_buy_threshold=-0.08,
            rotation_spread_threshold=0.12,
            volatility_cash_threshold=0.07,
            stop_loss=-0.06,
            take_profit=0.14,
        ),
        GraphPolicyConfig(
            name="graph_reversal_rotation",
            fast_lookback_hours=6,
            slow_lookback_hours=24,
            regime_lookback_hours=72,
            rebalance_hours=6,
            target_leverage=min(1.25, max_leverage),
            max_leverage=max_leverage,
            allow_short=False,
            momentum_threshold=0.08,
            dip_buy_threshold=-0.035,
            winner_sell_threshold=0.025,
            rotation_spread_threshold=0.055,
            volatility_cash_threshold=0.11,
            stop_loss=-0.08,
            take_profit=0.18,
        ),
        GraphPolicyConfig(
            name="graph_regime_short",
            fast_lookback_hours=6,
            slow_lookback_hours=24,
            regime_lookback_hours=120,
            rebalance_hours=6,
            target_leverage=min(1.5, max_leverage),
            max_leverage=max_leverage,
            allow_short=allow_short,
            momentum_threshold=0.02,
            dip_buy_threshold=-0.05,
            short_threshold=-0.045,
            market_regime_short_threshold=-0.03,
            market_regime_cash_threshold=-0.12,
            stop_loss=-0.07,
            take_profit=0.16,
        ),
    ]


def optimize_on_train(
    frames: Mapping[str, pd.DataFrame],
    configs: Iterable[GraphPolicyConfig],
    *,
    top_k: int = 8,
) -> list[SimulationResult]:
    results: list[SimulationResult] = []
    for cfg in configs:
        try:
            result = simulate_graph_policy(frames, cfg)
        except Exception:
            continue
        if result.metrics.max_drawdown > 0.95:
            continue
        results.append(result)
    results.sort(
        key=lambda r: (
            r.metrics.monthly_return - 0.5 * max(0.0, r.metrics.max_drawdown - 0.20),
            r.metrics.sortino,
            -r.metrics.ulcer_index,
        ),
        reverse=True,
    )
    return results[: int(top_k)]


def evaluate_slippage_sweep(
    frames: Mapping[str, pd.DataFrame],
    config: GraphPolicyConfig,
    *,
    slippage_bps: Sequence[float] = (0.0, 5.0, 10.0, 20.0),
    warmup_rows: int = 0,
) -> SlippageSweepResult:
    by_slippage: dict[float, SimulationMetrics] = {}
    for bps in slippage_bps:
        cfg = replace(config, slippage_bps=float(bps)).clipped()
        result = simulate_graph_policy(frames, cfg, warmup_rows=int(warmup_rows))
        by_slippage[float(bps)] = result.metrics
    return SlippageSweepResult(config=config, by_slippage=by_slippage)


def config_from_mapping(payload: Mapping[str, object], *, name: str | None = None) -> GraphPolicyConfig:
    fields = set(GraphPolicyConfig.__dataclass_fields__.keys())
    cleaned: dict[str, object] = {key: value for key, value in payload.items() if key in fields}
    if name is not None:
        cleaned["name"] = name
    return GraphPolicyConfig(**cleaned).clipped()


def write_result_json(path: str | Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summarize_sweep_rows(rows: Sequence[SlippageSweepResult]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        base = asdict(row.config)
        records.append(
            {
                "name": row.config.name,
                "worst_monthly_pct": 100.0 * row.worst_monthly_return,
                "worst_max_dd_pct": 100.0 * row.worst_max_drawdown,
                **{
                    f"slip{int(bps)}_monthly_pct": 100.0 * metric.monthly_return
                    for bps, metric in row.by_slippage.items()
                },
                **{
                    f"slip{int(bps)}_dd_pct": 100.0 * metric.max_drawdown
                    for bps, metric in row.by_slippage.items()
                },
                "trades_20bps": row.by_slippage[max(row.by_slippage)].num_trades,
                "sortino_20bps": row.by_slippage[max(row.by_slippage)].sortino,
                "config_json": json.dumps(base, sort_keys=True),
            }
        )
    return pd.DataFrame.from_records(records)
