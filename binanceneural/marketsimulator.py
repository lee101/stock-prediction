from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from differentiable_loss_utils import DEFAULT_MAKER_FEE_RATE, HOURLY_PERIODS_PER_YEAR


@dataclass
class SimulationConfig:
    maker_fee: float = DEFAULT_MAKER_FEE_RATE
    initial_cash: float = 10_000.0
    enable_probe_mode: bool = False
    probe_notional: float = 1.0
    max_hold_hours: Optional[int] = None


@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    price: float
    quantity: float
    notional: float
    fee: float
    cash_after: float
    inventory_after: float
    realized_pnl: float
    reason: str = "signal"


@dataclass
class SymbolResult:
    equity_curve: pd.Series
    trades: List[TradeRecord]
    per_hour: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    combined_equity: pd.Series
    per_symbol: Dict[str, SymbolResult]
    metrics: Dict[str, float] = field(default_factory=dict)


class BinanceMarketSimulator:
    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()

    def run(self, bars: pd.DataFrame, actions: pd.DataFrame) -> SimulationResult:
        prepared = _prepare_frame(bars, actions)
        per_symbol: Dict[str, SymbolResult] = {}
        for symbol, frame in prepared.groupby("symbol"):
            symbol_result = _simulate_symbol(frame, symbol, self.config)
            per_symbol[symbol] = symbol_result

        combined_equity = _combine_equity_curves(per_symbol)
        metrics = _compute_metrics(combined_equity)
        return SimulationResult(combined_equity=combined_equity, per_symbol=per_symbol, metrics=metrics)


def run_shared_cash_simulation(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config: Optional[SimulationConfig] = None,
) -> SimulationResult:
    """Simulate trades across symbols with a single shared cash balance."""
    sim_config = config or SimulationConfig()
    prepared = _prepare_frame(bars, actions)

    cash = float(sim_config.initial_cash)
    inventory: Dict[str, float] = {}
    cost_basis: Dict[str, float] = {}
    open_time: Dict[str, Optional[pd.Timestamp]] = {}
    probe_mode: Dict[str, bool] = {}

    amount_scale: Dict[str, float] = {
        symbol: _resolve_amount_scale(frame) for symbol, frame in prepared.groupby("symbol")
    }

    equity_values: List[float] = []
    per_hour_rows: List[Dict[str, float]] = []
    trades_by_symbol: Dict[str, List[TradeRecord]] = {}

    prepared = prepared.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    for ts, chunk in prepared.groupby("timestamp", sort=False):
        # Process sells first to release cash before new buys.
        for row in chunk.itertuples(index=False):
            symbol = str(getattr(row, "symbol")).upper()
            inv = float(inventory.get(symbol, 0.0))
            if inv <= 0:
                continue
            sell_price = float(getattr(row, "sell_price", 0.0) or 0.0)
            sell_amount = getattr(row, "sell_amount", None)
            trade_amount = getattr(row, "trade_amount", None)
            if sell_amount is None:
                sell_amount = trade_amount if trade_amount is not None else 0.0
            scale = amount_scale.get(symbol, 100.0)
            sell_intensity = float(np.clip(float(sell_amount) / scale, 0.0, 1.0))
            high = float(getattr(row, "high"))
            sell_fill = sell_price > 0 and high >= sell_price and sell_intensity > 0
            if not sell_fill:
                continue
            executed_sell = sell_intensity * inv
            if executed_sell <= 0:
                continue
            proceeds = executed_sell * sell_price * (1 - sim_config.maker_fee)
            cash += proceeds
            realized = (sell_price - float(cost_basis.get(symbol, 0.0))) * executed_sell
            inv -= executed_sell
            inventory[symbol] = inv if inv > 0 else 0.0
            if inv <= 0:
                cost_basis[symbol] = 0.0
                open_time[symbol] = None

            if sim_config.enable_probe_mode:
                if realized < 0:
                    probe_mode[symbol] = True
                elif realized > 0 and probe_mode.get(symbol):
                    probe_mode[symbol] = False

            trades_by_symbol.setdefault(symbol, []).append(
                TradeRecord(
                    timestamp=ts,
                    symbol=symbol,
                    side="sell",
                    price=sell_price,
                    quantity=executed_sell,
                    notional=executed_sell * sell_price,
                    fee=executed_sell * sell_price * sim_config.maker_fee,
                    cash_after=cash,
                    inventory_after=inventory[symbol],
                    realized_pnl=realized,
                )
            )

        # Then buys, consuming remaining cash.
        for row in chunk.itertuples(index=False):
            symbol = str(getattr(row, "symbol")).upper()
            buy_price = float(getattr(row, "buy_price", 0.0) or 0.0)
            buy_amount = getattr(row, "buy_amount", None)
            trade_amount = getattr(row, "trade_amount", None)
            if buy_amount is None:
                buy_amount = trade_amount if trade_amount is not None else 0.0
            scale = amount_scale.get(symbol, 100.0)
            buy_intensity = float(np.clip(float(buy_amount) / scale, 0.0, 1.0))
            low = float(getattr(row, "low"))
            buy_fill = buy_price > 0 and low <= buy_price and buy_intensity > 0
            if not buy_fill or cash <= 0:
                continue

            available_cash = cash
            if sim_config.enable_probe_mode and probe_mode.get(symbol) and sim_config.probe_notional > 0:
                available_cash = min(available_cash, float(sim_config.probe_notional))
            max_buy = available_cash / (buy_price * (1 + sim_config.maker_fee)) if buy_price > 0 else 0.0
            executed_buy = buy_intensity * max_buy
            if executed_buy <= 0:
                continue

            cost = executed_buy * buy_price * (1 + sim_config.maker_fee)
            cash -= cost
            inv = float(inventory.get(symbol, 0.0))
            if inv <= 0:
                cost_basis[symbol] = buy_price
                open_time[symbol] = ts
            else:
                cost_basis[symbol] = (cost_basis.get(symbol, buy_price) * inv + buy_price * executed_buy) / (
                    inv + executed_buy
                )
            inventory[symbol] = inv + executed_buy

            trades_by_symbol.setdefault(symbol, []).append(
                TradeRecord(
                    timestamp=ts,
                    symbol=symbol,
                    side="buy",
                    price=buy_price,
                    quantity=executed_buy,
                    notional=executed_buy * buy_price,
                    fee=executed_buy * buy_price * sim_config.maker_fee,
                    cash_after=cash,
                    inventory_after=inventory[symbol],
                    realized_pnl=0.0,
                )
            )

        # Max hold enforcement per symbol (at close).
        if sim_config.max_hold_hours is not None:
            for row in chunk.itertuples(index=False):
                symbol = str(getattr(row, "symbol")).upper()
                inv = float(inventory.get(symbol, 0.0))
                opened = open_time.get(symbol)
                if inv <= 0 or opened is None:
                    continue
                held_hours = (ts - opened).total_seconds() / 3600.0
                if held_hours < sim_config.max_hold_hours:
                    continue
                close = float(getattr(row, "close"))
                proceeds = inv * close * (1 - sim_config.maker_fee)
                cash += proceeds
                realized = (close - float(cost_basis.get(symbol, 0.0))) * inv
                inventory[symbol] = 0.0
                cost_basis[symbol] = 0.0
                open_time[symbol] = None
                trades_by_symbol.setdefault(symbol, []).append(
                    TradeRecord(
                        timestamp=ts,
                        symbol=symbol,
                        side="sell",
                        price=close,
                        quantity=inv,
                        notional=inv * close,
                        fee=inv * close * sim_config.maker_fee,
                        cash_after=cash,
                        inventory_after=0.0,
                        realized_pnl=realized,
                        reason="max_hold",
                    )
                )

        # Equity snapshot at timestamp (shared cash + inventories at close).
        close_values = {
            str(getattr(row, "symbol")).upper(): float(getattr(row, "close")) for row in chunk.itertuples(index=False)
        }
        inventory_value = 0.0
        for symbol, qty in inventory.items():
            close_price = close_values.get(symbol, 0.0)
            inventory_value += qty * close_price
        equity = cash + inventory_value
        equity_values.append(equity)
        per_hour_rows.append(
            {
                "timestamp": ts,
                "portfolio_value": equity,
                "cash": cash,
                "inventory_value": inventory_value,
            }
        )

    equity_curve = pd.Series(equity_values, index=prepared["timestamp"].drop_duplicates().values)
    metrics = _compute_metrics(equity_curve)
    per_symbol: Dict[str, SymbolResult] = {}
    for symbol, trades in trades_by_symbol.items():
        per_symbol[symbol] = SymbolResult(
            equity_curve=pd.Series(dtype=float),
            trades=trades,
            per_hour=pd.DataFrame(),
            metrics={},
        )
    return SimulationResult(combined_equity=equity_curve, per_symbol=per_symbol, metrics=metrics)


def _prepare_frame(bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    actions = actions.copy()
    if "timestamp" not in bars.columns or "timestamp" not in actions.columns:
        raise ValueError("Both bars and actions must include a timestamp column.")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)

    if "symbol" not in bars.columns:
        if "symbol" in actions.columns and actions["symbol"].nunique() == 1:
            bars["symbol"] = actions["symbol"].iloc[0]
        else:
            raise ValueError("Bars missing symbol column; provide symbol in bars or actions.")
    if "symbol" not in actions.columns:
        if "symbol" in bars.columns and bars["symbol"].nunique() == 1:
            actions["symbol"] = bars["symbol"].iloc[0]
        else:
            raise ValueError("Actions missing symbol column; provide symbol in bars or actions.")

    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    actions["symbol"] = actions["symbol"].astype(str).str.upper()

    required_cols = {"high", "low", "close"}
    missing = required_cols - set(bars.columns)
    if missing:
        raise ValueError(f"Bars missing required columns: {sorted(missing)}")

    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner")
    if merged.empty:
        raise ValueError("Merged dataframe is empty; ensure actions cover the provided bars.")
    return merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _resolve_amount_scale(frame: pd.DataFrame) -> float:
    candidates = []
    for col in ("buy_amount", "sell_amount", "trade_amount"):
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.notna().any():
                candidates.append(series.max())
    if not candidates:
        return 100.0
    max_val = float(np.nanmax(candidates))
    return 100.0 if max_val > 1.5 else 1.0


def _simulate_symbol(frame: pd.DataFrame, symbol: str, config: SimulationConfig) -> SymbolResult:
    cash = float(config.initial_cash)
    inventory = 0.0
    cost_basis = 0.0
    open_time: Optional[pd.Timestamp] = None
    probe_mode = False

    amount_scale = _resolve_amount_scale(frame)

    equity_values: List[float] = []
    per_hour_rows: List[Dict[str, float]] = []
    trades: List[TradeRecord] = []

    for row in frame.itertuples(index=False):
        ts = getattr(row, "timestamp")
        buy_price = float(getattr(row, "buy_price", 0.0) or 0.0)
        sell_price = float(getattr(row, "sell_price", 0.0) or 0.0)

        buy_amount = getattr(row, "buy_amount", None)
        sell_amount = getattr(row, "sell_amount", None)
        trade_amount = getattr(row, "trade_amount", None)
        if buy_amount is None:
            buy_amount = trade_amount if trade_amount is not None else 0.0
        if sell_amount is None:
            sell_amount = trade_amount if trade_amount is not None else 0.0

        buy_intensity = float(np.clip(float(buy_amount) / amount_scale, 0.0, 1.0))
        sell_intensity = float(np.clip(float(sell_amount) / amount_scale, 0.0, 1.0))

        low = float(getattr(row, "low"))
        high = float(getattr(row, "high"))
        close = float(getattr(row, "close"))

        buy_fill = buy_price > 0 and low <= buy_price and buy_intensity > 0
        sell_fill = sell_price > 0 and high >= sell_price and sell_intensity > 0

        executed_buy = 0.0
        executed_sell = 0.0

        if buy_fill:
            available_cash = cash
            if config.enable_probe_mode and probe_mode and config.probe_notional > 0:
                available_cash = min(available_cash, float(config.probe_notional))
            max_buy = available_cash / (buy_price * (1 + config.maker_fee)) if buy_price > 0 else 0.0
            executed_buy = buy_intensity * max_buy

        if executed_buy > 0:
            cost = executed_buy * buy_price * (1 + config.maker_fee)
            cash -= cost
            if inventory <= 0:
                cost_basis = buy_price
                open_time = ts
            else:
                cost_basis = (cost_basis * inventory + buy_price * executed_buy) / (inventory + executed_buy)
            inventory += executed_buy
            trades.append(
                TradeRecord(
                    timestamp=ts,
                    symbol=symbol,
                    side="buy",
                    price=buy_price,
                    quantity=executed_buy,
                    notional=executed_buy * buy_price,
                    fee=executed_buy * buy_price * config.maker_fee,
                    cash_after=cash,
                    inventory_after=inventory,
                    realized_pnl=0.0,
                )
            )

        if sell_fill and inventory > 0:
            executed_sell = sell_intensity * inventory

        if executed_sell > 0:
            proceeds = executed_sell * sell_price * (1 - config.maker_fee)
            cash += proceeds
            realized = (sell_price - cost_basis) * executed_sell
            inventory -= executed_sell
            if inventory <= 0:
                inventory = 0.0
                cost_basis = 0.0
                open_time = None

            if config.enable_probe_mode:
                if realized < 0:
                    probe_mode = True
                elif realized > 0 and probe_mode:
                    probe_mode = False

            trades.append(
                TradeRecord(
                    timestamp=ts,
                    symbol=symbol,
                    side="sell",
                    price=sell_price,
                    quantity=executed_sell,
                    notional=executed_sell * sell_price,
                    fee=executed_sell * sell_price * config.maker_fee,
                    cash_after=cash,
                    inventory_after=inventory,
                    realized_pnl=realized,
                )
            )

        if config.max_hold_hours is not None and inventory > 0 and open_time is not None:
            held_hours = (ts - open_time).total_seconds() / 3600.0
            if held_hours >= config.max_hold_hours:
                forced_qty = inventory
                proceeds = forced_qty * close * (1 - config.maker_fee)
                cash += proceeds
                realized = (close - cost_basis) * forced_qty
                inventory = 0.0
                cost_basis = 0.0
                open_time = None

                if config.enable_probe_mode:
                    if realized < 0:
                        probe_mode = True
                    elif realized > 0 and probe_mode:
                        probe_mode = False

                trades.append(
                    TradeRecord(
                        timestamp=ts,
                        symbol=symbol,
                        side="sell",
                        price=close,
                        quantity=forced_qty,
                        notional=forced_qty * close,
                        fee=forced_qty * close * config.maker_fee,
                        cash_after=cash,
                        inventory_after=inventory,
                        realized_pnl=realized,
                        reason="max_hold",
                    )
                )

        portfolio_value = cash + inventory * close
        equity_values.append(portfolio_value)
        per_hour_rows.append(
            {
                "timestamp": ts,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "inventory": inventory,
                "buy_filled": float(executed_buy > 0),
                "sell_filled": float(executed_sell > 0),
            }
        )

    equity_curve = pd.Series(equity_values, index=frame["timestamp"].values)
    per_hour = pd.DataFrame(per_hour_rows)
    metrics = _compute_metrics(equity_curve)
    return SymbolResult(equity_curve=equity_curve, trades=trades, per_hour=per_hour, metrics=metrics)


def _combine_equity_curves(per_symbol: Dict[str, SymbolResult]) -> pd.Series:
    if not per_symbol:
        return pd.Series(dtype=float)
    equity = {symbol: result.equity_curve for symbol, result in per_symbol.items()}
    combined = pd.DataFrame(equity).sort_index()
    combined = combined.ffill().dropna(how="all")
    if combined.empty:
        return pd.Series(dtype=float)
    combined = combined.ffill()
    return combined.sum(axis=1)


def _compute_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    if equity_curve.empty:
        return {"total_return": 0.0, "sortino": 0.0, "mean_hourly_return": 0.0}
    values = equity_curve.to_numpy(dtype=float)
    if len(values) < 2:
        return {"total_return": 0.0, "sortino": 0.0, "mean_hourly_return": 0.0}
    returns = np.diff(values) / np.clip(values[:-1], a_min=1e-8, a_max=None)
    mean_ret = returns.mean() if len(returns) else 0.0
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) else 0.0
    sortino = mean_ret / downside_std * np.sqrt(HOURLY_PERIODS_PER_YEAR) if downside_std > 0 else 0.0
    total_return = (values[-1] - values[0]) / values[0]
    return {
        "total_return": float(total_return),
        "sortino": float(sortino),
        "mean_hourly_return": float(mean_ret),
    }


def save_trade_plot(
    symbol: str,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    result: SymbolResult,
    output_path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    symbol = symbol.upper()
    bars = bars.copy()
    if "symbol" in bars.columns:
        bars = bars[bars["symbol"].astype(str).str.upper() == symbol]
    bars = bars.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bars["timestamp"], bars["close"], label="close", linewidth=1.25)

    buys = [trade for trade in result.trades if trade.side == "buy"]
    sells = [trade for trade in result.trades if trade.side == "sell"]

    if buys:
        ax.scatter(
            [trade.timestamp for trade in buys],
            [trade.price for trade in buys],
            marker="^",
            color="green",
            label="buy",
        )
    if sells:
        ax.scatter(
            [trade.timestamp for trade in sells],
            [trade.price for trade in sells],
            marker="v",
            color="red",
            label="sell",
        )

    ax.set_title(f"{symbol} simulated trades")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("price")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


__all__ = [
    "BinanceMarketSimulator",
    "SimulationConfig",
    "SimulationResult",
    "SymbolResult",
    "TradeRecord",
    "save_trade_plot",
    "run_shared_cash_simulation",
]
