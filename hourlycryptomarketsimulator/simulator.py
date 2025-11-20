from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    maker_fee: float = 0.0008
    initial_cash: float = 50_000.0
    symbol: str = "LINKUSD"


@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    side: str
    price: float
    quantity: float
    cash_after: float
    inventory_after: float


@dataclass
class SimulationResult:
    equity_curve: pd.Series
    trades: List[TradeRecord]
    per_hour: pd.DataFrame
    final_cash: float
    final_inventory: float
    metrics: Dict[str, float] = field(default_factory=dict)


class HourlyCryptoMarketSimulator:
    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()

    def run(self, bars: pd.DataFrame, actions: pd.DataFrame) -> SimulationResult:
        frame = self._prepare_frame(bars, actions)
        cash = self.config.initial_cash
        inventory = 0.0
        equity_values: List[float] = []
        per_hour_rows: List[Dict[str, float]] = []
        trades: List[TradeRecord] = []
        for row in frame.itertuples(index=False):
            buy_intensity = float(np.clip(getattr(row, "buy_amount", getattr(row, "trade_amount", 0.0)), 0.0, 1.0))
            sell_intensity = float(np.clip(getattr(row, "sell_amount", getattr(row, "trade_amount", 0.0)), 0.0, 1.0))
            buy_fill = bool(row.low <= row.buy_price and buy_intensity > 0)
            sell_fill = bool(row.high >= row.sell_price and sell_intensity > 0)
            executed_buy = 0.0
            executed_sell = 0.0
            if buy_fill:
                max_buy = cash / (row.buy_price * (1 + self.config.maker_fee)) if row.buy_price > 0 else 0.0
                executed_buy = buy_intensity * max_buy
            if sell_fill:
                executed_sell = sell_intensity * max(0.0, inventory)
            if executed_buy > 0:
                cash -= executed_buy * row.buy_price * (1 + self.config.maker_fee)
                inventory += executed_buy
                trades.append(
                    TradeRecord(
                        timestamp=row.timestamp,
                        side="buy",
                        price=float(row.buy_price),
                        quantity=executed_buy,
                        cash_after=cash,
                        inventory_after=inventory,
                    )
                )
            if executed_sell > 0:
                cash += executed_sell * row.sell_price * (1 - self.config.maker_fee)
                inventory -= executed_sell
                trades.append(
                    TradeRecord(
                        timestamp=row.timestamp,
                        side="sell",
                        price=float(row.sell_price),
                        quantity=executed_sell,
                        cash_after=cash,
                        inventory_after=inventory,
                    )
                )
            portfolio_value = cash + inventory * row.close
            equity_values.append(portfolio_value)
            per_hour_rows.append(
                {
                    "timestamp": row.timestamp,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "inventory": inventory,
                    "buy_filled": float(executed_buy > 0),
                    "sell_filled": float(executed_sell > 0),
                }
            )
        equity_curve = pd.Series(equity_values, index=frame["timestamp"].values)
        per_hour = pd.DataFrame(per_hour_rows)
        metrics = self._compute_metrics(equity_curve)
        return SimulationResult(
            equity_curve=equity_curve,
            trades=trades,
            per_hour=per_hour,
            final_cash=cash,
            final_inventory=inventory,
            metrics=metrics,
        )

    def _prepare_frame(self, bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
        bars = bars.copy()
        actions = actions.copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)
        merged = bars.merge(actions, on="timestamp", how="inner")
        if merged.empty:
            raise ValueError("Merged dataframe is empty; ensure actions cover the provided bars.")
        return merged.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _compute_metrics(equity_curve: pd.Series) -> Dict[str, float]:
        if equity_curve.empty:
            return {"total_return": 0.0, "sortino": 0.0}
        values = equity_curve.values
        returns = np.diff(values) / np.clip(values[:-1], a_min=1e-8, a_max=None)
        mean_ret = returns.mean() if len(returns) else 0.0
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) else 0.0
        sortino = mean_ret / downside_std * np.sqrt(24 * 365) if downside_std > 0 else 0.0
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        return {
            "total_return": float(total_return),
            "sortino": float(sortino),
            "mean_hourly_return": float(mean_ret),
        }
