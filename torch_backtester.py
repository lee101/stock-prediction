"""Vectorised daily backtesting with PyTorch autograd support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import torch
from loguru import logger


def _latest_csv(data_dir: Path, symbol: str) -> Path:
    candidates = sorted(data_dir.glob(f"{symbol}-*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No daily bar csv found for {symbol} in {data_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_daily_panel(
    symbols: Iterable[str],
    data_dir: Path = Path("backtestdata"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load open/close panels indexed by timestamp for the requested symbols."""

    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        csv_path = _latest_csv(data_dir, symbol)
        df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        df = df[["Open", "Close"]]
        df.columns = pd.MultiIndex.from_product([[symbol], df.columns], names=["symbol", "field"])
        frames.append(df)

    merged = pd.concat(frames, axis=1).dropna()
    opens = merged.xs("Open", axis=1, level="field")
    closes = merged.xs("Close", axis=1, level="field")
    return opens, closes


def prepare_tensors(
    symbols: Iterable[str],
    simulation_days: int,
    lookback: int = 5,
    device: torch.device | None = None,
    data_dir: Path = Path("backtestdata"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[pd.Timestamp]]:
    """Load price data and produce torch tensors suitable for simulation."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opens_df, closes_df = load_daily_panel(symbols, data_dir=data_dir)

    momentum = closes_df.pct_change(periods=lookback)
    forecasts_df = momentum.shift(1).dropna()

    aligned_opens = opens_df.loc[forecasts_df.index]
    aligned_closes = closes_df.loc[forecasts_df.index]

    if simulation_days:
        aligned_opens = aligned_opens.tail(simulation_days)
        aligned_closes = aligned_closes.tail(simulation_days)
        forecasts_df = forecasts_df.tail(simulation_days)

    opens_tensor = torch.tensor(aligned_opens.values, dtype=torch.float32, device=device)
    closes_tensor = torch.tensor(aligned_closes.values, dtype=torch.float32, device=device)
    forecasts_tensor = torch.tensor(forecasts_df.values, dtype=torch.float32, device=device)
    dates = list(aligned_opens.index)

    return opens_tensor, closes_tensor, forecasts_tensor, dates


@dataclass
class SimulationResult:
    equity_curve: torch.Tensor
    daily_returns: torch.Tensor
    asset_weights: torch.Tensor
    cash_weights: torch.Tensor

    def detach(self) -> "SimulationResult":
        return SimulationResult(
            equity_curve=self.equity_curve.detach().cpu(),
            daily_returns=self.daily_returns.detach().cpu(),
            asset_weights=self.asset_weights.detach().cpu(),
            cash_weights=self.cash_weights.detach().cpu(),
        )


class TorchDailyBacktester:
    """Daily backtester implemented with PyTorch tensors for autograd."""

    def __init__(
        self,
        trading_fee: float = 0.0,
        device: torch.device | None = None,
        trading_days: int = 252,
    ) -> None:
        self.cost_rate = float(trading_fee)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trading_days = trading_days

    def simulate(
        self,
        open_prices: torch.Tensor,
        close_prices: torch.Tensor,
        asset_weights: torch.Tensor,
        cash_weights: torch.Tensor,
        initial_capital: float = 100_000.0,
    ) -> SimulationResult:
        """Simulate trading with per-day weights. All tensors must share device/dtype."""

        opens = open_prices.to(self.device)
        closes = close_prices.to(self.device)
        weights = asset_weights.to(self.device)
        cash_w = cash_weights.to(self.device)

        if cash_w.ndim == 2 and cash_w.shape[1] == 1:
            cash_w = cash_w.squeeze(-1)

        dtype = opens.dtype
        equity = torch.tensor(initial_capital, dtype=dtype, device=self.device)
        equity_curve = []
        daily_returns = []

        prev_equity = equity
        for day in range(opens.shape[0]):
            w_assets = torch.clamp(weights[day], min=0.0)
            w_cash = torch.clamp(cash_w[day], min=0.0)

            total_weight = w_cash + w_assets.sum()
            if total_weight > 1.0:
                scale = 1.0 / total_weight
                w_assets = w_assets * scale
                w_cash = w_cash * scale
            else:
                w_cash = w_cash + (1.0 - total_weight)

            open_slice = opens[day]
            close_slice = closes[day]

            dollars_in_assets = equity * w_assets
            shares = dollars_in_assets / (open_slice + 1e-8)
            cash_balance = equity * w_cash

            portfolio_value = torch.sum(shares * close_slice) + cash_balance

            # Apply optional trading costs after valuation
            if self.cost_rate > 0:
                turnover = torch.sum(torch.abs(dollars_in_assets)) / (equity + 1e-8)
                portfolio_value = portfolio_value * (1.0 - self.cost_rate * turnover)

            equity = portfolio_value
            ret = portfolio_value / (prev_equity + 1e-8) - 1.0
            prev_equity = portfolio_value

            equity_curve.append(equity)
            daily_returns.append(ret)

        return SimulationResult(
            equity_curve=torch.stack(equity_curve),
            daily_returns=torch.stack(daily_returns),
            asset_weights=weights,
            cash_weights=cash_w,
        )

    def summarize(self, result: SimulationResult, initial_capital: float) -> dict:
        equity_curve = result.equity_curve
        daily_returns = result.daily_returns
        final_value = equity_curve[-1]
        total_return = final_value / initial_capital - 1.0
        avg_daily = daily_returns.mean()
        std_daily = daily_returns.std(unbiased=False)
        sharpe = torch.sqrt(torch.tensor(self.trading_days, dtype=equity_curve.dtype, device=equity_curve.device)) * (
            avg_daily / (std_daily + 1e-8)
        )
        max_drawdown = self._max_drawdown(equity_curve)

        return {
            "final_equity": final_value.item(),
            "total_return": total_return.item(),
            "sharpe": sharpe.item(),
            "max_drawdown": max_drawdown.item(),
        }

    @staticmethod
    def _max_drawdown(equity_curve: torch.Tensor) -> torch.Tensor:
        running_max, _ = torch.cummax(equity_curve, dim=0)
        drawdowns = 1.0 - equity_curve / (running_max + 1e-8)
        return drawdowns.max()


class SoftmaxForecastPolicy(torch.nn.Module):
    """Simple differentiable policy that maps forecasts to asset/cash weights."""

    def __init__(self, num_assets: int) -> None:
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(0.0))
        self.asset_bias = torch.nn.Parameter(torch.zeros(num_assets))
        self.cash_logit = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, forecasts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled = forecasts * torch.exp(self.temperature) + self.asset_bias
        batch = scaled.shape[0]
        cash_logits = self.cash_logit.expand(batch, 1)
        logits = torch.cat([scaled, cash_logits], dim=-1)
        weights = torch.softmax(logits, dim=-1)
        asset_weights = weights[..., :-1]
        cash_weights = weights[..., -1]
        return asset_weights, cash_weights


def optimise_policy(
    simulator: TorchDailyBacktester,
    forecasts: torch.Tensor,
    opens: torch.Tensor,
    closes: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
    initial_capital: float = 100_000.0,
) -> Tuple[SoftmaxForecastPolicy, SimulationResult]:
    policy = SoftmaxForecastPolicy(num_assets=opens.shape[1]).to(simulator.device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)

    for step in range(1, steps + 1):
        asset_w, cash_w = policy(forecasts)
        sim_result = simulator.simulate(opens, closes, asset_w, cash_w, initial_capital=initial_capital)
        final_equity = sim_result.equity_curve[-1]
        loss = -torch.log(final_equity)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % max(steps // 5, 1) == 0:
            logger.info(
                "[step {}] final equity {:.2f}, loss {:.4f}",
                step,
                final_equity.item(),
                loss.item(),
            )

    with torch.no_grad():
        asset_w, cash_w = policy(forecasts)
        final_result = simulator.simulate(opens, closes, asset_w, cash_w, initial_capital=initial_capital)

    return policy, final_result


def run_torch_backtest(
    symbols: Iterable[str],
    simulation_days: int,
    lookback: int = 5,
    optimisation_steps: int = 200,
    lr: float = 0.05,
    initial_capital: float = 100_000.0,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opens, closes, forecasts, dates = prepare_tensors(
        symbols,
        simulation_days=simulation_days,
        lookback=lookback,
        device=device,
    )

    simulator = TorchDailyBacktester(device=device)
    policy, sim_result = optimise_policy(
        simulator,
        forecasts,
        opens,
        closes,
        steps=optimisation_steps,
        lr=lr,
        initial_capital=initial_capital,
    )

    summary = simulator.summarize(sim_result, initial_capital)
    summary.update(
        {
            "device": str(device),
            "dates": [str(d.date()) for d in dates],
            "symbols": list(symbols),
            "policy_state": {k: v.detach().cpu().tolist() for k, v in policy.state_dict().items()},
        }
    )

    sim_cpu = sim_result.detach()
    summary["equity_curve"] = sim_cpu.equity_curve.squeeze().tolist()
    summary["daily_returns"] = sim_cpu.daily_returns.squeeze().tolist()
    summary["asset_weights"] = sim_cpu.asset_weights.tolist()
    summary["cash_weights"] = sim_cpu.cash_weights.tolist()

    return summary
