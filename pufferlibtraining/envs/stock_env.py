import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch

from src.leverage_settings import get_leverage_settings
from src.fees import get_fees_for_symbols


class StockTradingEnv(gym.Env):
    """
    Multi-asset trading environment with differentiable Torch PnL.

    The environment expects a dictionary mapping asset symbols to dataframes that
    already contain price history and (optionally) Datadog Toto forecast features.
    Each episode simulates portfolio rebalancing at market open and closes at the
    same day's close. Actions control target portfolio weights; the environment
    enforces a configurable leverage ceiling and applies financing charges for
    borrowed capital.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        asset_frames: Dict[str, pd.DataFrame],
        window_size: int = 30,
        initial_balance: float = 100_000.0,
        # Leverage behaviour
        leverage_limit: Optional[float] = None,
        borrowing_cost_annual: Optional[float] = None,
        trading_days_per_year: Optional[int] = None,
        # Fee model (all in bps except base per-asset fees which are looked up)
        transaction_cost_bps: float = 10.0,
        spread_bps: float = 1.0,
        # Two-tier leverage constraints and trade scheduling
        max_intraday_leverage: float = 4.0,
        max_overnight_leverage: float = 2.0,
        trade_timing: str = "open",  # "open" or "close"
        risk_scale: float = 1.0,
        feature_columns: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if not asset_frames:
            raise ValueError("asset_frames must contain at least one asset dataframe.")

        if window_size < 2:
            raise ValueError("window_size must be >= 2 to build contextual observations.")

        settings = get_leverage_settings()
        resolved_leverage_limit = settings.max_gross_leverage if leverage_limit is None else float(leverage_limit)
        resolved_borrowing_cost = settings.annual_cost if borrowing_cost_annual is None else float(borrowing_cost_annual)
        resolved_trading_days = settings.trading_days_per_year if trading_days_per_year is None else int(trading_days_per_year)

        self.asset_symbols = sorted(asset_frames.keys())
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.borrowing_cost_annual = float(resolved_borrowing_cost)
        self.transaction_cost = float(transaction_cost_bps) / 10_000.0
        self.spread_cost = float(spread_bps) / 10_000.0
        self.trading_days_per_year = max(1, int(resolved_trading_days))
        if self.borrowing_cost_annual > 0.0:
            self.borrowing_cost_daily = (1.0 + self.borrowing_cost_annual) ** (1.0 / self.trading_days_per_year) - 1.0
        else:
            self.borrowing_cost_daily = 0.0
        # Risk dial influences the effective intraday/overnight caps (monotonic)
        risk_scale = float(max(0.0, min(1.0, risk_scale)))
        intraday_cap = 1.0 + (float(max_intraday_leverage) - 1.0) * risk_scale
        overnight_cap = 1.0 + (float(max_overnight_leverage) - 1.0) * risk_scale
        if leverage_limit is None:
            resolved_leverage_limit = max(float(settings.max_gross_leverage), intraday_cap, overnight_cap)
        else:
            resolved_leverage_limit = float(leverage_limit)
        self.leverage_limit = resolved_leverage_limit
        self.max_intraday_leverage = min(intraday_cap, self.leverage_limit)
        self.max_overnight_leverage = min(overnight_cap, self.leverage_limit)
        if self.max_overnight_leverage > self.max_intraday_leverage:
            self.max_overnight_leverage = self.max_intraday_leverage
        trade_timing = (trade_timing or "open").strip().lower()
        if trade_timing not in {"open", "close"}:
            raise ValueError("trade_timing must be 'open' or 'close'")
        self.trade_timing = trade_timing
        self.device = device or torch.device("cpu")

        (
            self.feature_tensor,
            self.open_prices,
            self.close_prices,
            self.feature_names,
            self.dates,
        ) = self._prepare_asset_tensor(asset_frames, feature_columns=feature_columns)

        self.n_assets = len(self.asset_symbols)
        # Per-asset base fees (e.g., equities vs crypto) added on top of bps settings
        self.base_fee_rates = torch.tensor(
            get_fees_for_symbols(self.asset_symbols), dtype=torch.float32, device=self.device
        )
        self.feature_dim = self.feature_tensor.shape[-1]
        self.n_steps = self.feature_tensor.shape[0]
        if self.n_steps <= self.window_size + 1:
            raise ValueError("Insufficient overlapping history after alignment.")

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # Observation augments raw features with current allocation, balance ratio, and leverage.
        extra_features = 3  # allocation, balance ratio, leverage
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_assets, self.feature_dim + extra_features),
            dtype=np.float32,
        )

        self._reset_state()

    # ------------------------------------------------------------------ #
    # Environment lifecycle                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_state()
        observation = self._get_observation()
        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if action.shape != (self.n_assets,):
            raise ValueError(f"Action should have shape ({self.n_assets},), received {action.shape}")

        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        # Map raw actions to target weights; intraday ceiling applies to 'open' trading.
        scaled = torch.tanh(action_tensor) * self.max_intraday_leverage
        gross_intraday = scaled.abs().sum()
        if gross_intraday > self.max_intraday_leverage:
            scaled = scaled / (gross_intraday / self.max_intraday_leverage)
            gross_intraday = scaled.abs().sum()

        current_open = self.open_prices[self.current_index]
        current_close = self.close_prices[self.current_index]
        price_returns_oc = (current_close - current_open) / torch.clamp(current_open, min=1e-8)

        prev_value = self.portfolio_value
        per_asset_cost_rate = self.base_fee_rates + self.transaction_cost + self.spread_cost

        total_trading_cost = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        financing_cost = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        if self.trade_timing == "open":
            # Trade at the open: move current -> scaled, pay turnover costs now.
            turnover_open = torch.abs(scaled - self.current_weights)
            total_trading_cost += (turnover_open * prev_value * per_asset_cost_rate).sum()

            # Hold intraday at 'scaled' weights; charge financing on intraday leverage.
            financing_cost += torch.clamp(gross_intraday - 1.0, min=0.0) * prev_value * self.borrowing_cost_daily
            raw_profit = (scaled * price_returns_oc).sum() * prev_value

            # Auto-deleverage at close to overnight cap if necessary (pays extra turnover).
            gross_after = gross_intraday
            if gross_after > self.max_overnight_leverage:
                shrink = self.max_overnight_leverage / torch.clamp(gross_after, min=1e-8)
                overnight_weights = scaled * shrink
                turnover_close = torch.abs(overnight_weights - scaled)
                total_trading_cost += (turnover_close * prev_value * per_asset_cost_rate).sum()
            else:
                overnight_weights = scaled

            next_weights = overnight_weights.detach()
            gross_for_info = gross_intraday.detach()
        else:  # trade at close
            # Hold existing weights intraday, then rebalance to 'scaled' at the close.
            gross_hold = self.current_weights.abs().sum()
            financing_cost += torch.clamp(gross_hold - 1.0, min=0.0) * prev_value * self.borrowing_cost_daily
            raw_profit = (self.current_weights * price_returns_oc).sum() * prev_value

            # At close: apply new target but enforce overnight cap.
            gross_target = scaled.abs().sum()
            if gross_target > self.max_overnight_leverage:
                scaled = scaled / (gross_target / self.max_overnight_leverage)
                gross_target = scaled.abs().sum()
            turnover_close = torch.abs(scaled - self.current_weights)
            total_trading_cost += (turnover_close * prev_value * per_asset_cost_rate).sum()
            next_weights = scaled.detach()
            gross_for_info = gross_target.detach()

        net_profit = raw_profit - total_trading_cost - financing_cost
        self.portfolio_value = prev_value + net_profit

        step_return = net_profit / torch.clamp(prev_value, min=1e-8)
        reward = float((net_profit / self.initial_balance).clamp(min=-1e6, max=1e6).item())

        self.balance_history.append(float(self.portfolio_value.item()))
        self.leverage_history.append(float(gross_for_info.item()))
        # Turnover is embedded in costs; we report aggregate by dividing total by rate avg.
        self.turnover_history.append(float(torch.abs(next_weights - self.current_weights).sum().item()))
        self.returns_history.append(float(step_return.item()))

        trade_record = {
            "step": int(self.current_index - self.window_size),
            "weights_before": self.current_weights.detach().cpu().tolist(),
            "weights_after": next_weights.detach().cpu().tolist(),
            "raw_profit": float(raw_profit.item()),
            "net_profit": float(net_profit.item()),
            "transaction_cost": float(total_trading_cost.item()),
            "financing_cost": float(financing_cost.item()),
            "gross_exposure": float(gross_for_info.item()),
            "turnover": float(torch.abs(next_weights - self.current_weights).sum().item()),
            "trade_timing": self.trade_timing,
        }
        self.trades.append(trade_record)

        self.current_weights = next_weights
        self.latest_gross = gross_for_info
        self.last_step_return = step_return.detach()

        self.current_index += 1
        terminated = self.current_index >= (self.n_steps - 1)
        truncated = False
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._get_observation()
        )

        if self.dates is not None:
            raw_date = self.dates[self.current_index - 1]
            if isinstance(raw_date, np.datetime64):
                # Use pandas to normalise numpy datetime64 (preserves tz if present)
                date_value = pd.Timestamp(raw_date).isoformat()
            elif hasattr(raw_date, "isoformat"):
                date_value = raw_date.isoformat()
            else:
                date_value = str(raw_date)
        else:
            date_value = None

        info = {
            "portfolio_value": float(self.portfolio_value.item()),
            "step_return": float(step_return.item()),
            "gross_exposure": float(gross_for_info.item()),
            "turnover": float(torch.abs(next_weights - self.current_weights).sum().item()),
            "transaction_cost": float(total_trading_cost.item()),
            "financing_cost": float(financing_cost.item()),
            "raw_profit": float(raw_profit.item()),
            "net_profit": float(net_profit.item()),
            "date": date_value,
            "trade_timing": self.trade_timing,
            "max_intraday_leverage": float(self.max_intraday_leverage),
            "max_overnight_leverage": float(self.max_overnight_leverage),
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        weights = ", ".join(
            f"{sym}:{w:+.2f}"
            for sym, w in zip(self.asset_symbols, self.current_weights.tolist())
        )
        print(
            f"Step {self.current_index - self.window_size:04d} | "
            f"Value ${self.portfolio_value.item():,.2f} | "
            f"Exposure {float(self.latest_gross.item()):.2f} | "
            f"Weights [{weights}]"
        )

    def get_metrics(self) -> Dict[str, float]:
        if len(self.balance_history) < 2:
            return {}

        balance_tensor = torch.tensor(self.balance_history, dtype=torch.float32)
        returns_tensor = torch.tensor(self.returns_history, dtype=torch.float32)

        total_return = balance_tensor[-1] / balance_tensor[0] - 1.0
        sharpe = (
            returns_tensor.mean()
            / (returns_tensor.std(unbiased=False) + 1e-8)
            * math.sqrt(self.trading_days_per_year)
        )

        cumulative = torch.cumprod(1 + returns_tensor, dim=0)
        running_max = torch.empty_like(cumulative)
        max_val = torch.tensor(1.0, dtype=torch.float32)
        for idx, val in enumerate(cumulative):
            max_val = torch.maximum(max_val, val)
            running_max[idx] = max_val
        drawdown = (cumulative - running_max) / torch.clamp(running_max, min=1e-8)
        max_drawdown = drawdown.min()

        trade_profits = torch.tensor(
            [trade["net_profit"] for trade in self.trades],
            dtype=torch.float32,
        )
        if trade_profits.numel() > 0:
            win_rate = (trade_profits > 0).float().mean()
        else:
            win_rate = torch.tensor(0.0)

        return {
            "total_return": float(total_return.item()),
            "sharpe_ratio": float(sharpe.item()),
            "max_drawdown": float(max_drawdown.item()),
            "num_trades": int(len(self.trades)),
            "win_rate": float(win_rate.item()),
            "final_balance": float(balance_tensor[-1].item()),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _reset_state(self) -> None:
        self.current_index = self.window_size
        self.portfolio_value = torch.tensor(self.initial_balance, dtype=torch.float32, device=self.device)
        self.current_weights = torch.zeros(self.n_assets, dtype=torch.float32, device=self.device)
        self.latest_gross = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.last_step_return = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        self.balance_history: List[float] = [self.initial_balance]
        self.leverage_history: List[float] = [0.0]
        self.turnover_history: List[float] = []
        self.returns_history: List[float] = []
        self.trades: List[Dict[str, Any]] = []

    def _get_observation(self) -> np.ndarray:
        start = self.current_index - self.window_size
        end = self.current_index
        window_features = self.feature_tensor[start:end]

        allocation_info = self.current_weights.view(1, self.n_assets, 1).expand(self.window_size, -1, -1)
        balance_ratio = (self.portfolio_value / self.initial_balance).clamp(min=1e-8)
        balance_info = torch.full(
            (self.window_size, self.n_assets, 1),
            float(balance_ratio.item()),
            dtype=torch.float32,
            device=self.device,
        )
        leverage_info = torch.full(
            (self.window_size, self.n_assets, 1),
            float(self.latest_gross.item()),
            dtype=torch.float32,
            device=self.device,
        )
        observation = torch.cat(
            [window_features, allocation_info, balance_info, leverage_info],
            dim=-1,
        )
        return observation.detach().cpu().numpy().astype(np.float32)

    def _prepare_asset_tensor(
        self,
        asset_frames: Dict[str, pd.DataFrame],
        feature_columns: Optional[Sequence[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], Optional[np.ndarray]]:
        prepared: Dict[str, pd.DataFrame] = {}
        for symbol, frame in asset_frames.items():
            prepared[symbol] = self._standardise_frame(symbol, frame)

        date_sets = [set(df["date"].values) for df in prepared.values()]
        common_dates = sorted(set.intersection(*date_sets))
        if not common_dates:
            raise ValueError("No overlapping dates across provided asset dataframes.")

        aligned: Dict[str, pd.DataFrame] = {}
        for symbol, df in prepared.items():
            aligned_df = df[df["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
            aligned[symbol] = aligned_df

        numeric_maps: Dict[str, Dict[str, str]] = {}
        for symbol, df in aligned.items():
            numeric_cols = [
                col for col in df.columns
                if col != "date" and pd.api.types.is_numeric_dtype(df[col])
            ]
            normalised: Dict[str, str] = {}
            prefix = f"{symbol.lower()}_"
            for col in numeric_cols:
                base = col[len(prefix):] if col.startswith(prefix) else col
                normalised[base] = col
            numeric_maps[symbol] = normalised

        if feature_columns is None:
            common_feature_names = set.intersection(
                *(set(cols.keys()) for cols in numeric_maps.values())
            )
            if "open" not in common_feature_names or "close" not in common_feature_names:
                raise ValueError("Aligned dataframes must contain 'open' and 'close' columns.")
            feature_names = sorted(common_feature_names)
        else:
            feature_names = [name.lower() for name in feature_columns]
            common_feature_names = set(feature_names)
            for symbol, cols in numeric_maps.items():
                missing = [name for name in feature_names if name not in cols]
                if missing:
                    raise ValueError(f"Missing feature columns {missing} for symbol {symbol}.")

        feature_arrays: List[np.ndarray] = []
        open_list: List[np.ndarray] = []
        close_list: List[np.ndarray] = []
        for symbol in self.asset_symbols:
            df = aligned[symbol]
            cols_map = numeric_maps[symbol]
            selected_cols = [cols_map[name] for name in feature_names]
            feature_arrays.append(df[selected_cols].to_numpy(dtype=np.float32))
            open_list.append(df[cols_map["open"]].to_numpy(dtype=np.float32))
            close_list.append(df[cols_map["close"]].to_numpy(dtype=np.float32))

        feature_array = np.stack(feature_arrays, axis=1)  # (time, assets, features)
        open_array = np.stack(open_list, axis=1)
        close_array = np.stack(close_list, axis=1)

        feature_tensor = torch.from_numpy(feature_array).to(self.device)
        open_tensor = torch.from_numpy(open_array).to(self.device)
        close_tensor = torch.from_numpy(close_array).to(self.device)
        date_array = np.array(common_dates)

        return feature_tensor, open_tensor, close_tensor, feature_names, date_array

    @staticmethod
    def _standardise_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            raise ValueError(f"No data provided for symbol {symbol}.")

        df = frame.copy()
        df.columns = [col.lower() for col in df.columns]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"])
        else:
            df["date"] = pd.date_range(start="2000-01-01", periods=len(df), freq="D")

        numeric_cols = [
            col for col in df.columns
            if col != "date" and pd.api.types.is_numeric_dtype(df[col])
        ]
        df = df[["date"] + numeric_cols]
        df = df.dropna().reset_index(drop=True)
        return df
