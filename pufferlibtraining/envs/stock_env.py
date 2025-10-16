import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch


class StockTradingEnv(gym.Env):
    """
    Multi-asset trading environment with differentiable Torch PnL.

    The environment expects a dictionary mapping asset symbols to dataframes that
    already contain price history and (optionally) Amazon Toto forecast features.
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
        leverage_limit: float = 2.0,
        borrowing_cost_annual: float = 0.0675,
        transaction_cost_bps: float = 10.0,
        spread_bps: float = 1.0,
        feature_columns: Optional[Sequence[str]] = None,
        trading_days_per_year: int = 252,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if not asset_frames:
            raise ValueError("asset_frames must contain at least one asset dataframe.")

        if window_size < 2:
            raise ValueError("window_size must be >= 2 to build contextual observations.")

        self.asset_symbols = sorted(asset_frames.keys())
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.leverage_limit = float(leverage_limit)
        self.borrowing_cost_annual = float(borrowing_cost_annual)
        self.transaction_cost = float(transaction_cost_bps) / 10_000.0
        self.spread_cost = float(spread_bps) / 10_000.0
        self.trading_days_per_year = int(trading_days_per_year)
        self.borrowing_cost_daily = self.borrowing_cost_annual / self.trading_days_per_year
        self.device = device or torch.device("cpu")

        (
            self.feature_tensor,
            self.open_prices,
            self.close_prices,
            self.feature_names,
            self.dates,
        ) = self._prepare_asset_tensor(asset_frames, feature_columns=feature_columns)

        self.n_assets = len(self.asset_symbols)
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
        # Map raw actions to target weights while respecting leverage ceiling.
        target_weights = torch.tanh(action_tensor) * self.leverage_limit
        gross_exposure = target_weights.abs().sum()
        if gross_exposure > self.leverage_limit:
            target_weights = target_weights / (gross_exposure / self.leverage_limit)
            gross_exposure = target_weights.abs().sum()

        current_open = self.open_prices[self.current_index]
        current_close = self.close_prices[self.current_index]
        price_returns = (current_close - current_open) / torch.clamp(current_open, min=1e-8)

        prev_value = self.portfolio_value
        turnover = torch.abs(target_weights - self.current_weights)
        trade_cost = (turnover * prev_value) * self.transaction_cost
        spread_cost = (turnover * prev_value) * self.spread_cost
        total_trading_cost = trade_cost.sum() + spread_cost.sum()

        financing_cost = torch.clamp(gross_exposure - 1.0, min=0.0)
        financing_cost = financing_cost * prev_value * self.borrowing_cost_daily

        raw_profit = (target_weights * price_returns).sum() * prev_value
        net_profit = raw_profit - total_trading_cost - financing_cost
        self.portfolio_value = prev_value + net_profit

        step_return = net_profit / torch.clamp(prev_value, min=1e-8)
        reward = float((net_profit / self.initial_balance).clamp(min=-1e6, max=1e6).item())

        self.balance_history.append(float(self.portfolio_value.item()))
        self.leverage_history.append(float(gross_exposure.item()))
        self.turnover_history.append(float(turnover.sum().item()))
        self.returns_history.append(float(step_return.item()))

        trade_record = {
            "step": int(self.current_index - self.window_size),
            "weights_before": self.current_weights.detach().cpu().tolist(),
            "weights_after": target_weights.detach().cpu().tolist(),
            "raw_profit": float(raw_profit.item()),
            "net_profit": float(net_profit.item()),
            "transaction_cost": float(total_trading_cost.item()),
            "financing_cost": float(financing_cost.item()),
            "gross_exposure": float(gross_exposure.item()),
            "turnover": float(turnover.sum().item()),
        }
        self.trades.append(trade_record)

        self.current_weights = target_weights.detach()
        self.latest_gross = gross_exposure.detach()
        self.last_step_return = step_return.detach()

        self.current_index += 1
        terminated = self.current_index >= (self.n_steps - 1)
        truncated = False
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._get_observation()
        )

        info = {
            "portfolio_value": float(self.portfolio_value.item()),
            "step_return": float(step_return.item()),
            "gross_exposure": float(gross_exposure.item()),
            "turnover": float(turnover.sum().item()),
            "transaction_cost": float(total_trading_cost.item()),
            "financing_cost": float(financing_cost.item()),
            "raw_profit": float(raw_profit.item()),
            "net_profit": float(net_profit.item()),
            "date": self.dates[self.current_index - 1].isoformat() if self.dates is not None else None,
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
