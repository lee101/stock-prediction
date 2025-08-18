import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any


class DailyTradingEnv(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        max_position_size: float = 1.0,
        features: list = None,
        spread_pct: float = 0.0001,  # 0.01% spread (bid-ask)
        slippage_pct: float = 0.0001,  # 0.01% slippage
        min_commission: float = 1.0  # Minimum $1 commission per trade
    ):
        super().__init__()
        
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.min_commission = min_commission
        
        if features is None:
            self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        else:
            self.features = features
        
        self.prices = self.df[['Open', 'Close']].values
        self.feature_data = self.df[self.features].values
        
        self.n_days = len(self.df) - self.window_size - 1
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, len(self.features) + 3),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.returns = []
        self.positions_history = []
        self.balance_history = [self.initial_balance]
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        start_idx = self.current_step
        end_idx = start_idx + self.window_size
        
        window_data = self.feature_data[start_idx:end_idx]
        
        normalized_data = (window_data - np.mean(window_data, axis=0)) / (np.std(window_data, axis=0) + 1e-8)
        
        position_info = np.full((self.window_size, 1), self.position)
        
        balance_ratio = self.balance / self.initial_balance
        balance_info = np.full((self.window_size, 1), balance_ratio)
        
        if self.position != 0 and self.entry_price > 0:
            current_price = self.prices[end_idx - 1, 1]
            pnl = (current_price - self.entry_price) / self.entry_price * self.position
        else:
            pnl = 0.0
        pnl_info = np.full((self.window_size, 1), pnl)
        
        observation = np.concatenate([
            normalized_data,
            position_info,
            balance_info,
            pnl_info
        ], axis=1)
        
        return observation.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = float(np.clip(action[0], -1.0, 1.0))
        
        current_idx = self.current_step + self.window_size
        current_open = self.prices[current_idx, 0]
        current_close = self.prices[current_idx, 1]
        
        old_position = self.position
        new_position = action * self.max_position_size
        
        reward = 0.0
        
        if old_position != 0:
            position_return = (current_close - current_open) / current_open
            if old_position > 0:
                profit = position_return * abs(old_position)
            else:
                profit = -position_return * abs(old_position)
            
            reward += profit * self.balance
            self.balance *= (1 + profit)
        
        if old_position != new_position:
            position_change = abs(new_position - old_position)
            
            # Calculate total transaction costs
            trade_value = position_change * self.balance
            
            # Commission (percentage or minimum)
            commission = max(self.transaction_cost * trade_value, self.min_commission)
            
            # Spread cost (bid-ask spread)
            spread_cost = self.spread_pct * trade_value
            
            # Slippage cost (market impact)
            slippage_cost = self.slippage_pct * trade_value
            
            total_cost = commission + spread_cost + slippage_cost
            
            self.balance -= total_cost
            reward -= total_cost / self.initial_balance
            
            if new_position != 0:
                self.entry_price = current_close
            else:
                self.entry_price = 0.0
            
            self.trades.append({
                'step': self.current_step,
                'action': action,
                'old_position': old_position,
                'new_position': new_position,
                'price': current_close,
                'balance': self.balance
            })
        
        self.position = new_position
        self.positions_history.append(self.position)
        self.balance_history.append(self.balance)
        
        reward = reward / self.initial_balance
        
        self.current_step += 1
        done = self.current_step >= self.n_days
        
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        daily_return = (self.balance - self.balance_history[-2]) / self.balance_history[-2] if len(self.balance_history) > 1 else 0
        self.returns.append(daily_return)
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'trades': len(self.trades),
            'current_price': current_close,
            'daily_return': daily_return
        }
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, Position: {self.position:.3f}")
    
    def get_metrics(self) -> Dict[str, float]:
        if len(self.returns) == 0:
            return {}
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        returns_array = np.array(self.returns)
        sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252) if len(returns_array) > 0 else 0
        
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        winning_trades = sum(1 for t in self.trades if t.get('profit', 0) > 0)
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': total_trades,
            'win_rate': win_rate,
            'final_balance': self.balance
        }