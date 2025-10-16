from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path


@dataclass
class ModelConfig:
    symbol: str
    model_path: Path
    window_size: int = 30
    features: List[str] = None
    action_std: float = 0.5
    
    def __post_init__(self):
        if self.features is None:
            self.features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'RSI', 'Volume_Ratio',
                'High_Low_Ratio', 'Close_Open_Ratio'
            ]


@dataclass
class TradingConfig:
    symbols: List[str]
    initial_balance: float = 100000.0
    max_position_size: float = 0.47  # Based on optimal findings
    max_positions: int = 2  # Based on optimal findings
    transaction_cost: float = 0.001
    spread_pct: float = 0.0001
    slippage_pct: float = 0.0001
    min_commission: float = 1.0
    
    # Risk management
    stop_loss: Optional[float] = 0.05  # 5% stop loss
    take_profit: Optional[float] = 0.20  # 20% take profit
    max_drawdown_stop: Optional[float] = 0.10  # 10% max drawdown
    
    # Model paths
    models_dir: Path = Path("models")
    use_top_k_models: bool = True
    ensemble_predictions: bool = False
    
    # Trading mode
    paper_trading: bool = True
    dry_run: bool = False
    
    # Logging
    log_dir: Path = Path("rlinference/logs")
    log_level: str = "INFO"
    
    # Market data
    data_refresh_interval: int = 60  # seconds
    
    # Safety
    max_daily_trades: int = 50
    max_position_value: float = 50000  # Max $ per position
    circuit_breaker_loss: float = 0.15  # Stop trading if down 15% in a day
    
    def __post_init__(self):
        self.models_dir = Path(self.models_dir)
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AlpacaConfig:
    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"  # Paper trading by default
    
    @classmethod
    def from_env(cls, paper: bool = True):
        import os
        if paper:
            return cls(
                api_key=os.getenv("ALP_KEY_ID_PAPER"),
                secret_key=os.getenv("ALP_SECRET_KEY_PAPER"),
                base_url="https://paper-api.alpaca.markets"
            )
        else:
            return cls(
                api_key=os.getenv("ALP_KEY_ID_PROD"),
                secret_key=os.getenv("ALP_SECRET_KEY_PROD"),
                base_url="https://api.alpaca.markets"
            )