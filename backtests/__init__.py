"""
Backtesting module for trading strategy simulation.
"""

from .simulate_trading_strategies import TradingSimulator
from .visualization_logger import VisualizationLogger

__version__ = "1.0.0"
__all__ = ["TradingSimulator", "VisualizationLogger"]