"""
Strategy Training Package

Position sizing dataset collection and analysis for trading strategy optimization.
"""

try:
    from .collect_position_sizing_dataset import (
        DatasetCollector,
        TradingCalendar
    )
except ImportError:
    DatasetCollector = None
    TradingCalendar = None

from .collect_strategy_pnl_dataset import StrategyPnLCollector

try:
    from .analyze_dataset import DatasetAnalyzer
    from .analyze_strategy_dataset import StrategyDatasetAnalyzer
except ImportError:
    DatasetAnalyzer = None
    StrategyDatasetAnalyzer = None

__all__ = [
    'DatasetCollector',
    'TradingCalendar',
    'StrategyPnLCollector',
    'DatasetAnalyzer',
    'StrategyDatasetAnalyzer'
]

__version__ = '0.2.0'
