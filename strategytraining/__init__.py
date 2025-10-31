"""
Strategy Training Package

Position sizing dataset collection and analysis for trading strategy optimization.
"""

from .collect_position_sizing_dataset import (
    DatasetCollector,
    TradingCalendar
)
from .analyze_dataset import DatasetAnalyzer

__all__ = [
    'DatasetCollector',
    'TradingCalendar',
    'DatasetAnalyzer'
]

__version__ = '0.1.0'
