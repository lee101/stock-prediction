"""Bags.fm Neural Market Simulator for backtesting neural trading models."""

from .simulator import NeuralSimulator, BacktestResult, run_backtest

__all__ = ["NeuralSimulator", "BacktestResult", "run_backtest"]
