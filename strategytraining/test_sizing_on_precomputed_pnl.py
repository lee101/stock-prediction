#!/usr/bin/env python3
"""
Fast position sizing strategy testing using precomputed PnL data.

Uses strategytraining/ precomputed trades to quickly evaluate different
sizing strategies without re-running full market simulation.

Usage:
    python strategytraining/test_sizing_on_precomputed_pnl.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Deque
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marketsimulator.sizing_strategies import (
    FixedFractionStrategy,
    KellyStrategy,
    VolatilityTargetStrategy,
    CorrelationAwareStrategy,
    VolatilityAdjustedStrategy,
    MarketContext,
)
from trainingdata.load_correlation_utils import load_correlation_matrix


@dataclass
class SizingStrategyResult:
    """Results for a sizing strategy tested on precomputed trades."""
    strategy_name: str
    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    avg_position_size: float
    volatility: float
    win_rate: float
    gate_normal_days: int = 0
    gate_probe_days: int = 0
    gate_blocked_days: int = 0
    gate_config_name: Optional[str] = None
    daily_curve: List[Dict[str, object]] = field(default_factory=list)
    sortino_ratio: float = 0.0
    annualized_return_pct: float = 0.0
    annualization_days: float = 252.0
    symbol_gate_blocks: int = 0
    symbol_gate_probes: int = 0


@dataclass
class GlobalGateConfig:
    """Configuration for applying global PnL gating rules."""

    name: str
    window_days: int
    fail_mode: str = "probe"  # "probe" keeps trading at reduced size, "block" stops trading
    probe_fraction: float = 0.1
    min_positive: float = 1e-9
    use_strategy_pnl: bool = False
    scope: str = "account"  # "account" or "symbol_side"
    window_trades: int = 0

    def __post_init__(self) -> None:
        if self.window_days < 1:
            raise ValueError("window_days must be >= 1")
        if self.fail_mode not in {"probe", "block"}:
            raise ValueError("fail_mode must be 'probe' or 'block'")
        if not 0.0 <= self.probe_fraction <= 1.0:
            raise ValueError("probe_fraction must be within [0, 1]")
        if self.fail_mode == "probe" and self.probe_fraction == 0.0:
            raise ValueError("probe_fraction must be > 0 when fail_mode='probe'")
        if self.scope == "symbol_side" and self.window_trades < 1:
            raise ValueError("window_trades must be >= 1 for symbol_side scope")


class PrecomputedPnLSizingTester:
    """
    Fast sizing strategy tester using precomputed trade data.

    Takes trade-level PnL data and applies different sizing strategies
    to see how they would have performed.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        initial_capital: float = 100000,
        corr_data: Dict = None,
    ):
        """
        Args:
            trades_df: DataFrame with precomputed trades
            initial_capital: Starting capital
            corr_data: Correlation matrix data for advanced strategies
        """
        self.trades_df = trades_df.copy()
        self.initial_capital = initial_capital
        self.corr_data = corr_data

        # Sort by entry timestamp
        self.trades_df = self.trades_df.sort_values('entry_timestamp').reset_index(drop=True)

        # Add time index for chronological processing
        self.trades_df['time_idx'] = pd.to_datetime(self.trades_df['entry_timestamp'])
        self.trades_df['trade_day'] = self.trades_df['time_idx'].dt.normalize()

        # Cache daily ordering and baseline PnL (pre-sizing) for gating decisions
        trade_day_index = pd.Index(self.trades_df['trade_day'])
        self.trade_days = list(trade_day_index.drop_duplicates().sort_values())

        daily_baseline = (
            self.trades_df.groupby('trade_day')['pnl']
            .sum()
            .reindex(self.trade_days, fill_value=0.0)
        )
        self.baseline_day_pnls = daily_baseline.tolist()
        self.day_to_index = {day: idx for idx, day in enumerate(self.trade_days)}

    def _get_account_gate_state(
        self,
        day_idx: int,
        gate_config: GlobalGateConfig,
        strategy_history: List[float],
    ) -> Tuple[float, str]:
        """Evaluate an account-level gate for the provided day index."""

        if gate_config.use_strategy_pnl:
            if len(strategy_history) < gate_config.window_days:
                return 1.0, "normal"
            window_values = strategy_history[-gate_config.window_days:]
            window_sum = float(np.sum(window_values))
        else:
            if day_idx < gate_config.window_days:
                return 1.0, "normal"
            window_start = day_idx - gate_config.window_days
            window_values = self.baseline_day_pnls[window_start:day_idx]
            window_sum = float(np.sum(window_values))

        if window_sum > gate_config.min_positive:
            return 1.0, "normal"

        if gate_config.fail_mode == "probe":
            return gate_config.probe_fraction, "probe"
        return 0.0, "blocked"

    @staticmethod
    def _get_symbol_gate_state(
        gate_config: GlobalGateConfig,
        history: Deque[float],
    ) -> Tuple[float, str]:
        """Evaluate a symbol-direction gate based on recent trade history."""

        if len(history) < gate_config.window_trades:
            return 1.0, "normal"

        window_sum = float(np.sum(list(history)[-gate_config.window_trades:]))

        if window_sum > gate_config.min_positive:
            return 1.0, "normal"

        if gate_config.fail_mode == "probe":
            return gate_config.probe_fraction, "probe"
        return 0.0, "blocked"

    def run_strategy(
        self,
        strategy,
        strategy_name: str,
        gate_config: Optional[Union[GlobalGateConfig, List[GlobalGateConfig]]] = None,
    ) -> SizingStrategyResult:
        """
        Apply sizing strategy (optionally gated by global PnL rules).

        Key insight: The precomputed trades have a baseline position_size.
        We calculate what size the strategy would have used, then scale
        the PnL accordingly.

        Args:
            strategy: Sizing strategy instance
            strategy_name: Name for reporting
            gate_config: Optional global PnL gate configuration

        Returns:
            SizingStrategyResult with performance metrics
        """
        capital = self.initial_capital
        capital_history = [capital]
        position_sizes: List[float] = []
        scaled_pnls: List[float] = []
        gate_configs: List[GlobalGateConfig]
        if gate_config is None:
            gate_configs = []
        elif isinstance(gate_config, (list, tuple)):
            gate_configs = list(gate_config)
        else:
            gate_configs = [gate_config]

        for cfg in gate_configs:
            if cfg.scope not in {"account", "symbol_side"}:
                raise ValueError(f"Unsupported gate scope: {cfg.scope}")

        account_gates = [cfg for cfg in gate_configs if cfg.scope == "account"]
        symbol_gates = [cfg for cfg in gate_configs if cfg.scope == "symbol_side"]

        gate_mode_counts = {"normal": 0, "probe": 0, "blocked": 0}
        daily_curve: List[Dict[str, object]] = []
        daily_returns: List[float] = []
        downside_returns: List[float] = []
        day_freqs: List[float] = []
        account_gate_histories: Dict[str, List[float]] = {cfg.name: [] for cfg in account_gates}
        symbol_gate_histories: Dict[str, Dict[Tuple[str, str], Deque[float]]] = {
            cfg.name: defaultdict(lambda: deque(maxlen=cfg.window_trades)) for cfg in symbol_gates
        }
        symbol_gate_blocks = 0
        symbol_gate_probes = 0

        # Iterate day-by-day to evaluate gating rules
        for day_idx, day in enumerate(self.trade_days):
            day_multiplier = 1.0
            gate_mode = "normal"

            for cfg in account_gates:
                multiplier, mode = self._get_account_gate_state(
                    day_idx,
                    cfg,
                    account_gate_histories[cfg.name],
                )
                if mode == "blocked":
                    day_multiplier = 0.0
                    gate_mode = "blocked"
                    break
                if mode == "probe":
                    gate_mode = "probe" if gate_mode == "normal" else gate_mode
                    day_multiplier *= cfg.probe_fraction

            gate_mode_counts[gate_mode] += 1

            day_trades = self.trades_df[self.trades_df['trade_day'] == day]
            capital_before_day = capital
            day_has_crypto = bool(day_trades['is_crypto'].any())
            day_has_stock = bool((~day_trades['is_crypto']).any())

            if day_has_stock and not day_has_crypto:
                day_freq = 252.0
                day_class = "stock"
            elif day_has_crypto and not day_has_stock:
                day_freq = 365.0
                day_class = "crypto"
            else:
                day_freq = (252.0 + 365.0) / 2.0 if not day_trades.empty else 252.0
                day_class = "mixed" if not day_trades.empty else "none"

            # Within a day, still preserve timestamp ordering for concurrent trades
            for timestamp, trades_at_time in day_trades.groupby('time_idx'):
                current_equity = capital

                for idx, trade in trades_at_time.iterrows():
                    symbol = trade['symbol']
                    is_crypto = trade['is_crypto']

                    # Estimate the predicted return from the realized PnL
                    predicted_return = trade['pnl_pct']

                    # Estimate volatility from historical data or use default
                    if self.corr_data and symbol in self.corr_data.get('volatility_metrics', {}):
                        vol_metrics = self.corr_data['volatility_metrics'][symbol]
                        predicted_volatility = vol_metrics['annualized_volatility'] / np.sqrt(252)
                    else:
                        predicted_volatility = 0.02

                    ctx = MarketContext(
                        symbol=symbol,
                        predicted_return=abs(predicted_return),
                        predicted_volatility=predicted_volatility,
                        current_price=trade['entry_price'],
                        equity=current_equity,
                        is_crypto=is_crypto,
                        existing_position_value=0,
                    )

                    try:
                        sizing = strategy.calculate_size(ctx)
                        position_fraction = sizing.position_fraction
                    except Exception:
                        position_fraction = 0.5 / max(len(trades_at_time), 1)

                    # Apply account-level gate multiplier
                    effective_fraction = position_fraction * day_multiplier

                    # Apply symbol-direction gates sequentially
                    symbol_mode = "normal"
                    symbol_multiplier = 1.0
                    symbol_key = (symbol.upper(), "long" if position_fraction >= 0 else "short")
                    for cfg in symbol_gates:
                        history = symbol_gate_histories[cfg.name][symbol_key]
                        multiplier, mode = self._get_symbol_gate_state(cfg, history)
                        if mode == "blocked":
                            symbol_multiplier = 0.0
                            symbol_mode = "blocked"
                            break
                        if mode == "probe":
                            symbol_mode = "probe" if symbol_mode == "normal" else symbol_mode
                            symbol_multiplier *= cfg.probe_fraction

                    if symbol_mode == "blocked":
                        symbol_gate_blocks += 1
                    elif symbol_mode == "probe":
                        symbol_gate_probes += 1

                    effective_fraction *= symbol_multiplier

                    baseline_position_size = trade['position_size']
                    baseline_fraction = (
                        baseline_position_size * trade['entry_price'] / self.initial_capital
                    )

                    if gate_mode == "blocked":
                        size_multiplier = 0.0
                    else:
                        if baseline_fraction != 0:
                            size_multiplier = effective_fraction / baseline_fraction
                        else:
                            size_multiplier = effective_fraction

                        # Prevent extreme leverage swings but allow probe-level sizing (<0.1x)
                        size_multiplier = float(np.clip(size_multiplier, -10.0, 10.0))

                    scaled_pnl = trade['pnl'] * size_multiplier
                    capital += scaled_pnl

                    scaled_pnls.append(scaled_pnl)
                    position_sizes.append(effective_fraction)

                    # Update symbol histories with either realized or baseline pnl
                    for cfg in symbol_gates:
                        history = symbol_gate_histories[cfg.name][symbol_key]
                        if cfg.use_strategy_pnl:
                            gate_value = scaled_pnl if symbol_mode != "blocked" else trade['pnl']
                        else:
                            gate_value = trade['pnl']
                        history.append(float(gate_value))

                capital_history.append(capital)

            # Daily metrics for visualization/reporting
            day_pnl = capital - capital_before_day

            if capital_before_day > 0:
                daily_return = day_pnl / capital_before_day
            else:
                daily_return = 0.0

            daily_returns.append(daily_return)
            downside_returns.append(min(0.0, daily_return))
            day_freqs.append(day_freq)

            if len(daily_returns) > 1:
                rolling_mean = float(np.mean(daily_returns))
                rolling_std = float(np.std(daily_returns))
                current_annualization = float(np.mean(day_freqs))
                rolling_sharpe = (
                    (rolling_mean / (rolling_std + 1e-10)) * np.sqrt(current_annualization)
                    if rolling_std > 0
                    else 0.0
                )
                downside_array = np.array(downside_returns, dtype=float)
                downside_std = float(np.sqrt(np.mean(np.square(downside_array))))
                rolling_sortino = (
                    (rolling_mean / (downside_std + 1e-10)) * np.sqrt(current_annualization)
                    if downside_std > 0
                    else 0.0
                )
            else:
                rolling_sharpe = 0.0
                rolling_sortino = 0.0
                current_annualization = day_freq

            elapsed_days = len(daily_returns)
            if elapsed_days > 0 and capital > 0:
                rolling_ann_return = (capital / self.initial_capital) ** (current_annualization / elapsed_days) - 1
            else:
                rolling_ann_return = 0.0

            daily_curve.append({
                'date': day.isoformat(),
                'capital': float(capital),
                'daily_return': float(daily_return),
                'daily_pnl': float(day_pnl),
                'rolling_sharpe': float(rolling_sharpe),
                'rolling_sortino': float(rolling_sortino),
                'rolling_ann_return': float(rolling_ann_return),
                'annualization_days': float(current_annualization),
                'mode': gate_mode,
                'day_class': day_class,
            })

            for cfg in account_gates:
                if cfg.use_strategy_pnl:
                    if gate_mode == "blocked":
                        gate_value = self.baseline_day_pnls[day_idx]
                    else:
                        gate_value = day_pnl
                    account_gate_histories[cfg.name].append(float(gate_value))

        gate_config_name = "+".join(cfg.name for cfg in gate_configs) if gate_configs else None
        annualization_factor = float(np.mean(day_freqs)) if day_freqs else 252.0

        # Calculate metrics
        total_pnl = capital - self.initial_capital
        total_return_pct = total_pnl / self.initial_capital

        # Sharpe ratio
        if len(capital_history) > 1:
            returns = np.diff(capital_history) / capital_history[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(annualization_factor)
            volatility = np.std(returns) * np.sqrt(annualization_factor)
        else:
            sharpe = 0.0
            volatility = 0.0

        mean_daily_return = float(np.mean(daily_returns)) if daily_returns else 0.0
        if downside_returns:
            downside_array = np.array(downside_returns, dtype=float)
            downside_std = float(np.sqrt(np.mean(np.square(downside_array))))
        else:
            downside_std = 0.0

        if downside_std > 0:
            sortino = (mean_daily_return / (downside_std + 1e-10)) * np.sqrt(annualization_factor)
        else:
            sortino = 0.0

        if daily_returns:
            clipped_returns = np.clip(daily_returns, -0.9999, None)
            mean_log_return = float(np.mean(np.log1p(clipped_returns)))
            annualized_return_pct = float(np.expm1(mean_log_return * annualization_factor))
        else:
            annualized_return_pct = 0.0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for eq in capital_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        # Win rate
        wins = sum(1 for pnl in scaled_pnls if pnl > 0)
        win_rate = wins / len(scaled_pnls) if scaled_pnls else 0.0

        avg_position_size = np.mean(position_sizes) if position_sizes else 0.0

        return SizingStrategyResult(
            strategy_name=strategy_name,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            num_trades=len(scaled_pnls),
            avg_position_size=avg_position_size,
            volatility=volatility,
            win_rate=win_rate,
            gate_normal_days=gate_mode_counts["normal"],
            gate_probe_days=gate_mode_counts["probe"],
            gate_blocked_days=gate_mode_counts["blocked"],
            gate_config_name=gate_config_name,
            daily_curve=daily_curve,
            sortino_ratio=sortino,
            annualized_return_pct=annualized_return_pct,
            annualization_days=annualization_factor,
            symbol_gate_blocks=symbol_gate_blocks,
            symbol_gate_probes=symbol_gate_probes,
        )


def build_daily_metrics_df(results: List[SizingStrategyResult]) -> pd.DataFrame:
    """Flatten per-strategy daily curves into a single DataFrame."""

    rows: List[Dict[str, object]] = []
    for result in results:
        for point in result.daily_curve:
            rows.append({
                'strategy': result.strategy_name,
                'date': point['date'],
                'capital': point['capital'],
                'daily_return': point['daily_return'],
                'rolling_sharpe': point['rolling_sharpe'],
                'rolling_sortino': point.get('rolling_sortino', 0.0),
                'rolling_ann_return': point.get('rolling_ann_return', 0.0),
                'daily_pnl': point.get('daily_pnl', 0.0),
                'mode': point.get('mode', 'normal'),
                'day_class': point.get('day_class', 'unknown'),
                'gate_config': result.gate_config_name or '-',
                'annualization_days': point.get('annualization_days', result.annualization_days),
            })

    if not rows:
        return pd.DataFrame(columns=['strategy', 'date', 'capital', 'daily_return', 'daily_pnl', 'rolling_sharpe', 'rolling_sortino', 'rolling_ann_return', 'mode', 'day_class', 'gate_config', 'annualization_days'])

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['strategy', 'date']).reset_index(drop=True)


def generate_visualizations(
    results: List[SizingStrategyResult],
    daily_metrics_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 5,
) -> None:
    """Create aggregate plots for equity curves and rolling Sharpe."""

    if daily_metrics_df.empty:
        print("⚠️  No daily metrics available for visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    def _plot(
        metric: str,
        ylabel: str,
        filename: str,
        selected: List[SizingStrategyResult],
        target: Optional[float] = None,
        extra: Optional[List[SizingStrategyResult]] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))

        combined = list(selected)
        if extra:
            existing = {r.strategy_name for r in combined}
            for result in extra:
                if result.strategy_name not in existing:
                    combined.append(result)
                    existing.add(result.strategy_name)

        plotted = False
        for result in combined:
            strategy_df = daily_metrics_df[daily_metrics_df['strategy'] == result.strategy_name]
            if strategy_df.empty:
                continue
            ax.plot(strategy_df['date'], strategy_df[metric], label=result.strategy_name)
            plotted = True

        if not plotted:
            plt.close(fig)
            return

        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} over Time ({len(selected)} strategies)')
        ax.legend(loc='upper left', ncol=2, fontsize=9)
        if target is not None:
            ax.axhline(target, color='red', linestyle='--', linewidth=1.0, label=f'Target {target:.0%}')
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)

    top_by_return = sorted(results, key=lambda r: r.total_return_pct, reverse=True)[:top_n]
    top_by_sharpe = sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)[:top_n]
    top_by_sortino = sorted(results, key=lambda r: r.sortino_ratio, reverse=True)[:top_n]
    top_by_ann_return = sorted(results, key=lambda r: r.annualized_return_pct, reverse=True)[:top_n]

    gate_highlights = [
        r for r in results
        if r.gate_blocked_days > 0 or r.symbol_gate_blocks > 0
    ]
    gate_highlights = sorted(gate_highlights, key=lambda r: r.sharpe_ratio, reverse=True)[:top_n]

    _plot('capital', 'Equity (USD)', 'top_return_equity.png', top_by_return, extra=gate_highlights)
    _plot('rolling_sharpe', 'Rolling Sharpe (annualized)', 'top_sharpe_curves.png', top_by_sharpe, extra=gate_highlights)
    _plot('rolling_sortino', 'Rolling Sortino (annualized)', 'top_sortino_curves.png', top_by_sortino, extra=gate_highlights)
    _plot('rolling_ann_return', 'Rolling Annualized Return', 'rolling_ann_return.png', top_by_ann_return, target=0.60, extra=gate_highlights)


def main():
    print("=" * 80)
    print("FAST POSITION SIZING TESTING ON PRECOMPUTED PnL DATA")
    print("=" * 80)
    print()

    # Load latest full dataset
    dataset_path = Path("strategytraining/datasets")
    latest_files = sorted(dataset_path.glob("full_strategy_dataset_*_trades.parquet"))

    if not latest_files:
        print("ERROR: No precomputed trade data found!")
        print("Run: python strategytraining/collect_strategy_pnl_dataset.py")
        return

    trades_file = latest_files[-1]
    print(f"Loading: {trades_file.name}")

    trades_df = pd.read_parquet(trades_file)
    print(f"✓ Loaded {len(trades_df):,} trades")

    # Filter to test symbols
    test_symbols = ['BTCUSD', 'ETHUSD', 'AAPL', 'MSFT', 'NVDA', 'SPY']

    # Map symbol names (dataset uses BTC-USD, we use BTCUSD)
    symbol_mapping = {
        'BTC-USD': 'BTCUSD',
        'ETH-USD': 'ETHUSD',
    }

    trades_df['symbol'] = trades_df['symbol'].replace(symbol_mapping)

    # Load performance data to filter to profitable windows
    perf_file = str(trades_file).replace('_trades.parquet', '_strategy_performance.parquet')
    perf_df = pd.read_parquet(perf_file)
    perf_df['symbol'] = perf_df['symbol'].replace(symbol_mapping)

    # Filter to test symbols
    available_symbols = [s for s in test_symbols if s in trades_df['symbol'].values]
    print(f"Available test symbols: {', '.join(available_symbols)}")

    trades_df = trades_df[trades_df['symbol'].isin(available_symbols)].copy()
    perf_df = perf_df[perf_df['symbol'].isin(available_symbols)].copy()

    print(f"Total trades on test symbols: {len(trades_df):,}")

    # Filter to profitable strategy windows only
    profitable_windows = perf_df[perf_df['total_return'] > 0][['symbol', 'strategy', 'window_num']]
    print(f"Profitable strategy windows: {len(profitable_windows)} / {len(perf_df)} ({100*len(profitable_windows)/len(perf_df):.1f}%)")

    # Merge to keep only trades from profitable windows
    trades_df = trades_df.merge(
        profitable_windows,
        on=['symbol', 'strategy', 'window_num'],
        how='inner'
    )

    print(f"✓ Filtered to {len(trades_df):,} trades from profitable windows")
    print()

    # Load correlation data
    print("Loading correlation and volatility data...")
    try:
        corr_data = load_correlation_matrix()
        print(f"✓ Loaded correlation matrix")
    except Exception as e:
        print(f"⚠️  Could not load correlation data: {e}")
        corr_data = None
    print()

    # Initialize tester
    tester = PrecomputedPnLSizingTester(trades_df, corr_data=corr_data)

    # Global PnL gate configs
    day_probe_gate = GlobalGateConfig(
        name="GlobalDayPositiveProbe",
        window_days=1,
        fail_mode="probe",
        probe_fraction=0.15,
        min_positive=1e-9,
    )
    two_day_block_gate = GlobalGateConfig(
        name="GlobalTwoDayPositiveBlock",
        window_days=2,
        fail_mode="block",
        min_positive=1e-9,
    )
    unprofit_shutdown_gate = GlobalGateConfig(
        name="UnprofitShutdown_Window2",
        window_days=2,
        fail_mode="block",
        min_positive=1e-9,
        use_strategy_pnl=True,
    )
    stock_dir_shutdown_gate = GlobalGateConfig(
        name="StockDirShutdown_Window2",
        window_days=2,
        window_trades=2,
        fail_mode="block",
        min_positive=1e-9,
        use_strategy_pnl=True,
        scope="symbol_side",
    )

    # Define strategies to test (add gating variants for Kelly + fixed fraction)
    strategies = [
        # Baseline
        (FixedFractionStrategy(0.5), "Naive_50pct_Baseline", None),

        # Fixed allocations
        (FixedFractionStrategy(0.25), "Fixed_25pct", None),
        (FixedFractionStrategy(0.75), "Fixed_75pct", None),
        (FixedFractionStrategy(1.0), "Fixed_100pct", None),

        # Kelly variants
        (KellyStrategy(fraction=0.25, cap=1.0), "Kelly_25pct", None),
        (KellyStrategy(fraction=0.5, cap=1.0), "Kelly_50pct", None),
        (KellyStrategy(fraction=0.75, cap=1.0), "Kelly_75pct", None),
        (KellyStrategy(fraction=1.0, cap=1.0), "Kelly_100pct", None),

        # Volatility-based
        (VolatilityTargetStrategy(target_vol=0.10), "VolTarget_10pct", None),
        (VolatilityTargetStrategy(target_vol=0.15), "VolTarget_15pct", None),
        (VolatilityTargetStrategy(target_vol=0.20), "VolTarget_20pct", None),

        # Volatility-adjusted
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.10), "VolAdjusted_10pct", None),
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.15), "VolAdjusted_15pct", None),

        # Correlation-aware
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=2.0, fractional_kelly=0.25), "CorrAware_Conservative", None),
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=1.0, fractional_kelly=0.5), "CorrAware_Moderate", None),
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=0.5, fractional_kelly=0.75), "CorrAware_Aggressive", None),

        # Global gating experiments
        (FixedFractionStrategy(0.5), "Naive_50pct_DayProbeGate", day_probe_gate),
        (FixedFractionStrategy(0.5), "Naive_50pct_TwoDayBlock", two_day_block_gate),
        (KellyStrategy(fraction=0.5, cap=1.0), "Kelly_50pct_DayProbeGate", day_probe_gate),
        (KellyStrategy(fraction=0.5, cap=1.0), "Kelly_50pct_TwoDayBlock", two_day_block_gate),

        # Unprofit shutdown experiments (dynamic strategy PnL window)
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.10), "VolAdjusted_10pct_UnprofitShutdown", unprofit_shutdown_gate),
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.15), "VolAdjusted_15pct_UnprofitShutdown", unprofit_shutdown_gate),
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=1.0, fractional_kelly=0.5), "CorrAware_Moderate_UnprofitShutdown", unprofit_shutdown_gate),

        # Stock-direction shutdown only
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.10), "VolAdjusted_10pct_StockDirShutdown", stock_dir_shutdown_gate),
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.15), "VolAdjusted_15pct_StockDirShutdown", stock_dir_shutdown_gate),

        # Combined account + stock-direction shutdown
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.10), "VolAdjusted_10pct_UnprofitShutdown_StockDirShutdown", [unprofit_shutdown_gate, stock_dir_shutdown_gate]),
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.15), "VolAdjusted_15pct_UnprofitShutdown_StockDirShutdown", [unprofit_shutdown_gate, stock_dir_shutdown_gate]),

        # Correlation-aware with symbol shutdown
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=1.0, fractional_kelly=0.5), "CorrAware_Moderate_StockDirShutdown", stock_dir_shutdown_gate),
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=1.0, fractional_kelly=0.5), "CorrAware_Moderate_UnprofitShutdown_StockDirShutdown", [unprofit_shutdown_gate, stock_dir_shutdown_gate]),
    ]

    # Run all strategies
    print("Running sizing strategies on precomputed trades...")
    print()

    results = []
    for strategy, name, gate_cfg in strategies:
        result = tester.run_strategy(strategy, name, gate_config=gate_cfg)
        results.append(result)
        if isinstance(gate_cfg, (list, tuple)):
            gate_label = "+".join(cfg.name for cfg in gate_cfg)
        elif gate_cfg:
            gate_label = gate_cfg.name
        else:
            gate_label = ""
        gate_note = f" (gate: {gate_label})" if gate_label else ""
        print(f"  ✓ Completed: {name}{gate_note}")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Convert to DataFrame
    df_data = []
    for r in results:
        df_data.append({
            'Strategy': r.strategy_name,
            'Total PnL': f"${r.total_pnl:,.0f}",
            'Return': f"{r.total_return_pct:.2%}",
            'Sharpe': f"{r.sharpe_ratio:.2f}",
            'Sortino': f"{r.sortino_ratio:.2f}",
            'Max DD': f"{r.max_drawdown_pct:.2%}",
            'Volatility': f"{r.volatility:.2%}",
            'Win Rate': f"{r.win_rate:.2%}",
            'Avg Size': f"{r.avg_position_size:.2%}",
            'Trades': r.num_trades,
            'Ann Return': f"{r.annualized_return_pct:.2%}",
            'Ann Gap vs 60%': f"{(r.annualized_return_pct - 0.60):+.2%}",
            'Gate Config': r.gate_config_name or '-',
            'Probe Days': r.gate_probe_days,
            'Blocked Days': r.gate_blocked_days,
            'Symbol Blocks': r.symbol_gate_blocks,
            'Symbol Probes': r.symbol_gate_probes,
        })

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    print()

    target_ann_return = 0.60
    target_hit = [r for r in results if r.annualized_return_pct >= target_ann_return]
    print(f"Strategies at/above {target_ann_return:.0%} annualized return: {len(target_hit)} / {len(results)}")
    if target_hit:
        print("  -> " + ", ".join(r.strategy_name for r in target_hit))
    print()

    # Persist per-day metrics + visualizations
    daily_metrics_df = build_daily_metrics_df(results)
    reports_dir = Path("strategytraining/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    daily_metrics_path = reports_dir / "sizing_strategy_daily_metrics.csv"
    daily_metrics_df.to_csv(daily_metrics_path, index=False)
    print(f"Saved daily metrics to: {daily_metrics_path}")

    curves_dir = reports_dir / "sizing_curves"
    generate_visualizations(results, daily_metrics_df, curves_dir)
    print(f"Saved curve plots under: {curves_dir}")

    # Rank by Sharpe
    print("Top 5 by Sharpe Ratio:")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x.sharpe_ratio, reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {r.strategy_name:30s} Sharpe: {r.sharpe_ratio:6.2f}  "
              f"Return: {r.total_return_pct:7.2%}  DD: {r.max_drawdown_pct:6.2%}")
    print()

    # Rank by return
    print("Top 5 by Total Return:")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x.total_return_pct, reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {r.strategy_name:30s} Return: {r.total_return_pct:7.2%}  "
              f"Sharpe: {r.sharpe_ratio:6.2f}  DD: {r.max_drawdown_pct:6.2%}")
    print()

    gated_results = [r for r in results if r.gate_config_name]
    if gated_results:
        print("Global PnL Gate Summary:")
        print("-" * 80)
        for r in gated_results:
            print(
                f"  {r.strategy_name:30s} gate={r.gate_config_name:>25s} "
                f"probe_days={r.gate_probe_days:3d} blocked_days={r.gate_blocked_days:3d}"
            )
        print()

    symbol_gated_results = [r for r in results if r.symbol_gate_blocks or r.symbol_gate_probes]
    if symbol_gated_results:
        print("Symbol/Direction Gate Summary:")
        print("-" * 80)
        for r in symbol_gated_results:
            gate_name = r.gate_config_name or 'symbol_scope'
            print(
                f"  {r.strategy_name:30s} gate={gate_name:>25s} "
                f"symbol_blocks={r.symbol_gate_blocks:4d} symbol_probes={r.symbol_gate_probes:4d}"
            )
        print()

    # Save results
    output_file = Path("strategytraining/sizing_strategy_fast_test_results.json")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset': str(trades_file.name),
        'num_trades': len(trades_df),
        'symbols': available_symbols,
        'results': [
            {
                'strategy': r.strategy_name,
                'total_pnl': r.total_pnl,
                'return_pct': r.total_return_pct,
                'sharpe': r.sharpe_ratio,
                'max_dd_pct': r.max_drawdown_pct,
                'volatility': r.volatility,
                'win_rate': r.win_rate,
                'avg_size': r.avg_position_size,
                'num_trades': r.num_trades,
                'gate_config': r.gate_config_name,
                'gate_probe_days': r.gate_probe_days,
                'gate_blocked_days': r.gate_blocked_days,
                'gate_normal_days': r.gate_normal_days,
                'daily_curve': r.daily_curve,
                'sortino_ratio': r.sortino_ratio,
                'annualized_return_pct': r.annualized_return_pct,
                'annualization_days': r.annualization_days,
                'symbol_gate_blocks': r.symbol_gate_blocks,
                'symbol_gate_probes': r.symbol_gate_probes,
            }
            for r in results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Compare to baseline
    baseline = next((r for r in results if 'Baseline' in r.strategy_name), None)
    if baseline:
        print("=" * 80)
        print("COMPARISON TO BASELINE")
        print("=" * 80)
        print(f"Baseline (Naive 50%): Return {baseline.total_return_pct:.2%}, Sharpe {baseline.sharpe_ratio:.2f}")
        print()

        # Show top improvements
        improvements = []
        for r in results:
            if r.strategy_name != baseline.strategy_name:
                return_delta = r.total_return_pct - baseline.total_return_pct
                sharpe_delta = r.sharpe_ratio - baseline.sharpe_ratio
                improvements.append((r, return_delta, sharpe_delta))

        # Sort by return improvement
        improvements.sort(key=lambda x: x[1], reverse=True)

        print("Top improvements vs baseline:")
        for i, (r, ret_delta, sharpe_delta) in enumerate(improvements[:5], 1):
            print(f"  {i}. {r.strategy_name:30s} "
                  f"Return: {r.total_return_pct:7.2%} ({ret_delta:+.2%})  "
                  f"Sharpe: {r.sharpe_ratio:6.2f} ({sharpe_delta:+.2f})")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
