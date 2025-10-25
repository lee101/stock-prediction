from __future__ import annotations

import importlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - dependency optional at import
    import matplotlib

    matplotlib.use("Agg")  # type: ignore[attr-defined]
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from marketsimulator.logging_utils import logger

from .environment import activate_simulation
from .state import SimulationState, TradeExecution


@dataclass
class DailySnapshot:
    day_index: int
    phase: str
    timestamp: datetime
    equity: float
    cash: float
    positions: Dict[str, float]


@dataclass
class SymbolPerformance:
    symbol: str
    cash_flow: float
    market_value: float
    position_qty: float
    unrealized_pl: float
    total_value: float
    trades: int
    realised_pl: float


@dataclass
class SimulationReport:
    initial_cash: float
    final_cash: float
    final_equity: float
    total_return: float
    total_return_pct: float
    fees_paid: float
    trades_executed: int
    max_drawdown: float
    max_drawdown_pct: float
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    symbol_performance: List[SymbolPerformance] = field(default_factory=list)
    generated_files: List[Path] = field(default_factory=list)
    trade_executions: List[TradeExecution] = field(default_factory=list)

    def render_summary(self) -> str:
        lines = [
            "Simulation Summary",
            f"  Initial cash : ${self.initial_cash:,.2f}",
            f"  Final equity : ${self.final_equity:,.2f}",
            f"  Total return : ${self.total_return:,.2f} ({self.total_return_pct:.2%})",
            f"  Fees paid    : ${self.fees_paid:,.2f}",
            f"  Trades       : {self.trades_executed}",
            f"  Max drawdown : ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2%})",
            "",
            "Per-symbol performance:",
        ]
        if not self.symbol_performance:
            lines.append("  (no trades executed)")
        else:
            lines.append(
                "  Symbol | Cash Flow | Market Value | Position | Unrealized P/L | Realized P/L | Total Value | Trades"
            )
            for perf in self.symbol_performance:
                lines.append(
                    f"  {perf.symbol:<6} | "
                    f"${perf.cash_flow:>10,.2f} | "
                    f"${perf.market_value:>12,.2f} | "
                    f"{perf.position_qty:>8.3f} | "
                    f"${perf.unrealized_pl:>12,.2f} | "
                    f"${perf.realised_pl:>12,.2f} | "
                    f"${perf.total_value:>11,.2f} | "
                    f"{perf.trades:>6}"
                )
        return "\n".join(lines)


def _symbol_performance(state: SimulationState) -> List[SymbolPerformance]:
    cash_by_symbol: Dict[str, float] = defaultdict(float)
    trades_by_symbol: Dict[str, List[TradeExecution]] = defaultdict(list)

    for execution in state.trade_log:
        cash_by_symbol[execution.symbol] += execution.cash_delta
        trades_by_symbol[execution.symbol].append(execution)

    symbols: Iterable[str] = set(cash_by_symbol.keys()) | set(state.positions.keys())
    performance: List[SymbolPerformance] = []
    for symbol in sorted(symbols):
        position = state.positions.get(symbol)
        market_value = float(position.market_value) if position else 0.0
        position_qty = float(position.qty) if position else 0.0
        unrealized = float(position.unrealized_pl) if position else 0.0
        cash_flow = cash_by_symbol.get(symbol, 0.0)
        total_value = cash_flow + market_value
        realised = sum(trade.cash_delta for trade in trades_by_symbol.get(symbol, []))
        performance.append(
            SymbolPerformance(
                symbol=symbol,
                cash_flow=cash_flow,
                market_value=market_value,
                position_qty=position_qty,
                unrealized_pl=unrealized,
                total_value=total_value,
                trades=len(trades_by_symbol.get(symbol, [])),
                realised_pl=realised,
            )
        )
    return performance


def _build_report(state: SimulationState, daily_snapshots: List[DailySnapshot], initial_cash: float) -> SimulationReport:
    final_cash = float(state.cash)
    final_equity = float(state.equity)
    total_return = final_equity - initial_cash
    total_return_pct = (total_return / initial_cash) if initial_cash else 0.0
    max_dd_value, max_dd_pct = _compute_max_drawdown(daily_snapshots, initial_cash, final_equity)

    return SimulationReport(
        initial_cash=initial_cash,
        final_cash=final_cash,
        final_equity=final_equity,
        total_return=total_return,
        total_return_pct=total_return_pct,
        fees_paid=float(state.fees_paid),
        trades_executed=len(state.trade_log),
        max_drawdown=max_dd_value,
        max_drawdown_pct=max_dd_pct,
        daily_snapshots=daily_snapshots,
        symbol_performance=_symbol_performance(state),
        trade_executions=list(state.trade_log),
    )


def _compute_max_drawdown(
    snapshots: List[DailySnapshot], initial_cash: float, final_equity: float
) -> Tuple[float, float]:
    if not snapshots:
        equity_series = [initial_cash, final_equity]
    else:
        equity_series = [initial_cash]
        equity_series.extend(snap.equity for snap in snapshots)
    peak = equity_series[0]
    max_drawdown = 0.0
    for value in equity_series[1:]:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    max_drawdown_pct = (max_drawdown / peak) if peak else 0.0
    return max_drawdown, max_drawdown_pct


def _generate_plots(report: SimulationReport, output_dir: Path) -> List[Path]:
    if plt is None:
        logger.warning("matplotlib is unavailable; skipping plot generation")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    # Equity curve across the full simulation
    if report.daily_snapshots:
        sorted_snaps = sorted(report.daily_snapshots, key=lambda snap: snap.timestamp)
        times = [snap.timestamp for snap in sorted_snaps]
        equities = [snap.equity for snap in sorted_snaps]
        cashes = [snap.cash for snap in sorted_snaps]

        plt.figure(figsize=(9, 4))
        plt.plot(times, equities, label="Equity", marker="o")
        plt.plot(times, cashes, label="Cash", linestyle="--", marker="x", alpha=0.7)
        plt.xlabel("Timestamp")
        plt.ylabel("Value ($)")
        plt.title("Equity and Cash Over Simulation")
        plt.legend()
        plt.tight_layout()
        equity_path = output_dir / "equity_curve.png"
        plt.savefig(equity_path)
        plt.close()
        generated.append(equity_path)

        # Per-day snapshots
        grouped: Dict[int, List[DailySnapshot]] = defaultdict(list)
        for snap in sorted_snaps:
            grouped[snap.day_index].append(snap)
        for day_idx, snaps in grouped.items():
            day_path = output_dir / f"day_{day_idx + 1:02d}_equity.png"
            day_times = [snap.timestamp for snap in snaps]
            day_equities = [snap.equity for snap in snaps]
            plt.figure(figsize=(6, 3))
            plt.plot(day_times, day_equities, marker="o")
            plt.title(f"Equity Day {day_idx + 1}")
            plt.xlabel("Timestamp")
            plt.ylabel("Equity ($)")
            plt.tight_layout()
            plt.savefig(day_path)
            plt.close()
            generated.append(day_path)

    # Symbol contribution bar chart
    if report.symbol_performance:
        symbols = [perf.symbol for perf in report.symbol_performance]
        totals = [perf.total_value for perf in report.symbol_performance]
        plt.figure(figsize=(max(6, len(symbols) * 0.75), 4))
        plt.bar(symbols, totals)
        plt.ylabel("Contribution ($)")
        plt.title("Per-Symbol Net Contribution")
        plt.tight_layout()
        contrib_path = output_dir / "symbol_contributions.png"
        plt.savefig(contrib_path)
        plt.close()
        generated.append(contrib_path)

    return generated


def simulate_strategy(
    symbols: Sequence[str],
    days: int = 5,
    step_size: int = 24,
    initial_cash: float = 100_000.0,
    top_k: int = 4,
    output_dir: Optional[Path] = None,
    force_kronos: Optional[bool] = None,
    flatten_end: bool = False,
) -> SimulationReport:
    daily_snapshots: List[DailySnapshot] = []
    symbols = list(symbols)
    output_path = Path(output_dir) if output_dir else None

    with activate_simulation(
        symbols=symbols,
        initial_cash=initial_cash,
        force_kronos=force_kronos,
    ) as controller:
        trade_module = importlib.import_module("trade_stock_e2e")
        predict_module = importlib.import_module("predict_stock_forecasting")
        alpaca_module = importlib.import_module("alpaca_wrapper")
        previous_picks: Dict[str, Dict] = {}

        if hasattr(trade_module, "reset_symbol_entry_counters"):
            trade_module.reset_symbol_entry_counters(run_id="simulator")

        mid_steps = max(1, step_size // 2)
        end_steps = max(1, step_size - mid_steps)

        for day in range(days):
            current_time = controller.current_time()
            logger.info(f"[sim] Trading day {day + 1}/{days} @ {current_time}")

            try:
                forecasts = predict_module.make_predictions(
                    input_data_path="_simulator",
                    pred_name=f"SIM-DAY-{day + 1}",
                    retrain=False,
                    alpaca_wrapper=alpaca_module,
                )
                if forecasts is not None:
                    logger.info(f"[sim] Generated {len(forecasts)} forecasts for day {day + 1}")
            except Exception as exc:  # pragma: no cover - depends on external deps
                logger.warning(f"[sim] Forecast generation failed: {exc}")

            analyzed = trade_module.analyze_symbols(symbols)
            portfolio = trade_module.build_portfolio(
                analyzed,
                min_positions=trade_module.DEFAULT_MIN_CORE_POSITIONS,
                max_positions=max(top_k, trade_module.DEFAULT_MIN_CORE_POSITIONS),
                max_expanded=max(top_k, trade_module.EXPANDED_PORTFOLIO),
            )
            if portfolio:
                trade_module.log_trading_plan(portfolio, f"SIM-DAY-{day + 1}-OPEN")
            trade_module.manage_positions(portfolio, previous_picks, analyzed)

            open_summary = controller.summary()
            daily_snapshots.append(
                DailySnapshot(
                    day_index=day,
                    phase="open",
                    timestamp=current_time,
                    equity=open_summary["equity"],
                    cash=open_summary["cash"],
                    positions=open_summary["positions"],
                )
            )

            controller.advance_steps(mid_steps)

            close_picks = trade_module.manage_market_close(symbols, portfolio, analyzed)
            close_summary = controller.summary()
            daily_snapshots.append(
                DailySnapshot(
                    day_index=day,
                    phase="close",
                    timestamp=controller.current_time(),
                    equity=close_summary["equity"],
                    cash=close_summary["cash"],
                    positions=close_summary["positions"],
                )
            )
            previous_picks = close_picks

            controller.advance_steps(end_steps)

        state = controller.state
        if flatten_end:
            _flatten_positions(controller)
            state = controller.state
        report = _build_report(state, daily_snapshots, initial_cash)
        logger.info("\n" + report.render_summary())
        if output_path:
            generated = _generate_plots(report, output_path)
            report.generated_files.extend(generated)
        return report


def _flatten_positions(controller) -> None:
    state = controller.state
    if not state.positions:
        logger.info("[sim] No positions to flatten at end of run.")
        return
    logger.info(f"[sim] Flattening {len(state.positions)} open positions at end of run.")
    for symbol, position in list(state.positions.items()):
        qty = float(position.qty)
        if abs(qty) < 1e-9:
            continue
        price = float(position.current_price)
        state.close_position(symbol, price=price, qty=abs(qty))
    logger.info(f"[sim] End-of-run flatten complete. Cash: {state.cash:.2f} Equity: {state.equity:.2f}")
