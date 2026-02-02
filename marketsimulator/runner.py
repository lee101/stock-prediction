from __future__ import annotations

import importlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - dependency optional at import
    import matplotlib

    matplotlib.use("Agg")  # type: ignore[attr-defined]
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from marketsimulator.logging_utils import logger

from src.fixtures import crypto_symbols

from .data_feed import DEFAULT_DATA_ROOT
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
    positions_detail: Dict[str, Dict[str, float]] = field(default_factory=dict)


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
    trading_fees_paid: float
    financing_cost_paid: float
    trades_executed: int
    max_drawdown: float
    max_drawdown_pct: float
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    symbol_performance: List[SymbolPerformance] = field(default_factory=list)
    generated_files: List[Path] = field(default_factory=list)
    trade_executions: List[TradeExecution] = field(default_factory=list)
    symbol_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    price_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    daily_analysis: List[Dict[str, Any]] = field(default_factory=list)

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


def _build_report(
    state: SimulationState,
    daily_snapshots: List[DailySnapshot],
    initial_cash: float,
    *,
    symbol_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    price_history: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    daily_analysis: Optional[List[Dict[str, Any]]] = None,
) -> SimulationReport:
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
        trading_fees_paid=max(0.0, float(state.fees_paid) - float(state.financing_cost_paid)),
        financing_cost_paid=float(state.financing_cost_paid),
        trades_executed=len(state.trade_log),
        max_drawdown=max_dd_value,
        max_drawdown_pct=max_dd_pct,
        daily_snapshots=daily_snapshots,
        symbol_performance=_symbol_performance(state),
        trade_executions=list(state.trade_log),
        symbol_metadata=dict(symbol_metadata or {}),
        price_history=dict(price_history or {}),
        daily_analysis=list(daily_analysis or []),
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


def _resolve_data_root() -> Path:
    env_value = os.getenv("MARKETSIM_DATA_ROOT", "").strip()
    if not env_value:
        return DEFAULT_DATA_ROOT

    candidate = Path(env_value).expanduser()
    if not candidate.exists():
        logger.warning(
            "[sim] MARKETSIM_DATA_ROOT=%s does not exist; falling back to %s",
            candidate,
            DEFAULT_DATA_ROOT,
        )
        return DEFAULT_DATA_ROOT

    return candidate


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
    symbol_metadata: Dict[str, Dict[str, Any]] = {}
    price_history: Dict[str, List[Dict[str, Any]]] = {}
    daily_analysis: List[Dict[str, Any]] = []

    data_root = _resolve_data_root()

    with activate_simulation(
        symbols=symbols,
        initial_cash=initial_cash,
        data_root=data_root,
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

            forecasts = None
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

            for symbol, data in analyzed.items():
                symbol_str = str(symbol)
                symbol_upper = symbol_str.upper()
                meta = symbol_metadata.setdefault(symbol_str, {})
                meta["symbol"] = symbol_str
                strategy = data.get("strategy")
                if strategy:
                    meta["strategy"] = strategy
                trade_mode = data.get("trade_mode")
                if trade_mode:
                    meta["trade_mode"] = trade_mode
                side = data.get("side")
                if side:
                    meta["side"] = side
                meta["asset_class"] = "crypto" if symbol_upper in crypto_symbols else "equity"
                meta["is_crypto"] = symbol_upper in crypto_symbols

            if portfolio:
                trade_module.log_trading_plan(portfolio, f"SIM-DAY-{day + 1}-OPEN")
            trade_module.manage_positions(portfolio, previous_picks, analyzed)

            open_summary = controller.summary()
            open_positions_detail = {
                symbol: dict(details) for symbol, details in open_summary.get("positions_detail", {}).items()
            }
            daily_snapshots.append(
                DailySnapshot(
                    day_index=day,
                    phase="open",
                    timestamp=current_time,
                    equity=open_summary["equity"],
                    cash=open_summary["cash"],
                    positions=open_summary["positions"],
                    positions_detail=open_positions_detail,
                )
            )

            controller.advance_steps(mid_steps)

            close_picks = trade_module.manage_market_close(symbols, portfolio, analyzed)
            close_summary = controller.summary()
            close_positions_detail = {
                symbol: dict(details) for symbol, details in close_summary.get("positions_detail", {}).items()
            }
            daily_snapshots.append(
                DailySnapshot(
                    day_index=day,
                    phase="close",
                    timestamp=controller.current_time(),
                    equity=close_summary["equity"],
                    cash=close_summary["cash"],
                    positions=close_summary["positions"],
                    positions_detail=close_positions_detail,
                )
            )
            previous_picks = close_picks

            controller.advance_steps(end_steps)
            strategy_counts: Dict[str, int] = defaultdict(int)
            trade_mode_counts: Dict[str, int] = defaultdict(int)
            for symbol, data in portfolio.items():
                strategy = data.get("strategy") or "unknown"
                strategy_counts[strategy] += 1
                mode = data.get("trade_mode") or symbol_metadata.get(symbol, {}).get("trade_mode") or "normal"
                trade_mode_counts[mode] += 1
            probe_candidates = sum(1 for data in analyzed.values() if data.get("trade_mode") == "probe")
            blocked_candidates = sum(1 for data in analyzed.values() if data.get("trade_blocked"))
            daily_analysis.append(
                {
                    "day_index": day,
                    "timestamp": current_time.isoformat(),
                    "symbols_analyzed": len(analyzed),
                    "portfolio_size": len(portfolio),
                    "forecasts_generated": len(forecasts) if forecasts is not None else 0,
                    "strategy_counts": dict(strategy_counts),
                    "trade_mode_counts": dict(trade_mode_counts),
                    "probe_candidates": probe_candidates,
                    "blocked_candidates": blocked_candidates,
                }
            )

        sim_state = controller.state
        for symbol, series in sim_state.prices.items():
            frame = series.frame.iloc[: series.cursor + 1]
            symbol_history: List[Dict[str, Any]] = []
            for _, row in frame.iterrows():
                raw_ts = row.get("timestamp")
                if hasattr(raw_ts, "isoformat"):
                    ts_value = raw_ts.isoformat()
                else:
                    ts_value = str(raw_ts)
                symbol_history.append(
                    {
                        "timestamp": ts_value,
                        "close": row.get("Close"),
                        "open": row.get("Open"),
                        "high": row.get("High"),
                        "low": row.get("Low"),
                        "volume": row.get("Volume"),
                    }
                )
            price_history[symbol] = symbol_history
        for position_symbol in sim_state.positions.keys():
            if position_symbol in symbol_metadata:
                continue
            symbol_upper = position_symbol.upper()
            symbol_metadata[position_symbol] = {
                "symbol": position_symbol,
                "asset_class": "crypto" if symbol_upper in crypto_symbols else "equity",
                "is_crypto": symbol_upper in crypto_symbols,
            }

        state = controller.state
        if flatten_end:
            _flatten_positions(controller)
            state = controller.state
        report = _build_report(
            state,
            daily_snapshots,
            initial_cash,
            symbol_metadata=symbol_metadata,
            price_history=price_history,
            daily_analysis=daily_analysis,
        )
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
