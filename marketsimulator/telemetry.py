from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .runner import DailySnapshot, SimulationReport, SymbolPerformance

ScalarSummary = Dict[str, float]


def _initial_summary() -> ScalarSummary:
    return {
        "realised_pnl": 0.0,
        "cash_flow": 0.0,
        "market_value": 0.0,
        "total_value": 0.0,
        "unrealized_pl": 0.0,
        "trades": 0.0,
    }


def compute_equity_timeseries(report: SimulationReport) -> List[Dict[str, Any]]:
    """Derive a per-day equity curve from simulation snapshots."""
    if not report.daily_snapshots:
        return []

    closes: Dict[int, DailySnapshot] = {}
    for snapshot in sorted(report.daily_snapshots, key=lambda snap: (snap.day_index, snap.phase, snap.timestamp)):
        if snapshot.phase == "close" or snapshot.day_index not in closes:
            closes[snapshot.day_index] = snapshot

    equity_curve: List[Dict[str, Any]] = []
    previous_equity = float(report.initial_cash)

    for day_index in sorted(closes):
        snapshot = closes[day_index]
        equity = float(snapshot.equity)
        cash = float(snapshot.cash)
        denom = previous_equity if abs(previous_equity) > 1e-9 else None
        daily_return = ((equity - previous_equity) / denom) if denom else 0.0
        initial = float(report.initial_cash)
        cumulative = ((equity - initial) / initial) if abs(initial) > 1e-9 else 0.0
        open_positions = sum(1 for qty in snapshot.positions.values() if abs(float(qty)) > 1e-9)
        gross_qty = sum(abs(float(qty)) for qty in snapshot.positions.values())
        equity_curve.append(
            {
                "day_index": day_index,
                "timestamp": snapshot.timestamp,
                "equity": equity,
                "cash": cash,
                "daily_return": daily_return,
                "cumulative_return": cumulative,
                "open_positions": float(open_positions),
                "gross_position_qty": float(gross_qty),
            }
        )
        previous_equity = equity

    return equity_curve


def _summarise_performance(
    performances: Iterable[SymbolPerformance],
    metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, ScalarSummary]]:
    asset_summary: MutableMapping[str, ScalarSummary] = defaultdict(_initial_summary)
    mode_summary: MutableMapping[str, ScalarSummary] = defaultdict(_initial_summary)
    strategy_summary: MutableMapping[str, ScalarSummary] = defaultdict(_initial_summary)

    for perf in performances:
        symbol_meta = metadata.get(perf.symbol, {})
        asset_key = str(symbol_meta.get("asset_class") or ("crypto" if perf.symbol.upper().endswith("USD") else "equity"))
        mode_key = str(symbol_meta.get("trade_mode") or "unknown")
        strategy_key = str(symbol_meta.get("strategy") or "unassigned")

        for target in (asset_summary[asset_key], mode_summary[mode_key], strategy_summary[strategy_key]):
            target["realised_pnl"] += float(perf.realised_pl)
            target["cash_flow"] += float(perf.cash_flow)
            target["market_value"] += float(perf.market_value)
            target["unrealized_pl"] += float(perf.unrealized_pl)
            target["total_value"] += float(perf.total_value)
            target["trades"] += float(perf.trades)

    return {
        "asset": dict(asset_summary),
        "trade_mode": dict(mode_summary),
        "strategy": dict(strategy_summary),
    }


def compute_breakdowns(report: SimulationReport) -> Dict[str, Dict[str, ScalarSummary]]:
    """Aggregate realised PnL and positioning by asset class, trade mode, and strategy."""
    return _summarise_performance(report.symbol_performance, report.symbol_metadata)


def build_symbol_performance_table(report: SimulationReport) -> Tuple[Sequence[str], List[Sequence[Any]]]:
    """Prepare a tabular view of per-symbol results for logging dashboards."""
    columns: List[str] = [
        "symbol",
        "trades",
        "cash_flow",
        "market_value",
        "unrealized_pl",
        "realised_pl",
        "total_value",
        "strategy",
        "trade_mode",
        "asset_class",
    ]
    rows: List[Sequence[Any]] = []

    for perf in sorted(report.symbol_performance, key=lambda p: p.symbol):
        symbol_meta = report.symbol_metadata.get(perf.symbol, {})
        rows.append(
            [
                perf.symbol,
                float(perf.trades),
                float(perf.cash_flow),
                float(perf.market_value),
                float(perf.unrealized_pl),
                float(perf.realised_pl),
                float(perf.total_value),
                symbol_meta.get("strategy", "unassigned"),
                symbol_meta.get("trade_mode", "unknown"),
                symbol_meta.get("asset_class", "equity"),
            ]
        )

    return columns, rows


def compute_risk_timeseries(report: SimulationReport) -> List[Dict[str, Any]]:
    """Compute leverage, exposure, and drawdown metrics for each recorded snapshot."""
    if not report.daily_snapshots:
        return []

    risk_series: List[Dict[str, Any]] = []
    peak_equity = float(report.initial_cash)

    for snapshot in sorted(
        report.daily_snapshots, key=lambda snap: (snap.timestamp, snap.day_index, snap.phase)
    ):
        positions_detail = snapshot.positions_detail or {}
        gross_exposure = sum(abs(float(details.get("market_value", 0.0))) for details in positions_detail.values())
        net_exposure = sum(float(details.get("market_value", 0.0)) for details in positions_detail.values())
        equity = float(snapshot.equity)
        cash = float(snapshot.cash)
        peak_equity = max(peak_equity, equity)
        drawdown = max(0.0, peak_equity - equity)
        drawdown_pct = drawdown / peak_equity if peak_equity > 1e-9 else 0.0
        leverage = gross_exposure / equity if abs(equity) > 1e-9 else 0.0

        risk_series.append(
            {
                "timestamp": snapshot.timestamp,
                "day_index": snapshot.day_index,
                "phase": snapshot.phase,
                "equity": equity,
                "cash": cash,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "leverage": leverage,
                "drawdown": drawdown,
                "drawdown_pct": drawdown_pct,
            }
        )
    return risk_series


def compute_fee_breakdown(report: SimulationReport) -> Dict[str, float]:
    """Return financing vs. trading fee totals."""
    return {
        "fees/total": float(report.fees_paid),
        "fees/trading": float(report.trading_fees_paid),
        "fees/financing": float(report.financing_cost_paid),
    }


def build_portfolio_stack_series(report: SimulationReport) -> Tuple[Sequence[str], List[Sequence[Any]]]:
    """Return per-symbol market values per snapshot suitable for stacked visualisations."""
    columns = ["timestamp", "day_index", "phase", "symbol", "market_value", "abs_market_value"]
    rows: List[Sequence[Any]] = []
    for snapshot in sorted(
        report.daily_snapshots, key=lambda snap: (snap.timestamp, snap.day_index, snap.phase)
    ):
        timestamp = snapshot.timestamp.isoformat() if hasattr(snapshot.timestamp, "isoformat") else snapshot.timestamp
        for symbol, details in (snapshot.positions_detail or {}).items():
            market_value = float(details.get("market_value", 0.0))
            rows.append(
                [
                    timestamp,
                    snapshot.day_index,
                    snapshot.phase,
                    symbol,
                    market_value,
                    abs(market_value),
                ]
            )
        if not snapshot.positions_detail:
            rows.append([timestamp, snapshot.day_index, snapshot.phase, "__cash__", 0.0, 0.0])
    return columns, rows


def build_trade_events_table(report: SimulationReport) -> Tuple[Sequence[str], List[Sequence[Any]]]:
    """Tabular summary of trade executions for scatter charts."""
    columns = ["timestamp", "symbol", "side", "qty", "price", "notional", "fee", "cash_delta", "slip_bps"]
    rows: List[Sequence[Any]] = []
    for trade in report.trade_executions:
        rows.append(
            [
                trade.timestamp.isoformat() if hasattr(trade.timestamp, "isoformat") else trade.timestamp,
                trade.symbol,
                trade.side,
                float(trade.qty),
                float(trade.price),
                float(trade.notional),
                float(trade.fee),
                float(trade.cash_delta),
                float(trade.slip_bps),
            ]
        )
    return columns, rows


def build_price_history_table(report: SimulationReport) -> Tuple[Sequence[str], List[Sequence[Any]]]:
    """Flatten price history for logging or external plotting."""
    columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    rows: List[Sequence[Any]] = []
    for symbol, history in report.price_history.items():
        for entry in history:
            rows.append(
                [
                    symbol,
                    entry.get("timestamp"),
                    entry.get("open"),
                    entry.get("high"),
                    entry.get("low"),
                    entry.get("close"),
                    entry.get("volume"),
                ]
            )
    return columns, rows


def summarize_daily_analysis(report: SimulationReport) -> Dict[str, Any]:
    """Aggregate daily analysis signals recorded during simulation."""
    if not getattr(report, "daily_analysis", None):
        return {}
    total_days = len(report.daily_analysis)
    aggregate: Dict[str, Any] = {
        "days_recorded": total_days,
        "avg_symbols_analyzed": 0.0,
        "avg_portfolio_size": 0.0,
        "avg_forecasts": 0.0,
        "probe_candidates": 0.0,
        "blocked_candidates": 0.0,
    }
    strategy_counts: MutableMapping[str, float] = defaultdict(float)
    mode_counts: MutableMapping[str, float] = defaultdict(float)

    for record in report.daily_analysis:
        aggregate["avg_symbols_analyzed"] += record.get("symbols_analyzed", 0.0)
        aggregate["avg_portfolio_size"] += record.get("portfolio_size", 0.0)
        aggregate["avg_forecasts"] += record.get("forecasts_generated", 0.0)
        aggregate["probe_candidates"] += record.get("probe_candidates", 0.0)
        aggregate["blocked_candidates"] += record.get("blocked_candidates", 0.0)
        for strategy, count in record.get("strategy_counts", {}).items():
            strategy_counts[strategy] += count
        for mode, count in record.get("trade_mode_counts", {}).items():
            mode_counts[mode] += count

    if total_days > 0:
        aggregate["avg_symbols_analyzed"] /= total_days
        aggregate["avg_portfolio_size"] /= total_days
        aggregate["avg_forecasts"] /= total_days
        aggregate["probe_candidates"] /= total_days
        aggregate["blocked_candidates"] /= total_days
    aggregate["strategy_counts"] = dict(strategy_counts)
    aggregate["trade_mode_counts"] = dict(mode_counts)
    return aggregate
