"""
Utility for extracting baseline PnL benchmarks from production logs and DeepSeek agent simulations.

Outputs JSON and Markdown summaries into evaltests/ for downstream comparison against RL runs.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import alpaca_wrapper as _alpaca_wrapper  # type: ignore  # noqa: WPS433
except Exception:
    _alpaca_wrapper = None  # type: ignore[assignment]
else:
    if hasattr(_alpaca_wrapper, "get_all_positions"):
        _alpaca_wrapper.get_all_positions = lambda: []  # type: ignore[assignment]
    if hasattr(_alpaca_wrapper, "get_account"):
        _alpaca_wrapper.get_account = lambda: SimpleNamespace(  # type: ignore[assignment]
            equity=10_000.0,
            cash=8_000.0,
            buying_power=12_000.0,
            multiplier=1.0,
        )
    if hasattr(_alpaca_wrapper, "get_clock"):
        _alpaca_wrapper.get_clock = lambda: SimpleNamespace(  # type: ignore[assignment]
            is_open=True,
            next_open=None,
            next_close=None,
        )
    if hasattr(_alpaca_wrapper, "re_setup_vars"):
        _alpaca_wrapper.re_setup_vars = lambda *_, **__: None  # type: ignore[assignment]

from deepseek_wrapper import call_deepseek_chat  # type: ignore
from stockagent.agentsimulator.data_models import AccountPosition, AccountSnapshot, TradingPlan
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentdeepseek.agent import simulate_deepseek_plan
from stockagentdeepseek_entrytakeprofit.agent import simulate_deepseek_entry_takeprofit_plan
from stockagentdeepseek_maxdiff.agent import simulate_deepseek_maxdiff_plan
from stockagentdeepseek_neural.agent import simulate_deepseek_neural_plan
from stockagentdeepseek_neural.forecaster import ModelForecastSummary, NeuralForecast

TRADE_HISTORY_PATH = REPO_ROOT / "strategy_state" / "trade_history.json"
TRADE_LOG_PATH = REPO_ROOT / "trade_stock_e2e.log"
OUTPUT_JSON = REPO_ROOT / "evaltests" / "baseline_pnl_summary.json"
OUTPUT_MARKDOWN = REPO_ROOT / "evaltests" / "baseline_pnl_summary.md"

SNAPSHOT_PATTERN = re.compile(
    r"\|\s+Portfolio snapshot recorded: value=\$(?P<value>-?\d+(?:\.\d+)?), "
    r"global risk threshold=(?P<risk>-?\d+(?:\.\d+)?)x"
)


def _parse_iso_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_trade_history(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def summarise_trade_history(history: Mapping[str, Sequence[Mapping[str, object]]]) -> dict:
    total_trades = 0
    total_pnl = 0.0
    by_symbol: MutableMapping[str, float] = defaultdict(float)
    by_date: MutableMapping[str, float] = defaultdict(float)
    realized: List[Tuple[datetime, float]] = []

    for key, entries in history.items():
        symbol_hint = key.split("|", 1)[0] if isinstance(key, str) else None
        for entry in entries or []:
            if not isinstance(entry, Mapping):
                continue
            pnl = float(entry.get("pnl", 0.0) or 0.0)
            total_trades += 1
            total_pnl += pnl

            symbol = entry.get("symbol")
            if not isinstance(symbol, str):
                symbol = symbol_hint
            if isinstance(symbol, str):
                by_symbol[symbol.upper()] += pnl

            closed_at = entry.get("closed_at")
            if isinstance(closed_at, str):
                try:
                    closed_dt = _parse_iso_datetime(closed_at)
                except ValueError:
                    continue
                trade_date = closed_dt.date().isoformat()
                by_date[trade_date] += pnl
                realized.append((closed_dt, pnl))

    realized.sort(key=lambda item: item[0])
    cumulative_curve: List[Tuple[str, float]] = []
    running = 0.0
    for closed_dt, pnl in realized:
        running += pnl
        cumulative_curve.append((closed_dt.isoformat(), running))

    return {
        "total_trades": total_trades,
        "total_realized_pnl": total_pnl,
        "pnl_by_symbol": dict(sorted(by_symbol.items())),
        "pnl_by_date": dict(sorted(by_date.items())),
        "cumulative_curve": cumulative_curve,
    }


def summarise_trade_log(path: Path) -> dict:
    if not path.exists():
        return {"snapshots": {"count": 0}}

    exposures: List[float] = []
    thresholds: List[float] = []
    timestamps: List[datetime] = []

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = SNAPSHOT_PATTERN.search(line)
            if not match:
                continue
            value = float(match.group("value"))
            risk = float(match.group("risk"))
            exposures.append(value)
            thresholds.append(risk)
            try:
                timestamp = datetime.fromisoformat(line[:19])
            except ValueError:
                continue
            timestamps.append(timestamp)

    if not exposures:
        return {"snapshots": {"count": 0}}

    first_ts = timestamps[0] if timestamps else None
    last_ts = timestamps[-1] if timestamps else None
    duration_days = None
    if first_ts and last_ts:
        duration_days = (last_ts - first_ts).total_seconds() / 86400.0

    return {
        "snapshots": {
            "count": len(exposures),
            "min_exposure": min(exposures),
            "max_exposure": max(exposures),
            "avg_exposure": sum(exposures) / len(exposures),
            "latest_exposure": exposures[-1],
            "latest_threshold": thresholds[-1],
            "duration_days": duration_days,
            "start_timestamp": first_ts.isoformat() if first_ts else None,
            "end_timestamp": last_ts.isoformat() if last_ts else None,
        }
    }


@contextmanager
def patched_deepseek_response(payload: Mapping[str, object]) -> Iterator[None]:
    raw_text = json.dumps(payload)

    def _fake_call(*_: object, **__: object) -> str:
        return raw_text

    original = call_deepseek_chat
    try:
        globals_ns = globals()
        globals_ns["call_deepseek_chat"] = _fake_call  # keep module attribute consistent
        import deepseek_wrapper as deepseek_module  # noqa: WPS433  (module import inside function)
        import stockagentdeepseek.agent as deepseek_agent  # noqa: WPS433
        import stockagentdeepseek_neural.agent as deepseek_neural  # noqa: WPS433

        deepseek_module.call_deepseek_chat = _fake_call  # type: ignore[attr-defined]
        deepseek_agent.call_deepseek_chat = _fake_call  # type: ignore[attr-defined]
        deepseek_neural.call_deepseek_chat = _fake_call  # type: ignore[attr-defined]
        yield
    finally:
        globals()["call_deepseek_chat"] = original
        try:
            import deepseek_wrapper as deepseek_module  # noqa: WPS433
            import stockagentdeepseek.agent as deepseek_agent  # noqa: WPS433
            import stockagentdeepseek_neural.agent as deepseek_neural  # noqa: WPS433

            deepseek_module.call_deepseek_chat = original  # type: ignore[attr-defined]
            deepseek_agent.call_deepseek_chat = original  # type: ignore[attr-defined]
            deepseek_neural.call_deepseek_chat = original  # type: ignore[attr-defined]
        except Exception:
            pass


@contextmanager
def offline_alpaca_state() -> Iterator[None]:
    try:
        import alpaca_wrapper as alp  # noqa: WPS433
    except Exception:
        yield
        return

    original_positions = getattr(alp, "get_all_positions", None)
    original_account = getattr(alp, "get_account", None)
    original_clock = getattr(alp, "get_clock", None)

    def _fake_positions() -> list:
        return []

    def _fake_account() -> SimpleNamespace:
        return SimpleNamespace(
            equity=10_000.0,
            cash=8_000.0,
            buying_power=12_000.0,
            multiplier=1.0,
        )

    def _fake_clock() -> SimpleNamespace:
        return SimpleNamespace(is_open=True, next_open=None, next_close=None)

    try:
        if original_positions is not None:
            alp.get_all_positions = _fake_positions  # type: ignore[assignment]
        if original_account is not None:
            alp.get_account = _fake_account  # type: ignore[assignment]
        if original_clock is not None:
            alp.get_clock = _fake_clock  # type: ignore[assignment]
        yield
    finally:
        if original_positions is not None:
            alp.get_all_positions = original_positions  # type: ignore[assignment]
        if original_account is not None:
            alp.get_account = original_account  # type: ignore[assignment]
        if original_clock is not None:
            alp.get_clock = original_clock  # type: ignore[assignment]


def _build_sample_market_bundle() -> MarketDataBundle:
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [110.0, 112.0, 111.0],
            "close": [112.0, 113.5, 114.0],
            "high": [112.0, 114.0, 115.0],
            "low": [109.0, 110.5, 110.0],
        },
        index=index,
    )
    return MarketDataBundle(
        bars={"AAPL": frame},
        lookback_days=3,
        as_of=index[-1].to_pydatetime(),
    )


def _build_account_snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        equity=10_000.0,
        cash=8_000.0,
        buying_power=12_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[
            AccountPosition(
                symbol="AAPL",
                quantity=0.0,
                side="flat",
                market_value=0.0,
                avg_entry_price=0.0,
                unrealized_pl=0.0,
                unrealized_plpc=0.0,
            )
        ],
    )


def _sample_plan_payload() -> dict[str, object]:
    return {
        "target_date": "2025-01-02",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 5,
                "execution_session": "market_open",
                "entry_price": 110.0,
                "exit_price": 114.0,
                "exit_reason": "initial position",
                "notes": "increase exposure",
            },
            {
                "symbol": "AAPL",
                "action": "sell",
                "quantity": 5,
                "execution_session": "market_close",
                "entry_price": 110.0,
                "exit_price": 114.0,
                "exit_reason": "close for profit",
                "notes": "close position",
            },
        ],
        "risk_notes": "Focus on momentum while keeping exposure bounded.",
        "focus_symbols": ["AAPL"],
        "stop_trading_symbols": [],
        "execution_window": "market_open",
        "metadata": {"capital_allocation_plan": "Allocate 100% to AAPL for the session."},
    }


def _build_neural_forecasts(symbols: Iterable[str]) -> Dict[str, NeuralForecast]:
    forecasts: Dict[str, NeuralForecast] = {}
    summary = ModelForecastSummary(
        model="manual_toto",
        config_name="baseline",
        average_price_mae=1.25,
        forecasts={"next_close": 114.0, "expected_return": 0.035},
    )
    for symbol in symbols:
        forecasts[symbol] = NeuralForecast(
            symbol=symbol,
            combined={"next_close": 114.0, "expected_return": 0.035},
            best_model="manual_toto",
            selection_source="baseline_script",
            model_summaries={"manual_toto": summary},
        )
    return forecasts


def run_deepseek_benchmarks() -> dict:
    plan_payload = _sample_plan_payload()
    bundle = _build_sample_market_bundle()
    snapshot = _build_account_snapshot()
    target_date = date(2025, 1, 2)

    results: dict[str, object] = {}

    with patched_deepseek_response(plan_payload), offline_alpaca_state():
        base = simulate_deepseek_plan(
            market_data=bundle,
            account_snapshot=snapshot,
            target_date=target_date,
        )
        entry_tp = simulate_deepseek_entry_takeprofit_plan(
            market_data=bundle,
            account_snapshot=snapshot,
            target_date=target_date,
        )
        maxdiff = simulate_deepseek_maxdiff_plan(
            market_data=bundle,
            account_snapshot=snapshot,
            target_date=target_date,
        )
        neural = simulate_deepseek_neural_plan(
            market_data=bundle,
            account_snapshot=snapshot,
            target_date=target_date,
            forecasts=_build_neural_forecasts(["AAPL"]),
        )

    results["base_plan"] = {
        "realized_pnl": base.simulation.realized_pnl,
        "fees": base.simulation.total_fees,
        "net_pnl": base.simulation.realized_pnl - base.simulation.total_fees,
        "ending_cash": base.simulation.ending_cash,
        "ending_equity": base.simulation.ending_equity,
        "num_trades": len(base.simulation.final_positions),
    }
    results["entry_takeprofit"] = entry_tp.simulation.summary(
        starting_nav=snapshot.cash, periods=1
    )
    results["maxdiff"] = maxdiff.simulation.summary(
        starting_nav=snapshot.cash, periods=1
    )
    results["neural"] = {
        "realized_pnl": neural.simulation.realized_pnl,
        "fees": neural.simulation.total_fees,
        "net_pnl": neural.simulation.realized_pnl - neural.simulation.total_fees,
        "ending_cash": neural.simulation.ending_cash,
        "ending_equity": neural.simulation.ending_equity,
    }
    return results


def render_markdown(summary: Mapping[str, object]) -> str:
    lines = ["# Baseline PnL Snapshot", ""]
    trade_hist = summary.get("trade_history", {})
    if isinstance(trade_hist, Mapping):
        lines.append("## Realised Trades")
        lines.append(f"- Total trades: {trade_hist.get('total_trades', 0)}")
        lines.append(f"- Total realised PnL: {trade_hist.get('total_realized_pnl', 0.0):.2f}")
        by_symbol = trade_hist.get("pnl_by_symbol", {})
        if isinstance(by_symbol, Mapping) and by_symbol:
            lines.append("")
            lines.append("| Symbol | PnL |")
            lines.append("| --- | ---: |")
            for symbol, pnl in sorted(by_symbol.items()):
                lines.append(f"| {symbol} | {pnl:.2f} |")
        lines.append("")

    snapshots = summary.get("trade_log", {}).get("snapshots") if isinstance(summary.get("trade_log"), Mapping) else None
    if isinstance(snapshots, Mapping) and snapshots.get("count"):
        lines.append("## Portfolio Snapshots")
        lines.append(f"- Entries: {snapshots['count']}")
        lines.append(f"- Exposure range: {snapshots['min_exposure']:.2f} → {snapshots['max_exposure']:.2f}")
        lines.append(f"- Latest exposure: {snapshots['latest_exposure']:.2f}")
        lines.append(f"- Latest risk threshold: {snapshots['latest_threshold']:.2f}x")
        if snapshots.get("start_timestamp") and snapshots.get("end_timestamp"):
            lines.append(
                f"- Span: {snapshots['start_timestamp']} → {snapshots['end_timestamp']} "
                f"({snapshots.get('duration_days', 0.0):.1f} days)"
            )
        lines.append("")

    deepseek = summary.get("deepseek", {})
    if isinstance(deepseek, Mapping):
        lines.append("## DeepSeek Benchmark")
        for name, payload in deepseek.items():
            if not isinstance(payload, Mapping):
                continue
            lines.append(f"- **{name}**: net PnL {payload.get('net_pnl', float('nan')):.4f}, "
                         f"realized {payload.get('realized_pnl', float('nan')):.4f}, "
                         f"fees {payload.get('fees', float('nan')):.4f}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    history = load_trade_history(TRADE_HISTORY_PATH)
    trade_hist_summary = summarise_trade_history(history)
    trade_log_summary = summarise_trade_log(TRADE_LOG_PATH)

    try:
        deepseek_summary = run_deepseek_benchmarks()
    except Exception as exc:  # noqa: BLE001
        deepseek_summary = {"error": str(exc)}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trade_history": trade_hist_summary,
        "trade_log": trade_log_summary,
        "deepseek": deepseek_summary,
    }

    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUTPUT_MARKDOWN.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
