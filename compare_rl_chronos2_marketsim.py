#!/usr/bin/env python3
"""Compare daily RL and Chronos2 marketsim runs with visual artifacts.

This is intended for stock daily experiments where:
1. RL is evaluated through ``pufferlib_market.hourly_replay.simulate_daily_policy``.
2. Chronos2 is evaluated through ``pnlforecast.comparison_v2.RigorousBacktester``.
3. The script writes a JSON summary, an equity-curve PNG, and optional RL
   marketsim video / Plotly HTML artifacts for fast visual inspection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch
    from marketsimlong.config import DataConfigLong
    from marketsimlong.data import DailyDataLoader
    from pnlforecast.comparison_v2 import RigorousBacktester
    from pufferlib_market.hourly_replay import MktdData, Position
    from src.marketsim_video import MarketsimTrace


DEFAULT_RL_CHECKPOINT = "pufferlib_market/prod_ensemble/tp10.pt"
DEFAULT_RL_DATA_PATH = "pufferlib_market/data/stocks12_daily_val.bin"
DEFAULT_MAX_STEPS = 120
DEFAULT_CHRONOS_WARMUP_DAYS = 120
DEFAULT_TRACE_NUM_PAIRS = 4
DEFAULT_VIDEO_FPS = 1
DEFAULT_EQUITY_PLOT_DPI = 160
DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "META",
    "TSLA",
    "SPY",
    "QQQ",
    "PLTR",
    "JPM",
    "V",
    "AMZN",
]
DEFAULT_OUTPUT_ROOT = "marketsim_compare_outputs"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from pnlforecast.comparison_v2 import ExitMode

    parser = argparse.ArgumentParser(description="Compare RL and Chronos2 daily marketsim runs.")
    parser.add_argument("--rl-checkpoint", default=DEFAULT_RL_CHECKPOINT)
    parser.add_argument("--rl-data-path", default=DEFAULT_RL_DATA_PATH)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--end-date", required=True, help="Inclusive UTC date, e.g. 2026-04-08")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive UTC date; when provided, the date span determines max-steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Sim days when --start-date is omitted; otherwise must match the start/end span. "
            f"Defaults to {DEFAULT_MAX_STEPS} when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory. Defaults to an auto-generated per-run path under marketsim_compare_outputs/.",
    )
    parser.add_argument("--rl-max-leverage", type=float, default=2.0)
    parser.add_argument("--chronos-leverage", type=float, default=2.0)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--periods-per-year", type=float, default=252.0)
    parser.add_argument("--chronos-top-n", type=int, default=2)
    parser.add_argument(
        "--chronos-warmup-days",
        type=int,
        default=DEFAULT_CHRONOS_WARMUP_DAYS,
        help=(
            "Extra historical days to load before --start-date for Chronos indicators and rolling state. "
            f"Defaults to {DEFAULT_CHRONOS_WARMUP_DAYS}."
        ),
    )
    parser.add_argument(
        "--chronos-exit-mode",
        choices=[mode.value for mode in ExitMode],
        default=ExitMode.HOLD_WITH_ADAPTIVE_TARGET.value,
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    return parser.parse_args(argv)


def _normalize_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    if not normalized:
        raise ValueError("At least one symbol is required")
    return tuple(normalized)


def _resolve_date_window(args: argparse.Namespace) -> tuple[date, date, int]:
    end_day = date.fromisoformat(str(args.end_date))
    if args.start_date:
        start_day = date.fromisoformat(str(args.start_date))
        max_steps = (end_day - start_day).days
        if max_steps < 0:
            raise ValueError(f"start_date {start_day} is after end_date {end_day}")
        requested_steps = None if args.max_steps is None else int(args.max_steps)
        if requested_steps is not None and requested_steps != max_steps:
            raise ValueError(
                f"start_date/end_date span is {max_steps} days but --max-steps was {requested_steps}; "
                f"either omit --start-date or set --max-steps {max_steps}"
            )
    else:
        max_steps = DEFAULT_MAX_STEPS if args.max_steps is None else int(args.max_steps)
        if max_steps < 0:
            raise ValueError(f"max_steps must be non-negative, got {max_steps}")
        start_day = end_day - timedelta(days=max_steps)
    return start_day, end_day, int(max_steps)


def _slug_token(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(value).strip().upper()).strip("-")
    return text or "NA"


def _symbol_label(symbols: Iterable[str], *, max_symbols: int = 3) -> str:
    normalized = [str(sym).strip().upper() for sym in symbols if str(sym).strip()]
    if not normalized:
        return "NO-SYMBOLS"
    head = normalized[:max_symbols]
    label = "_".join(_slug_token(sym) for sym in head)
    extra = len(normalized) - len(head)
    if extra > 0:
        label = f"{label}__PLUS{extra}"
    return label


def _resolve_output_dir(
    requested: str | None,
    *,
    start_day: date,
    end_day: date,
    symbols: Iterable[str],
    max_steps: int,
    root: str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    if requested:
        return Path(requested)

    base_name = (
        f"{start_day.isoformat()}__{end_day.isoformat()}__"
        f"{int(max_steps)}d__{_symbol_label(symbols)}"
    )
    root_path = Path(root)
    candidate = root_path / base_name
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        numbered = root_path / f"{base_name}__run{suffix}"
        if not numbered.exists():
            return numbered
        suffix += 1


def _reserve_output_dir(
    requested: str | None,
    *,
    start_day: date,
    end_day: date,
    symbols: Iterable[str],
    max_steps: int,
    root: str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    if requested:
        explicit = Path(requested)
        explicit.mkdir(parents=True, exist_ok=False)
        return explicit

    Path(root).mkdir(parents=True, exist_ok=True)
    while True:
        candidate = _resolve_output_dir(
            None,
            start_day=start_day,
            end_day=end_day,
            symbols=symbols,
            max_steps=max_steps,
            root=root,
        )
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue


def _should_render_rl_trace(*, skip_video: bool, skip_html: bool) -> bool:
    return not skip_video or not skip_html


def _safe_write_json(path: Path, payload: dict[str, object]) -> Path | None:
    from binance_worksteal.reporting import safe_write_summary_json

    written_path, error = safe_write_summary_json(path, payload)
    if error:
        print(f"WARN: failed to write JSON to {path}: {error}", file=sys.stderr)
        return None
    return written_path


def _build_failure_payload(
    *,
    args: argparse.Namespace,
    output_dir: Path | None,
    symbols: Iterable[str],
    start_day: date | None,
    end_day: date | None,
    max_steps: int | None,
    exc: Exception,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "rl_checkpoint": str(args.rl_checkpoint),
        "rl_data_path": str(args.rl_data_path),
        "symbols": list(symbols),
        "output_dir": str(output_dir) if output_dir is not None else None,
    }
    if start_day is not None and end_day is not None and max_steps is not None:
        payload["window"] = {
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "max_steps": int(max_steps),
        }
    return payload


def _tail_for_window(data: "MktdData", max_steps: int) -> "MktdData":
    from pufferlib_market.evaluate_tail import _slice_tail

    if int(data.num_timesteps) < int(max_steps) + 1:
        raise ValueError(
            f"RL data has {data.num_timesteps} timesteps; need at least {max_steps + 1} for a {max_steps}-day sim."
        )
    return _slice_tail(data, steps=int(max_steps))


def _daily_ohlc(data: "MktdData") -> np.ndarray:
    return np.stack(
        [
            data.prices[:, :, 0],
            data.prices[:, :, 1],
            data.prices[:, :, 2],
            data.prices[:, :, 3],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _clone_position(pos: "Position | None") -> "Position | None":
    from pufferlib_market.hourly_replay import Position

    if pos is None:
        return None
    return Position(
        sym=int(pos.sym),
        is_short=bool(pos.is_short),
        qty=float(pos.qty),
        entry_price=float(pos.entry_price),
    )


def _trace_from_daily_sim(
    *,
    data: "MktdData",
    result,
    dates: pd.DatetimeIndex,
) -> "MarketsimTrace":
    from src.marketsim_video import MarketsimTrace, OrderTick

    close = data.prices[:, :, 3].astype(np.float32, copy=False)
    ohlc = _daily_ohlc(data)
    trace = MarketsimTrace(symbols=[str(sym).upper() for sym in data.symbols], prices=close, prices_ohlc=ohlc)
    positions = list(getattr(result, "position_history", []) or [])
    equity_curve = np.asarray(getattr(result, "equity_curve", []), dtype=np.float64)
    total_steps = min(int(result.evaluated_steps or len(positions)), len(dates) - 1, len(close) - 1)
    previous: Position | None = None

    for step in range(total_steps):
        pos = _clone_position(positions[step]) if step < len(positions) else None
        orders = []
        if pos is not None and (
            previous is None
            or previous.sym != pos.sym
            or previous.is_short != pos.is_short
            or abs(previous.qty - pos.qty) > 1e-9
        ):
            orders.append(
                OrderTick(
                    sym=int(pos.sym),
                    price=float(pos.entry_price),
                    is_short=bool(pos.is_short),
                )
            )
        pos_sym = -1 if pos is None else int(pos.sym)
        pos_short = bool(pos.is_short) if pos is not None else False
        equity = float(equity_curve[step + 1]) if step + 1 < len(equity_curve) else float("nan")
        trace.record(
            step=step,
            action_id=int(result.actions[step]) if step < len(result.actions) else 0,
            position_sym=pos_sym,
            position_is_short=pos_short,
            equity=equity,
            orders=orders,
        )
        previous = pos
    return trace


def _build_chronos_equity_series(result) -> pd.Series:
    states = list(getattr(result, "daily_states", []) or [])
    if not states:
        return pd.Series(dtype=np.float64)
    idx = pd.DatetimeIndex([pd.Timestamp(state.date, tz="UTC") for state in states])
    values = np.asarray([float(state.total_equity) for state in states], dtype=np.float64)
    return pd.Series(values, index=idx, name="chronos2_equity")


def _plot_equity_curves(
    *,
    rl_equity: pd.Series,
    chronos_equity: pd.Series,
    out_path: Path,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    if not rl_equity.empty:
        rl_equity.plot(ax=ax, label="RL", linewidth=2.0, color="#0f766e")
    if not chronos_equity.empty:
        chronos_equity.plot(ax=ax, label="Chronos2", linewidth=2.0, color="#b45309")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_EQUITY_PLOT_DPI)
    plt.close(fig)
    return out_path


def _loaded_symbols(symbols: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    stock_symbols = tuple(str(sym).upper() for sym in symbols if not str(sym).upper().endswith("USD"))
    crypto_symbols = tuple(str(sym).upper() for sym in symbols if str(sym).upper().endswith("USD"))
    return stock_symbols, crypto_symbols


def _chronos_history_start_date(start_day: date, warmup_days: int) -> date:
    resolved_warmup = int(warmup_days)
    if resolved_warmup < 0:
        raise ValueError(f"chronos_warmup_days must be non-negative, got {resolved_warmup}")
    return start_day - timedelta(days=resolved_warmup)


def _run_comparison(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    start_day: date,
    end_day: date,
    max_steps: int,
    symbols: tuple[str, ...],
) -> dict[str, object]:
    import torch
    from marketsimlong.config import DataConfigLong
    from marketsimlong.data import DailyDataLoader
    from pnlforecast.comparison_v2 import ExitMode, RigorousBacktester
    from pufferlib_market.evaluate_multiperiod import load_policy, make_policy_fn
    from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy
    from src.marketsim_video import render_html_plotly, render_mp4

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    rl_data_full = read_mktd(Path(args.rl_data_path))
    rl_data = _tail_for_window(rl_data_full, max_steps=max_steps)
    features_per_sym = int(rl_data.features.shape[2])
    loaded = load_policy(
        args.rl_checkpoint,
        rl_data.num_symbols,
        arch="auto",
        hidden_size=None,
        device=device,
        features_per_sym=features_per_sym,
    )
    rl_policy_fn = make_policy_fn(
        loaded.policy,
        num_symbols=rl_data.num_symbols,
        deterministic=True,
        device=device,
    )
    rl_result = simulate_daily_policy(
        rl_data,
        rl_policy_fn,
        max_steps=max_steps,
        fee_rate=float(args.fee_rate),
        fill_buffer_bps=float(args.fill_buffer_bps),
        max_leverage=float(args.rl_max_leverage),
        periods_per_year=float(args.periods_per_year),
        action_allocation_bins=int(loaded.action_allocation_bins),
        action_level_bins=int(loaded.action_level_bins),
        action_max_offset_bps=float(loaded.action_max_offset_bps),
        enable_drawdown_profit_early_exit=False,
    )

    stock_symbols, crypto_symbols = _loaded_symbols(symbols)
    data_loader = DailyDataLoader(
        DataConfigLong(
            stock_symbols=stock_symbols,
            crypto_symbols=crypto_symbols,
            start_date=_chronos_history_start_date(start_day, int(args.chronos_warmup_days)),
            end_date=end_day,
        )
    )
    data_loader.load_all_symbols()
    chronos_backtester = RigorousBacktester(
        data_loader,
        initial_capital=float(args.initial_capital),
        leverage=float(args.chronos_leverage),
    )
    chronos_result = chronos_backtester.run_chronos_full_v2(
        list(symbols),
        start_day,
        end_day,
        exit_mode=ExitMode(str(args.chronos_exit_mode)),
        top_n=int(args.chronos_top_n),
    )

    rl_dates = pd.date_range(pd.Timestamp(start_day, tz="UTC"), pd.Timestamp(end_day, tz="UTC"), freq="D")
    rl_equity_values = np.asarray(getattr(rl_result, "equity_curve", []), dtype=np.float64)
    rl_equity = pd.Series(rl_equity_values[: len(rl_dates)], index=rl_dates[: len(rl_equity_values)], name="rl_equity")
    chronos_equity = _build_chronos_equity_series(chronos_result)

    summary = {
        "window": {
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "max_steps": int(max_steps),
        },
        "artifacts_dir": str(output_dir),
        "rl": {
            "checkpoint": str(args.rl_checkpoint),
            "data_path": str(args.rl_data_path),
            "symbols": [str(sym).upper() for sym in rl_data.symbols],
            "arch": loaded.arch,
            "hidden_size": int(loaded.hidden_size),
            "action_allocation_bins": int(loaded.action_allocation_bins),
            "action_level_bins": int(loaded.action_level_bins),
            "action_max_offset_bps": float(loaded.action_max_offset_bps),
            "max_leverage": float(args.rl_max_leverage),
            "result": {
                "total_return": float(rl_result.total_return),
                "sortino": float(rl_result.sortino),
                "max_drawdown": float(rl_result.max_drawdown),
                "num_trades": int(rl_result.num_trades),
                "win_rate": float(rl_result.win_rate),
                "avg_hold_steps": float(rl_result.avg_hold_steps),
            },
        },
        "chronos2": {
            "symbols": list(symbols),
            "exit_mode": str(args.chronos_exit_mode),
            "top_n": int(args.chronos_top_n),
            "leverage": float(args.chronos_leverage),
            "warmup_days": int(args.chronos_warmup_days),
            "result": {
                "strategy_name": str(chronos_result.strategy_name),
                "total_return_pct": float(chronos_result.total_return_pct),
                "annualized_return_pct": float(chronos_result.annualized_return_pct),
                "sharpe_ratio": float(chronos_result.sharpe_ratio),
                "max_drawdown_pct": float(chronos_result.max_drawdown_pct),
                "win_rate": float(chronos_result.win_rate),
                "total_trades": int(chronos_result.total_trades),
            },
        },
    }

    summary_path = output_dir / "summary.json"
    _safe_write_json(summary_path, summary)

    equity_plot_path = _plot_equity_curves(
        rl_equity=rl_equity,
        chronos_equity=chronos_equity,
        out_path=output_dir / "equity_curves.png",
        title=(
            f"RL vs Chronos2 Daily Marketsim "
            f"({start_day.isoformat()} to {end_day.isoformat()}, {int(max_steps)} steps)"
        ),
    )

    artifact_paths: dict[str, str] = {"summary_json": str(summary_path), "equity_plot": str(equity_plot_path)}

    should_render_trace = _should_render_rl_trace(skip_video=bool(args.skip_video), skip_html=bool(args.skip_html))
    rl_trace = _trace_from_daily_sim(data=rl_data, result=rl_result, dates=rl_dates) if should_render_trace else None
    if not args.skip_video and rl_trace is not None:
        video_path = output_dir / "rl_marketsim.mp4"
        render_mp4(
            rl_trace,
            video_path,
            num_pairs=min(DEFAULT_TRACE_NUM_PAIRS, rl_data.num_symbols),
            fps=DEFAULT_VIDEO_FPS,
            title=f"RL marketsim {start_day.isoformat()} to {end_day.isoformat()}",
        )
        artifact_paths["rl_video"] = str(video_path)
    if not args.skip_html and rl_trace is not None:
        html_path = output_dir / "rl_marketsim.html"
        render_html_plotly(
            rl_trace,
            html_path,
            title=f"RL marketsim {start_day.isoformat()} to {end_day.isoformat()}",
            animated=True,
        )
        artifact_paths["rl_html"] = str(html_path)

    artifact_manifest = output_dir / "artifacts.json"
    _safe_write_json(artifact_manifest, artifact_paths)
    return {"summary": summary, "artifacts": artifact_paths}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir: Path | None = None
    symbols: tuple[str, ...] = ()
    start_day: date | None = None
    end_day: date | None = None
    max_steps: int | None = None
    try:
        start_day, end_day, max_steps = _resolve_date_window(args)
        symbols = _normalize_symbols(args.symbols)
        output_dir = _reserve_output_dir(
            args.output_dir,
            start_day=start_day,
            end_day=end_day,
            symbols=symbols,
            max_steps=max_steps,
        )
        payload = _run_comparison(
            args=args,
            output_dir=output_dir,
            start_day=start_day,
            end_day=end_day,
            max_steps=max_steps,
            symbols=symbols,
        )
    except Exception as exc:
        failure_payload = _build_failure_payload(
            args=args,
            output_dir=output_dir,
            symbols=symbols,
            start_day=start_day,
            end_day=end_day,
            max_steps=max_steps,
            exc=exc,
        )
        if output_dir is not None:
            failure_path = output_dir / "failure.json"
            written = _safe_write_json(failure_path, failure_payload)
            if written is not None:
                print(f"ERROR: {exc}\nFailure summary: {written}", file=sys.stderr)
            else:
                print(f"ERROR: {exc}", file=sys.stderr)
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
