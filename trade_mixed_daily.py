#!/usr/bin/env python3
"""
Mixed stock+crypto daily RL utility.

This script deliberately reuses the same calendar-day alignment as
`pufferlib_market.export_data_daily` so mixed backtests and live inference do
not drift away from the environment used for training.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from pufferlib_market.export_data_daily import export_binary
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy
from pufferlib_market.inference import PPOTrader, TradingSignal
from pufferlib_market.metrics import annualize_total_return
from src.mixed_daily_utils import align_daily_price_frames, latest_snapshot, summarize_symbol_coverage


DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "AMZN",
    "META",
    "TSLA",
    "PLTR",
    "NET",
    "JPM",
    "V",
    "SPY",
    "QQQ",
    "NFLX",
    "AMD",
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "LTCUSD",
    "AVAXUSD",
    "DOGEUSD",
    "LINKUSD",
    "AAVEUSD",
]

DEFAULT_CHECKPOINT = "pufferlib_market/checkpoints/mixed23_fresh_targeted/reg_combo_2/best.pt"
INITIAL_CASH = 10_000.0
REPO = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EffectiveSignal:
    as_of: str
    raw_action: str
    raw_symbol: Optional[str]
    raw_direction: Optional[str]
    raw_confidence: float
    raw_value_estimate: float
    allocation_pct: float
    level_offset_bps: float
    tradable_today: Optional[bool]
    effective_action: str
    effective_symbol: Optional[str]
    effective_direction: Optional[str]


def _load_trader(
    checkpoint: str,
    *,
    symbols: list[str],
    device: str,
) -> tuple[PPOTrader, bool]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    disable_shorts = bool(payload.get("disable_shorts", False)) if isinstance(payload, dict) else False
    trader = PPOTrader(checkpoint, device=device, long_only=disable_shorts, symbols=symbols)
    return trader, disable_shorts


def _expected_num_symbols_from_checkpoint(checkpoint: str) -> int | None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", {}) if isinstance(payload, dict) else {}
    weight = state.get("encoder.0.weight")
    if weight is None:
        weight = state.get("input_proj.weight")
    if weight is None or len(weight.shape) != 2:
        return None
    obs_size = int(weight.shape[1])
    remainder = obs_size - 5
    if remainder <= 0 or remainder % 17 != 0:
        return None
    return remainder // 17


def _checkpoint_hint_tokens(checkpoint: Path) -> set[str]:
    tokens: set[str] = set()
    for part in checkpoint.parts:
        for token in re.split(r"[^A-Za-z0-9]+", part.lower()):
            if token:
                tokens.add(token)
    return tokens


def _infer_symbols_from_local_mktd(checkpoint: str) -> list[str] | None:
    expected_num_symbols = _expected_num_symbols_from_checkpoint(checkpoint)
    if expected_num_symbols is None:
        return None

    data_dir = REPO / "pufferlib_market" / "data"
    if not data_dir.exists():
        return None

    checkpoint_tokens = _checkpoint_hint_tokens(Path(checkpoint))
    best: tuple[int, int, float, list[str]] | None = None
    for path in data_dir.glob("*.bin"):
        try:
            data = read_mktd(path)
        except Exception:
            continue
        if data.num_symbols != expected_num_symbols:
            continue

        stem_tokens = set(token for token in re.split(r"[^A-Za-z0-9]+", path.stem.lower()) if token)
        overlap = len(checkpoint_tokens & stem_tokens)
        val_bias = 1 if "val" in stem_tokens or "valid" in stem_tokens else 0
        score = (overlap, val_bias, float(path.stat().st_mtime), list(data.symbols))
        if best is None or score[:3] > best[:3]:
            best = score

    if best is None:
        return None
    return [str(sym).upper() for sym in best[3]]


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [str(sym).upper() for sym in args.symbols]
    inferred = _infer_symbols_from_local_mktd(args.checkpoint)
    if inferred:
        return inferred
    return list(DEFAULT_SYMBOLS)


def _mask_short_logits(logits: torch.Tensor, trader: PPOTrader) -> torch.Tensor:
    masked = logits.clone()
    masked[:, 1 + trader.side_block :] = torch.finfo(masked.dtype).min
    return masked


def _policy_action(trader: PPOTrader, obs, *, disable_shorts: bool, deterministic: bool) -> int:
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(trader.device)
    with torch.no_grad():
        logits, _ = trader.policy(obs_t)
        if disable_shorts:
            logits = _mask_short_logits(logits, trader)
        if deterministic:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())


def _apply_portfolio_state(
    trader: PPOTrader,
    *,
    cash: float,
    current_symbol: Optional[str],
    current_direction: Optional[str],
    entry_price: float,
    position_qty: float,
    hold_days: int,
) -> None:
    trader.cash = float(cash)
    trader.entry_price = float(entry_price)
    trader.position_qty = float(position_qty)
    trader.hold_hours = int(hold_days)
    trader.step = min(int(hold_days), trader.max_steps)
    trader.current_position = None
    if not current_symbol or current_direction not in {"long", "short"}:
        return
    sym_idx = trader.SYMBOLS.index(current_symbol.upper())
    trader.current_position = sym_idx if current_direction == "long" else trader.num_symbols + sym_idx


def _effective_signal(signal: TradingSignal, tradable_today: bool, current_position_active: bool, as_of: str) -> EffectiveSignal:
    effective_action = signal.action
    effective_symbol = signal.symbol
    effective_direction = signal.direction
    if signal.symbol and not tradable_today:
        if current_position_active:
            effective_action = "hold_closed_market"
            effective_symbol = None
            effective_direction = None
        else:
            effective_action = "flat_closed_market"
            effective_symbol = None
            effective_direction = None
    return EffectiveSignal(
        as_of=as_of,
        raw_action=signal.action,
        raw_symbol=signal.symbol,
        raw_direction=signal.direction,
        raw_confidence=float(signal.confidence),
        raw_value_estimate=float(signal.value_estimate),
        allocation_pct=float(signal.allocation_pct),
        level_offset_bps=float(signal.level_offset_bps),
        tradable_today=bool(tradable_today) if signal.symbol else None,
        effective_action=effective_action,
        effective_symbol=effective_symbol,
        effective_direction=effective_direction,
    )


def run_once(args: argparse.Namespace) -> EffectiveSignal:
    symbols = _resolve_symbols(args)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    trader, _ = _load_trader(args.checkpoint, symbols=symbols, device=device)
    aligned = align_daily_price_frames(
        symbols,
        data_root=args.data_root,
        end_date=args.end,
        min_days=max(args.min_days, 61),
    )
    snap = latest_snapshot(aligned)
    _apply_portfolio_state(
        trader,
        cash=args.cash,
        current_symbol=args.current_symbol,
        current_direction=args.current_direction,
        entry_price=args.entry_price,
        position_qty=args.position_qty,
        hold_days=args.hold_days,
    )
    signal = trader.get_signal(snap.feature_matrix, snap.prices)
    signal_tradable = True if signal.symbol is None else bool(snap.tradable[signal.symbol])
    result = _effective_signal(
        signal,
        tradable_today=signal_tradable,
        current_position_active=bool(trader.current_position is not None),
        as_of=snap.as_of.strftime("%Y-%m-%d"),
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(result), indent=2, sort_keys=True))
    return result


def run_backtest(args: argparse.Namespace) -> dict[str, object]:
    symbols = _resolve_symbols(args)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    trader, disable_shorts = _load_trader(args.checkpoint, symbols=symbols, device=device)

    with tempfile.TemporaryDirectory(prefix="mixed_daily_") as tmpdir:
        data_path = Path(tmpdir) / "window.bin"
        export_binary(
            symbols=symbols,
            data_root=Path(args.data_root),
            output_path=data_path,
            start_date=args.start,
            end_date=args.end,
            min_days=max(args.min_days, args.max_steps + 1),
        )
        data = read_mktd(data_path)

        def policy_fn(obs) -> int:
            return _policy_action(trader, obs, disable_shorts=disable_shorts, deterministic=not args.sample)

        result = simulate_daily_policy(
            data,
            policy_fn,
            max_steps=min(args.max_steps, data.num_timesteps - 1),
            fee_rate=args.fee_rate,
            max_leverage=args.max_leverage,
            periods_per_year=args.periods_per_year,
            short_borrow_apr=args.short_borrow_apr,
            initial_cash=INITIAL_CASH,
            action_allocation_bins=trader.action_allocation_bins,
            action_level_bins=trader.action_level_bins,
            action_max_offset_bps=trader.action_max_offset_bps,
            enable_drawdown_profit_early_exit=not args.disable_drawdown_profit_early_exit,
            drawdown_profit_early_exit_min_steps=args.drawdown_profit_early_exit_min_steps,
            drawdown_profit_early_exit_progress_fraction=args.drawdown_profit_early_exit_progress_fraction,
            early_exit_max_drawdown=args.early_exit_max_drawdown,
            early_exit_min_sortino=args.early_exit_min_sortino,
        )

    steps = min(args.max_steps, data.num_timesteps - 1)
    evaluated_steps = int(result.evaluated_steps) if int(result.evaluated_steps) > 0 else int(steps)
    report = {
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "symbols": symbols,
        "date_range": {"start": args.start, "end": args.end},
        "aligned_days": int(data.num_timesteps),
        "requested_max_steps": int(steps),
        "eval_steps": evaluated_steps,
        "total_return": float(result.total_return),
        "annualized_return": float(
            annualize_total_return(
                result.total_return,
                periods=max(evaluated_steps, 1),
                periods_per_year=args.periods_per_year,
            )
        ),
        "sortino": float(result.sortino),
        "max_drawdown": float(result.max_drawdown),
        "num_trades": int(result.num_trades),
        "win_rate": float(result.win_rate),
        "avg_hold_steps": float(result.avg_hold_steps),
        "stopped_early": bool(result.stopped_early),
        "stop_reason": str(result.stop_reason),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def run_audit(args: argparse.Namespace) -> list[dict[str, object]]:
    symbols = _resolve_symbols(args)
    rows = summarize_symbol_coverage(symbols, data_root=args.data_root)
    payload = [
        {
            "symbol": row.symbol,
            "first_date": row.first_date.strftime("%Y-%m-%d"),
            "last_date": row.last_date.strftime("%Y-%m-%d"),
            "num_rows": row.num_rows,
        }
        for row in rows
    ]
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mixed daily RL backtest and signal utility")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument(
        "--disable-drawdown-profit-early-exit",
        action="store_true",
        help="Disable the default drawdown-vs-profit screening exit in the daily replay.",
    )
    parser.add_argument(
        "--drawdown-profit-early-exit-min-steps",
        type=int,
        default=20,
        help="Minimum replay length before any early-exit screening rule can stop the run.",
    )
    parser.add_argument(
        "--drawdown-profit-early-exit-progress-fraction",
        type=float,
        default=0.5,
        help="Progress fraction that must elapse before any early-exit screening rule can stop the run.",
    )
    parser.add_argument(
        "--early-exit-max-drawdown",
        type=float,
        default=None,
        help="Optional max drawdown threshold, e.g. 0.25 to stop once running drawdown reaches 25%%.",
    )
    parser.add_argument(
        "--early-exit-min-sortino",
        type=float,
        default=None,
        help="Optional running Sortino threshold to stop obvious losers during long backtests.",
    )
    parser.add_argument("--cash", type=float, default=INITIAL_CASH)
    parser.add_argument("--current-symbol", default=None)
    parser.add_argument("--current-direction", choices=["long", "short"], default=None)
    parser.add_argument("--position-qty", type=float, default=0.0)
    parser.add_argument("--entry-price", type=float, default=0.0)
    parser.add_argument("--hold-days", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Emit one aligned daily signal")
    mode.add_argument("--backtest", action="store_true", help="Run a daily backtest on an exported temp window")
    mode.add_argument("--audit", action="store_true", help="Print first/last available dates for the selected symbols")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.audit:
        run_audit(args)
        return
    if args.once:
        run_once(args)
        return
    run_backtest(args)


if __name__ == "__main__":
    main()
