#!/usr/bin/env python3
"""Normalize Binance strategy eval artifacts into one head-to-head leaderboard.

The Binance research tree has several eval paths with different CSV schemas:
daily PPO checkpoints, deterministic rules, daily XGB rankers, XGB portfolio
packing, and hourly portfolio-pack rolling validation. This script reads those
artifacts, converts them into common monthly/risk/trade columns, and applies
the same rough promotion gates used in the progress notes.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_PATTERNS = (
    "pufferlib_market/checkpoints/binance33_1x_smooth/leaderboard_120d.csv",
    "analysis/binance33_rule*.csv",
    "analysis/binance33_linear_rule*.csv",
    "analysis/binance33_linear_cvar_pack*.csv",
    "analysis/binance33_linear_symbol_filter*.csv",
    "analysis/binance33_regime_gate*.csv",
    "analysis/binance33_easy_universe*.csv",
    "analysis/binance33_xgb_*.csv",
    "analysis/binance33_lgbm*.csv",
    "analysis/binance33_meta_anneal*.csv",
    "analysis/binance33_meta_channel_gate_grid*.csv",
    "analysis/binance33_xgb_portfolio_pack*.csv",
    "analysis/binance_pack_rolling_validation*_summary.csv",
    "analysis/binance_pack_current*_120d*.csv",
)


@dataclass(frozen=True)
class NormalizedRow:
    family: str
    source: str
    config: str
    eval_scope: str
    eval_days: int | None
    slip_bps: float | None
    median_monthly_pct: float
    risk_monthly_pct: float | None
    median_total_pct: float | None
    risk_total_pct: float | None
    neg_windows: int | None
    windows: int | None
    drawdown_pct: float | None
    trades: float | None
    pass_27: int
    pass_30: int
    score: float
    notes: str = ""


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _int(value: Any, default: int | None = None) -> int | None:
    out = _float(value)
    if out is None:
        return default
    return int(out)


def _monthly_from_total(total_pct: float | None, days: int | None) -> float | None:
    if total_pct is None or days is None or days <= 0:
        return None
    base = max(1e-9, 1.0 + float(total_pct) / 100.0)
    return 100.0 * (base ** (30.0 / float(days)) - 1.0)


def _get(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return None


def _config_label(row: dict[str, str], fallback: str) -> str:
    direct = _get(row, "config", "candidate_id", "experiment", "name")
    if direct:
        pieces = [direct]
        for key in (
            "max_leverage",
            "max_gross",
            "max_weight",
            "lookback_days",
            "cvar_alpha",
            "risk_power",
            "edge_power",
            "rebalance_style",
            "exclude_count",
            "exclude_symbols",
            "book_mode",
            "threshold",
            "market_gate",
            "channel",
            "max_hold_bars",
            "circuit_breaker_dd_pct",
            "fill_buffer_bps",
            "top_k",
            "top_n",
            "train_variant",
            "allocation_mode",
            "side_mode",
            "rebalance_days",
            "min_abs_score",
            "btc_min_return_20d",
            "gate",
            "scope",
            "short_borrow_apr",
            "label_horizon",
            "train_days",
            "rounds",
            "min_take_profit_bps",
        ):
            value = _get(row, key)
            if value:
                pieces.append(f"{key}={value}")
        return " ".join(pieces)
    return fallback


def _infer_family(path: Path, row: dict[str, str]) -> str:
    text = str(path).lower()
    if "leaderboard_120d" in text and "binance33_1x_smooth" in text:
        return "ppo_binance33_daily"
    if "linear_cvar_pack" in text:
        return "linear_cvar_pack"
    if "linear_symbol_filter" in text:
        return "linear_symbol_filter"
    if "rule" in text:
        if "linear_rule" in text:
            return "linear_rule_daily"
        return "rule_daily"
    if "regime_gate" in text:
        return "linear_rule_regime_gate"
    if "easy_universe" in text:
        return "xgb_easy_universe"
    if "portfolio_pack" in text and "binance33_xgb" in text:
        return "xgb_daily_pack"
    if "binance33_xgb" in text:
        return "xgb_daily"
    if "binance33_lgbm" in text:
        return "lgbm_daily"
    if "meta_anneal" in text:
        return "meta_anneal"
    if "meta_channel_gate_grid" in text:
        return "meta_channel_grid"
    if "rolling_validation" in text and text.endswith("_summary.csv"):
        return "hourly_pack_rolling"
    if "binance_pack_current" in text:
        side = str(row.get("side_mode", "")).strip().lower()
        return f"hourly_pack_current_{side or 'mixed'}"
    if "binance_pack" in text:
        return "hourly_pack"
    return "unknown"


def _window_counts(row: dict[str, str]) -> tuple[int | None, int | None]:
    neg = _int(_get(row, "neg_windows", "negative_windows", "negative_cells"))
    windows = _int(_get(row, "windows", "n_windows", "cells"))
    return neg, windows


def _drawdown(row: dict[str, str]) -> float | None:
    return _float(
        _get(
            row,
            "worst_max_drawdown_pct",
            "p90_dd_pct",
            "max_drawdown_pct",
            "median_max_drawdown_pct",
        )
    )


def _trades(row: dict[str, str]) -> float | None:
    return _float(_get(row, "min_num_sells", "median_trades", "num_sells", "total_trades"))


def _normalize_row(path: Path, row: dict[str, str], *, min_trades: int, min_windows: int) -> NormalizedRow | None:
    family = _infer_family(path, row)
    eval_days = _int(_get(row, "eval_days", "window_days"))
    slip_bps = _float(_get(row, "slip_bps", "fill_buffer_bps"))

    median_total = _float(_get(row, "median_pct", "median_total_return_pct", "total_return_pct"))
    risk_total = _float(_get(row, "p10_pct", "p10_total_return_pct", "worst_total_return_pct"))
    median_monthly = _float(_get(row, "median_monthly_pct", "median_monthly_return_pct", "monthly_return_pct"))
    risk_monthly = _float(_get(row, "p10_monthly_pct", "worst_monthly_pct", "worst_monthly_return_pct"))
    if median_monthly is None:
        median_monthly = _monthly_from_total(median_total, eval_days)
    if risk_monthly is None:
        risk_monthly = _monthly_from_total(risk_total, eval_days)
    if median_monthly is None:
        return None

    neg, windows = _window_counts(row)
    dd = _drawdown(row)
    trades = _trades(row)
    config = _config_label(row, path.stem)
    eval_scope = "rolling" if family.endswith("_rolling") else "single_slice"

    pass_27 = _passes_gate(
        median_monthly=median_monthly,
        risk_monthly=risk_monthly,
        neg_windows=neg,
        drawdown_pct=dd,
        trades=trades,
        windows=windows,
        target_monthly_pct=27.0,
        max_dd_pct=20.0,
        min_trades=min_trades,
        min_windows=min_windows,
    )
    pass_30 = _passes_gate(
        median_monthly=median_monthly,
        risk_monthly=risk_monthly,
        neg_windows=neg,
        drawdown_pct=dd,
        trades=trades,
        windows=windows,
        target_monthly_pct=30.0,
        max_dd_pct=20.0,
        min_trades=min_trades,
        min_windows=min_windows,
    )
    score = _score(
        median_monthly=median_monthly,
        risk_monthly=risk_monthly,
        neg_windows=neg,
        drawdown_pct=dd,
        trades=trades,
        min_trades=min_trades,
    )
    notes: list[str] = []
    if trades is not None and trades < min_trades:
        notes.append("under_trades")
    if windows is not None and windows < min_windows:
        notes.append("few_windows")
    if neg is not None and neg > 0:
        notes.append("negative_windows")
    if dd is not None and dd > 20.0:
        notes.append("high_dd")
    if risk_monthly is not None and risk_monthly < 0.0:
        notes.append("negative_tail")

    return NormalizedRow(
        family=family,
        source=str(path),
        config=config,
        eval_scope=eval_scope,
        eval_days=eval_days,
        slip_bps=slip_bps,
        median_monthly_pct=float(median_monthly),
        risk_monthly_pct=risk_monthly,
        median_total_pct=median_total,
        risk_total_pct=risk_total,
        neg_windows=neg,
        windows=windows,
        drawdown_pct=dd,
        trades=trades,
        pass_27=pass_27,
        pass_30=pass_30,
        score=score,
        notes=";".join(notes),
    )


def _passes_gate(
    *,
    median_monthly: float,
    risk_monthly: float | None,
    neg_windows: int | None,
    drawdown_pct: float | None,
    trades: float | None,
    windows: int | None,
    target_monthly_pct: float,
    max_dd_pct: float,
    min_trades: int,
    min_windows: int,
) -> int:
    if median_monthly < target_monthly_pct:
        return 0
    if risk_monthly is not None and risk_monthly < 0.0:
        return 0
    if neg_windows is not None and neg_windows > 0:
        return 0
    if drawdown_pct is not None and drawdown_pct > max_dd_pct:
        return 0
    if trades is not None and trades < min_trades:
        return 0
    if windows is not None and windows < min_windows:
        return 0
    return 1


def _score(
    *,
    median_monthly: float,
    risk_monthly: float | None,
    neg_windows: int | None,
    drawdown_pct: float | None,
    trades: float | None,
    min_trades: int,
) -> float:
    tail = risk_monthly if risk_monthly is not None else min(median_monthly, 0.0)
    dd = drawdown_pct if drawdown_pct is not None else 35.0
    neg = neg_windows if neg_windows is not None else 1
    trade_shortfall = max(0.0, float(min_trades) - float(trades or 0.0))
    return float(median_monthly + 0.75 * tail - 1.25 * dd - 18.0 * neg - 3.0 * trade_shortfall)


def _read_rows(path: Path, *, min_trades: int, min_windows: int) -> list[NormalizedRow]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except UnicodeDecodeError:
        return []
    out: list[NormalizedRow] = []
    for row in rows:
        norm = _normalize_row(path, row, min_trades=min_trades, min_windows=min_windows)
        if norm is not None:
            out.append(norm)
    return out


def _expand_patterns(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[") else [Path(pattern)]
        for path in matches:
            if path.is_file() and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def load_leaderboard(paths: Iterable[Path], *, min_trades: int, min_windows: int = 20) -> list[NormalizedRow]:
    rows: list[NormalizedRow] = []
    for path in paths:
        rows.extend(_read_rows(path, min_trades=min_trades, min_windows=min_windows))
    rows.sort(key=lambda row: row.score, reverse=True)
    return rows


def _write_csv(path: Path, rows: Sequence[NormalizedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(NormalizedRow.__dataclass_fields__)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", default=[], help="Additional CSV path or glob to include")
    parser.add_argument("--no-defaults", action="store_true", help="Only read paths passed via --csv")
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--min-windows", type=int, default=20)
    parser.add_argument("--top", type=int, default=30)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path, default=Path(f"analysis/binance_head_to_head_{stamp}.csv"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    patterns: list[str] = []
    if not args.no_defaults:
        patterns.extend(DEFAULT_PATTERNS)
    patterns.extend(args.csv)
    paths = _expand_patterns(patterns)
    rows = load_leaderboard(paths, min_trades=int(args.min_trades), min_windows=int(args.min_windows))
    _write_csv(args.out, rows)

    print(f"read {len(paths)} source files, normalized {len(rows)} rows")
    print(f"wrote {args.out}")
    print()
    print("=== Binance head-to-head ===")
    for row in rows[: int(args.top)]:
        risk = "n/a" if row.risk_monthly_pct is None else f"{row.risk_monthly_pct:+.2f}%"
        dd = "n/a" if row.drawdown_pct is None else f"{row.drawdown_pct:.2f}%"
        trades = "n/a" if row.trades is None else f"{row.trades:g}"
        print(
            f"{row.score:+8.2f} pass30={row.pass_30} pass27={row.pass_27} "
            f"{row.median_monthly_pct:+7.2f}%/mo risk={risk:>8} dd={dd:>8} "
            f"trades={trades:>5} {row.family} {row.config}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
