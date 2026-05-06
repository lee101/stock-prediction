#!/usr/bin/env python3
"""Greedy symbol-exclusion search for a Binance33 linear rule.

The goal is to test the "cut hard/bad pairs" hypothesis without selecting on
the validation set. We greedily choose excluded symbols by training-history
goodness, then score each accepted exclusion set on validation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, read_mktd
from scripts.search_binance33_linear_rules import _eval_rule
from scripts.search_binance33_regime_gates import _load_rule_from_csv


def _parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError(f"expected at least one float in {value!r}")
    return out


def _masked_data(data: MktdData, exclude_symbols: set[str]) -> MktdData:
    if not exclude_symbols:
        return data
    excluded = {symbol.upper() for symbol in exclude_symbols}
    tradable = (
        np.asarray(data.tradable, dtype=np.uint8).copy()
        if data.tradable is not None
        else np.ones((data.num_timesteps, data.num_symbols), dtype=np.uint8)
    )
    for idx, symbol in enumerate(data.symbols):
        if symbol.upper() in excluded:
            tradable[:, idx] = 0
    return MktdData(
        version=int(data.version),
        symbols=list(data.symbols),
        features=data.features,
        prices=data.prices,
        tradable=tradable,
    )


def _goodness(row: dict[str, float | int | str]) -> float:
    return float(
        float(row["median_monthly_pct"])
        + 0.75 * float(row["p10_monthly_pct"])
        - 1.25 * float(row["p90_dd_pct"])
        - 18.0 * int(row["neg_windows"])
    )


def _eval_filtered(
    data: MktdData,
    rule,
    *,
    exclude_symbols: set[str],
    scope: str,
    eval_days: int,
    stride: int,
    slippage_bps: float,
    fee_rate: float,
    fill_buffer_bps: float,
    decision_lag: int,
    max_leverage: float,
    periods_per_year: float,
    short_borrow_apr: float,
    target_monthly_pct: float,
    target_max_dd_pct: float,
) -> dict[str, float | int | str]:
    row = _eval_rule(
        _masked_data(data, exclude_symbols),
        rule,
        eval_days=int(eval_days),
        stride=int(stride),
        slippage_bps=float(slippage_bps),
        fee_rate=float(fee_rate),
        fill_buffer_bps=float(fill_buffer_bps),
        decision_lag=int(decision_lag),
        max_leverage=float(max_leverage),
        periods_per_year=float(periods_per_year),
        short_borrow_apr=float(short_borrow_apr),
        target_monthly_pct=float(target_monthly_pct),
        target_max_dd_pct=float(target_max_dd_pct),
    )
    row["scope"] = scope
    row["exclude_count"] = int(len(exclude_symbols))
    row["exclude_symbols"] = ",".join(sorted(exclude_symbols))
    row["goodness_score"] = _goodness(row)
    return row


def _fieldnames() -> list[str]:
    return [
        "scope",
        "config",
        "coeffs_json",
        "exclude_count",
        "exclude_symbols",
        "max_leverage",
        "rebalance_days",
        "min_abs_score",
        "btc_min_return_20d",
        "eval_days",
        "slip_bps",
        "fill_buffer_bps",
        "median_pct",
        "p10_pct",
        "p90_pct",
        "median_monthly_pct",
        "p10_monthly_pct",
        "worst_monthly_pct",
        "best_monthly_pct",
        "neg_windows",
        "windows",
        "p90_dd_pct",
        "median_smooth",
        "median_ulcer",
        "median_sortino",
        "median_trades",
        "target_monthly_pct",
        "target_max_dd_pct",
        "passes_target",
        "goodness_score",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-csv", type=Path, default=Path("analysis/binance33_linear_rule_search_20260430.csv"))
    parser.add_argument("--config", default="rand0045_rb1_thr0.5_btcmom20gt-0.05")
    parser.add_argument("--train-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_train.bin"))
    parser.add_argument("--eval-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_linear_symbol_filter.csv"))
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--search-leverage", type=float, default=1.25)
    parser.add_argument("--validate-leverage-grid", default="1.0,1.2,1.25,1.305,1.4")
    parser.add_argument("--max-exclude", type=int, default=6)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.08)
    parser.add_argument("--target-monthly-pct", type=float, default=30.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    args = parser.parse_args()

    rule = _load_rule_from_csv(args.rule_csv, args.config)
    train_data = read_mktd(args.train_data)
    val_data = read_mktd(args.eval_data)
    leverages = _parse_float_list(args.validate_leverage_grid)
    excluded: set[str] = set()
    rows: list[dict[str, float | int | str]] = []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_fieldnames())
        writer.writeheader()

        baseline_train = _eval_filtered(
            train_data,
            rule,
            exclude_symbols=excluded,
            scope="train_search",
            eval_days=int(args.eval_days),
            stride=int(args.train_stride),
            slippage_bps=float(args.slippage_bps),
            fee_rate=float(args.fee_rate),
            fill_buffer_bps=float(args.fill_buffer_bps),
            decision_lag=int(args.decision_lag),
            max_leverage=float(args.search_leverage),
            periods_per_year=float(args.periods_per_year),
            short_borrow_apr=float(args.short_borrow_apr),
            target_monthly_pct=float(args.target_monthly_pct),
            target_max_dd_pct=float(args.target_max_dd_pct),
        )
        rows.append(baseline_train)
        writer.writerow(baseline_train)
        fh.flush()
        best_train_goodness = float(baseline_train["goodness_score"])

        for leverage in leverages:
            val_row = _eval_filtered(
                val_data,
                rule,
                exclude_symbols=excluded,
                scope="val",
                eval_days=int(args.eval_days),
                stride=int(args.val_stride),
                slippage_bps=float(args.slippage_bps),
                fee_rate=float(args.fee_rate),
                fill_buffer_bps=float(args.fill_buffer_bps),
                decision_lag=int(args.decision_lag),
                max_leverage=float(leverage),
                periods_per_year=float(args.periods_per_year),
                short_borrow_apr=float(args.short_borrow_apr),
                target_monthly_pct=float(args.target_monthly_pct),
                target_max_dd_pct=float(args.target_max_dd_pct),
            )
            rows.append(val_row)
            writer.writerow(val_row)
            fh.flush()

        for step in range(1, int(args.max_exclude) + 1):
            candidates: list[dict[str, float | int | str]] = []
            for symbol in train_data.symbols:
                sym = symbol.upper()
                if sym in excluded:
                    continue
                candidate_excluded = set(excluded)
                candidate_excluded.add(sym)
                row = _eval_filtered(
                    train_data,
                    rule,
                    exclude_symbols=candidate_excluded,
                    scope=f"train_candidate_{step}",
                    eval_days=int(args.eval_days),
                    stride=int(args.train_stride),
                    slippage_bps=float(args.slippage_bps),
                    fee_rate=float(args.fee_rate),
                    fill_buffer_bps=float(args.fill_buffer_bps),
                    decision_lag=int(args.decision_lag),
                    max_leverage=float(args.search_leverage),
                    periods_per_year=float(args.periods_per_year),
                    short_borrow_apr=float(args.short_borrow_apr),
                    target_monthly_pct=float(args.target_monthly_pct),
                    target_max_dd_pct=float(args.target_max_dd_pct),
                )
                candidates.append(row)
            if not candidates:
                break
            candidates.sort(key=lambda item: float(item["goodness_score"]), reverse=True)
            best = candidates[0]
            if float(best["goodness_score"]) <= best_train_goodness:
                print(f"step {step}: no train improvement; stopping")
                break
            excluded = {part for part in str(best["exclude_symbols"]).split(",") if part}
            best_train_goodness = float(best["goodness_score"])
            best["scope"] = f"train_selected_{step}"
            rows.append(best)
            writer.writerow(best)
            fh.flush()
            print(
                f"step {step}: exclude={best['exclude_symbols']} "
                f"train_good={best['goodness_score']:+.2f} "
                f"train_mo={best['median_monthly_pct']:+.2f}% "
                f"dd90={best['p90_dd_pct']:.2f}% neg={best['neg_windows']}/{best['windows']}",
                flush=True,
            )

            for leverage in leverages:
                val_row = _eval_filtered(
                    val_data,
                    rule,
                    exclude_symbols=excluded,
                    scope=f"val_selected_{step}",
                    eval_days=int(args.eval_days),
                    stride=int(args.val_stride),
                    slippage_bps=float(args.slippage_bps),
                    fee_rate=float(args.fee_rate),
                    fill_buffer_bps=float(args.fill_buffer_bps),
                    decision_lag=int(args.decision_lag),
                    max_leverage=float(leverage),
                    periods_per_year=float(args.periods_per_year),
                    short_borrow_apr=float(args.short_borrow_apr),
                    target_monthly_pct=float(args.target_monthly_pct),
                    target_max_dd_pct=float(args.target_max_dd_pct),
                )
                rows.append(val_row)
                writer.writerow(val_row)
                fh.flush()

    val_rows = [row for row in rows if str(row["scope"]).startswith("val")]
    val_rows.sort(key=lambda item: float(item["goodness_score"]), reverse=True)
    print("\n=== Best validation filtered linear rules ===")
    for row in val_rows[:20]:
        print(
            f"good={row['goodness_score']:+7.2f} {row['median_monthly_pct']:+7.2f}%/mo "
            f"p10={row['p10_monthly_pct']:+7.2f}% dd90={row['p90_dd_pct']:.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} pass={row['passes_target']} "
            f"lev={row['max_leverage']} excl={row['exclude_symbols'] or 'none'}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
